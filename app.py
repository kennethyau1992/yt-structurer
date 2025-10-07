import os
import re
import json
import time
import hashlib
import sqlite3
import gzip
import base64
import secrets
import functools
import asyncio
import aiohttp
import bcrypt
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from contextlib import contextmanager
import threading
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential

import streamlit as st

# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Thread-safe database manager with connection pooling"""
    
    _local = threading.local()
    
    def __init__(self, db_path: str = "user_data.db"):
        self.db_path = db_path
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get thread-local connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                isolation_level='DEFERRED'
            )
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception as e:
            self._local.conn.rollback()
            raise e
    
    def _init_database(self):
        """Initialize database schema with indexes"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    username TEXT PRIMARY KEY,
                    language TEXT DEFAULT 'English',
                    use_asr_fallback BOOLEAN DEFAULT 1,
                    deepseek_model TEXT DEFAULT 'deepseek-reasoner',
                    temperature REAL DEFAULT 0.1,
                    browser_for_cookies TEXT DEFAULT 'none',
                    performance_mode TEXT DEFAULT 'balanced',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    username TEXT,
                    service TEXT,
                    api_key TEXT,
                    PRIMARY KEY (username, service),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    url TEXT NOT NULL,
                    video_title TEXT,
                    video_channel TEXT,
                    video_description TEXT,
                    video_duration TEXT,
                    video_thumbnail TEXT,
                    video_published TEXT,
                    language TEXT,
                    model_used TEXT,
                    performance_mode TEXT,
                    status TEXT,
                    transcript_length INTEGER,
                    processing_time TEXT,
                    input_type TEXT DEFAULT 'url',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    history_id TEXT PRIMARY KEY,
                    executive_summary BLOB,
                    detailed_transcript BLOB,
                    error_message TEXT,
                    FOREIGN KEY (history_id) REFERENCES history(id) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    username TEXT PRIMARY KEY,
                    token TEXT NOT NULL,
                    expires_at DATETIME NOT NULL,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_username ON history(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_status ON history(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_search ON history(video_title, video_channel)")
            
            # Create default admin user if not exists
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        with self.get_connection() as conn:
            result = conn.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
            if not result:
                # Use bcrypt for password hashing
                salt = bcrypt.gensalt(rounds=12)
                password_hash = bcrypt.hashpw("admin123".encode('utf-8'), salt).decode('utf-8')
                
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    ("admin", password_hash)
                )
                
                # Add API keys from environment
                api_keys = {
                    'supadata': os.getenv("ADMIN_SUPADATA_KEY", ""),
                    'assemblyai': os.getenv("ADMIN_ASSEMBLYAI_KEY", ""),
                    'deepseek': os.getenv("ADMIN_DEEPSEEK_KEY", ""),
                    'youtube': os.getenv("ADMIN_YOUTUBE_KEY", "")
                }
                
                for service, key in api_keys.items():
                    if key:
                        conn.execute(
                            "INSERT INTO api_keys (username, service, api_key) VALUES (?, ?, ?)",
                            ("admin", service, key)
                        )

# ==================== HTTP CLIENT WITH CONNECTION POOLING ====================

class OptimizedHTTPClient:
    """HTTP client with connection pooling and retry logic"""
    
    _session = None
    _lock = threading.Lock()
    
    @classmethod
    def get_session(cls):
        """Get singleton session with connection pooling"""
        if cls._session is None:
            with cls._lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
                    )
                    
                    adapter = HTTPAdapter(
                        max_retries=retry_strategy,
                        pool_connections=10,
                        pool_maxsize=20,
                        pool_block=False
                    )
                    
                    cls._session.mount("http://", adapter)
                    cls._session.mount("https://", adapter)
        
        return cls._session

# ==================== YOUTUBE INFO EXTRACTOR ====================

class OptimizedYouTubeInfoExtractor:
    """Optimized YouTube info extractor with caching"""
    
    # Pre-compiled regex patterns
    _patterns = [
        re.compile(r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)'),
        re.compile(r'youtube\.com/v/([^&\n?#]+)')
    ]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    @functools.lru_cache(maxsize=500)
    def extract_video_id(self, url: str) -> Optional[str]:
        """Cached video ID extraction"""
        for pattern in self._patterns:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_video_info(_self, url: str) -> Dict[str, str]:
        """Cached video info retrieval"""
        video_id = _self.extract_video_id(url)
        if not video_id:
            return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}
        
        if _self.api_key:
            api_info = _self._get_info_from_api_cached(video_id)
            if api_info:
                return api_info
        
        return _self._get_info_from_ytdlp(url)
    
    @functools.lru_cache(maxsize=1000)
    def _get_info_from_api_cached(self, video_id: str) -> Optional[Dict[str, str]]:
        """Cached API call"""
        try:
            params = {
                'part': 'snippet,contentDetails',
                'id': video_id,
                'key': self.api_key,
                'fields': 'items(snippet(title,description,channelTitle,publishedAt,thumbnails/medium/url),contentDetails/duration)'
            }
            
            response = OptimizedHTTPClient.get_session().get(
                f"{self.base_url}/videos",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    item = data['items'][0]
                    snippet = item.get('snippet', {})
                    content_details = item.get('contentDetails', {})
                    
                    description = snippet.get('description', '')
                    truncated_desc = (description[:500] + "...") if len(description) > 500 else description
                    
                    return {
                        "title": snippet.get('title', 'Unknown Video'),
                        "description": truncated_desc,
                        "duration": content_details.get('duration', ''),
                        "channel": snippet.get('channelTitle', ''),
                        "published_at": snippet.get('publishedAt', ''),
                        "thumbnail": snippet.get('thumbnails', {}).get('medium', {}).get('url', '')
                    }
        except Exception:
            pass
        
        return None
    
    def _get_info_from_ytdlp(self, url: str) -> Dict[str, str]:
        """Fallback to yt-dlp"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                description = info.get('description', '') or ''
                truncated_desc = (description[:500] + "...") if len(description) > 500 else description
                
                return {
                    "title": info.get('title', 'Unknown Video'),
                    "description": truncated_desc,
                    "duration": str(info.get('duration', '')),
                    "channel": info.get('uploader', ''),
                    "published_at": info.get('upload_date', ''),
                    "thumbnail": info.get('thumbnail', '')
                }
        except Exception:
            pass
            
        return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}

# ==================== USER DATA MANAGER ====================

class OptimizedUserDataManager:
    """Optimized user data manager using SQLite"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.youtube_extractor = None
    
    def set_youtube_extractor(self, youtube_api_key: str = None):
        self.youtube_extractor = OptimizedYouTubeInfoExtractor(youtube_api_key)
    
    def get_user_settings(self, username: str) -> Dict:
        """Fast settings retrieval"""
        with self.db.get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM user_settings WHERE username = ?", 
                (username,)
            ).fetchone()
            
            if result:
                return dict(result)
            else:
                default_settings = {
                    "language": "English",
                    "use_asr_fallback": True,
                    "deepseek_model": "deepseek-reasoner",
                    "temperature": 0.1,
                    "browser_for_cookies": "none",
                    "performance_mode": "balanced"
                }
                self.save_user_settings(username, default_settings)
                return default_settings
    
    def save_user_settings(self, username: str, settings: Dict):
        """Atomic settings update"""
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO user_settings (username, language, use_asr_fallback, 
                    deepseek_model, temperature, browser_for_cookies, performance_mode, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    language = excluded.language,
                    use_asr_fallback = excluded.use_asr_fallback,
                    deepseek_model = excluded.deepseek_model,
                    temperature = excluded.temperature,
                    browser_for_cookies = excluded.browser_for_cookies,
                    performance_mode = excluded.performance_mode,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                username, 
                settings.get("language", "English"),
                settings.get("use_asr_fallback", True),
                settings.get("deepseek_model", "deepseek-reasoner"),
                settings.get("temperature", 0.1),
                settings.get("browser_for_cookies", "none"),
                settings.get("performance_mode", "balanced")
            ))
    
    @staticmethod
    def _compress(text: str) -> bytes:
        """Compress text for storage"""
        if not text:
            return b''
        return gzip.compress(text.encode('utf-8'))
    
    @staticmethod
    def _decompress(data: bytes) -> str:
        """Decompress text from storage"""
        if not data:
            return ''
        return gzip.decompress(data).decode('utf-8')
    
    def add_to_history(self, username: str, entry: Dict):
        """Optimized history insertion with compression"""
        if 'id' not in entry:
            entry['id'] = hashlib.md5(
                f"{entry.get('url', '')}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]
        
        # Enhance with YouTube info if available
        if self.youtube_extractor and entry.get('url') and entry.get('input_type', 'url') == 'url':
            video_info = self.youtube_extractor.get_video_info(entry['url'])
            entry.update({
                "video_title": video_info.get("title", entry.get("title", "Unknown Video")),
                "video_description": video_info.get("description", ""),
                "video_duration": video_info.get("duration", ""),
                "video_channel": video_info.get("channel", ""),
                "video_published": video_info.get("published_at", ""),
                "video_thumbnail": video_info.get("thumbnail", "")
            })
        
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO history (
                    id, username, url, video_title, video_channel, video_description,
                    video_duration, video_thumbnail, video_published, language, model_used,
                    performance_mode, status, transcript_length, processing_time, input_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry['id'], username, entry.get('url', ''),
                entry.get('video_title', 'Unknown'),
                entry.get('video_channel', ''),
                entry.get('video_description', '')[:500],
                entry.get('video_duration', ''),
                entry.get('video_thumbnail', ''),
                entry.get('video_published', ''),
                entry.get('language', 'English'),
                entry.get('model_used', ''),
                entry.get('performance_mode', 'balanced'),
                entry.get('status', 'Processing'),
                entry.get('transcript_length', 0),
                entry.get('processing_time', '0s'),
                entry.get('input_type', 'url')
            ))
            
            conn.execute("""
                INSERT INTO transcripts (history_id, executive_summary, detailed_transcript, error_message)
                VALUES (?, ?, ?, ?)
            """, (
                entry['id'],
                self._compress(entry.get('executive_summary', '')),
                self._compress(entry.get('detailed_transcript', '')),
                entry.get('error', '')
            ))
    
    def update_history_entry(self, username: str, entry_id: str, updates: Dict):
        """Efficient partial update"""
        with self.db.get_connection() as conn:
            metadata_fields = ['status', 'transcript_length', 'processing_time', 
                             'video_title', 'video_channel', 'language', 'model_used']
            
            update_parts = []
            values = []
            for field in metadata_fields:
                if field in updates:
                    update_parts.append(f"{field} = ?")
                    values.append(updates[field])
            
            if update_parts:
                values.extend([username, entry_id])
                conn.execute(f"""
                    UPDATE history SET {', '.join(update_parts)}
                    WHERE username = ? AND id = ?
                """, values)
            
            transcript_updates = []
            transcript_values = []
            
            for field in ['executive_summary', 'detailed_transcript', 'error']:
                if field in updates:
                    db_field = 'error_message' if field == 'error' else field
                    transcript_updates.append(f"{db_field} = ?")
                    
                    if field == 'error':
                        transcript_values.append(updates[field])
                    else:
                        transcript_values.append(self._compress(updates[field]))
            
            if transcript_updates:
                transcript_values.append(entry_id)
                conn.execute(f"""
                    UPDATE transcripts SET {', '.join(transcript_updates)}
                    WHERE history_id = ?
                """, transcript_values)
    
    def get_user_history_paginated(self, username: str, limit: int = 20, 
                                   offset: int = 0, filters: Dict = None) -> List[Dict]:
        """Paginated history retrieval"""
        where_clauses = ["h.username = ?"]
        params = [username]
        
        if filters:
            if filters.get('status'):
                where_clauses.append("h.status = ?")
                params.append(filters['status'])
            
            if filters.get('language'):
                where_clauses.append("h.language = ?")
                params.append(filters['language'])
            
            if filters.get('search'):
                where_clauses.append(
                    "(h.video_title LIKE ? OR h.video_channel LIKE ? OR h.url LIKE ?)"
                )
                search_term = f"%{filters['search']}%"
                params.extend([search_term, search_term, search_term])
        
        where_clause = " AND ".join(where_clauses)
        params.extend([limit, offset])
        
        with self.db.get_connection() as conn:
            results = conn.execute(f"""
                SELECT h.*, t.error_message
                FROM history h
                LEFT JOIN transcripts t ON h.id = t.history_id
                WHERE {where_clause}
                ORDER BY h.timestamp DESC
                LIMIT ? OFFSET ?
            """, params).fetchall()
            
            return [dict(row) for row in results]
    
    def get_history_entry_full(self, username: str, entry_id: str) -> Optional[Dict]:
        """Get full entry including transcripts"""
        with self.db.get_connection() as conn:
            result = conn.execute("""
                SELECT h.*, 
                       t.executive_summary, t.detailed_transcript, t.error_message
                FROM history h
                LEFT JOIN transcripts t ON h.id = t.history_id
                WHERE h.username = ? AND h.id = ?
            """, (username, entry_id)).fetchone()
            
            if result:
                entry = dict(result)
                entry['executive_summary'] = self._decompress(entry.get('executive_summary'))
                entry['detailed_transcript'] = self._decompress(entry.get('detailed_transcript'))
                return entry
            return None
    
    def get_history_statistics(self, username: str) -> Dict:
        """Optimized statistics with aggregation"""
        with self.db.get_connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_videos,
                    SUM(transcript_length) as total_length,
                    AVG(transcript_length) as avg_length,
                    MAX(timestamp) as most_recent,
                    GROUP_CONCAT(DISTINCT language) as languages,
                    GROUP_CONCAT(DISTINCT model_used) as models,
                    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed_count
                FROM history
                WHERE username = ?
            """, (username,)).fetchone()
            
            if stats and stats['total_videos'] > 0:
                languages = stats['languages'].split(',') if stats['languages'] else []
                models = stats['models'].split(',') if stats['models'] else []
                
                return {
                    "total_videos": stats['total_videos'],
                    "total_transcript_length": stats['total_length'] or 0,
                    "average_transcript_length": int(stats['avg_length'] or 0),
                    "most_recent": stats['most_recent'],
                    "languages_used": [l for l in languages if l],
                    "models_used": [m for m in models if m],
                    "success_rate": f"{stats['completed_count']}/{stats['total_videos']}"
                }
            
            return {
                "total_videos": 0,
                "total_transcript_length": 0,
                "average_transcript_length": 0,
                "most_recent": None,
                "languages_used": [],
                "models_used": [],
                "success_rate": "0/0"
            }
    
    def delete_history_entry(self, username: str, entry_id: str):
        """Cascade delete"""
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM history WHERE username = ? AND id = ?",
                (username, entry_id)
            )
    
    def clear_user_history(self, username: str):
        """Efficient bulk delete"""
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM history WHERE username = ?", (username,))

# ==================== AUTHENTICATION MANAGER ====================

class SecureAuthManager:
    """Secure authentication with bcrypt and sessions"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.failed_attempts = {}
        self.MAX_FAILED_ATTEMPTS = 5
        self.LOCKOUT_DURATION = timedelta(minutes=15)
        self._load_users_from_env()
    
    def _load_users_from_env(self):
        """Load additional users from environment variables"""
        env_users = {}
        
        for key, value in os.environ.items():
            if key.endswith('_PASSWORD') and not key.startswith('ADMIN_'):
                username = key[:-9].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                
                salt = bcrypt.gensalt(rounds=12)
                env_users[username]["password_hash"] = bcrypt.hashpw(
                    value.encode('utf-8'), salt
                ).decode('utf-8')
                
            elif key.endswith('_SUPADATA_KEY') and not key.startswith('ADMIN_'):
                username = key[:-13].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["supadata"] = value
                
            elif key.endswith('_ASSEMBLYAI_KEY') and not key.startswith('ADMIN_'):
                username = key[:-14].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["assemblyai"] = value
                
            elif key.endswith('_DEEPSEEK_KEY') and not key.startswith('ADMIN_'):
                username = key[:-12].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["deepseek"] = value
                
            elif key.endswith('_YOUTUBE_KEY') and not key.startswith('ADMIN_'):
                username = key[:-12].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["youtube"] = value
        
        # Add users to database
        with self.db.get_connection() as conn:
            for username, user_data in env_users.items():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
                        (username, user_data["password_hash"])
                    )
                    
                    for service, key in user_data.get("api_keys", {}).items():
                        if key:
                            conn.execute(
                                "INSERT OR REPLACE INTO api_keys (username, service, api_key) VALUES (?, ?, ?)",
                                (username, service, key)
                            )
                except Exception:
                    pass
    
    def is_account_locked(self, username: str) -> tuple[bool, int]:
        """Check if account is locked, return (is_locked, minutes_remaining)"""
        if username in self.failed_attempts:
            count, last_attempt = self.failed_attempts[username]
            if count >= self.MAX_FAILED_ATTEMPTS:
                elapsed = datetime.now() - last_attempt
                if elapsed < self.LOCKOUT_DURATION:
                    remaining = self.LOCKOUT_DURATION - elapsed
                    minutes = int(remaining.total_seconds() / 60)
                    return True, minutes
                else:
                    del self.failed_attempts[username]
        return False, 0
    
    def authenticate(self, username: str, password: str) -> tuple[bool, Optional[str]]:
        """Authenticate with security features"""
        is_locked, minutes = self.is_account_locked(username)
        if is_locked:
            return False, f"Account locked. Try again in {minutes} minutes."
        
        with self.db.get_connection() as conn:
            result = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,)
            ).fetchone()
            
            if result:
                try:
                    password_valid = bcrypt.checkpw(
                        password.encode('utf-8'), 
                        result['password_hash'].encode('utf-8')
                    )
                except Exception:
                    password_valid = False
                
                if password_valid:
                    if username in self.failed_attempts:
                        del self.failed_attempts[username]
                    return True, None
            
            # Failed attempt
            if username not in self.failed_attempts:
                self.failed_attempts[username] = (1, datetime.now())
            else:
                count, _ = self.failed_attempts[username]
                self.failed_attempts[username] = (count + 1, datetime.now())
            
            remaining = self.MAX_FAILED_ATTEMPTS - self.failed_attempts[username][0]
            if remaining > 0:
                return False, f"Invalid credentials. {remaining} attempts remaining."
            else:
                return False, "Account locked due to too many failed attempts."
    
    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        """Get user API keys"""
        with self.db.get_connection() as conn:
            results = conn.execute(
                "SELECT service, api_key FROM api_keys WHERE username = ?",
                (username,)
            ).fetchall()
            
            return {row['service']: row['api_key'] for row in results}

# ==================== YOUTUBE COOKIE MANAGER ====================

class YouTubeCookieManager:
    """Manages YouTube cookies for yt-dlp"""
    
    @staticmethod
    def get_ydl_opts(use_cookies: bool = True, browser: str = 'chrome') -> dict:
        """Get yt-dlp options with cookie configuration"""
        base_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        if use_cookies:
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                base_opts['cookiefile'] = render_cookies
            else:
                env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
                if env_cookies and os.path.exists(env_cookies):
                    base_opts['cookiefile'] = env_cookies
        
        base_opts['headers'] = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        return base_opts

# ==================== OPTIMIZED CHUNKER ====================

class OptimizedPerformanceChunker:
    """High-performance chunking with pre-compiled patterns"""
    
    SECTION_BREAK = re.compile(r'\n\s*\n')
    SENTENCE_END = re.compile(r'[.!?]\s+')
    NEWLINE = re.compile(r'\n')
    
    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.configs = {
            "speed": {
                "max_chunk_length": 6000,
                "overlap_length": 100,
                "max_parallel_workers": 4,
                "summary_truncate_length": 15000,
                "processing_timeout": 45
            },
            "balanced": {
                "max_chunk_length": 8000,
                "overlap_length": 200,
                "max_parallel_workers": 3,
                "summary_truncate_length": 25000,
                "processing_timeout": 90
            },
            "quality": {
                "max_chunk_length": 12000,
                "overlap_length": 400,
                "max_parallel_workers": 2,
                "summary_truncate_length": None,
                "processing_timeout": 180
            }
        }
        self.config = self.configs[performance_mode]
        self.min_chunk_length = 1000
    
    def get_processing_config(self) -> Dict:
        return {"mode": self.performance_mode, **self.config}
    
    def should_chunk_text(self, text: str) -> bool:
        text_length = len(text)
        if self.performance_mode == "speed":
            return text_length > 12000
        elif self.performance_mode == "balanced":
            return text_length > 20000
        else:
            return text_length > 30000
    
    def prepare_text_for_summary(self, text: str) -> str:
        """Prepare text for summary with smart truncation"""
        truncate_length = self.config.get("summary_truncate_length")
        
        if truncate_length and len(text) > truncate_length:
            beginning = text[:truncate_length // 2]
            end = text[-(truncate_length // 2):]
            
            beginning_break = beginning.rfind('\n\n')
            if beginning_break == -1:
                beginning_break = beginning.rfind('. ')
            if beginning_break > truncate_length // 3:
                beginning = beginning[:beginning_break + 2]
            
            end_break = end.find('\n\n')
            if end_break == -1:
                end_break = end.find('. ')
            if end_break != -1 and end_break < len(end) // 3:
                end = end[end_break + 2:]
            
            return beginning + "\n\n[... content truncated for speed ...]\n\n" + end
        
        return text
    
    @functools.lru_cache(maxsize=100)
    def _find_split_points_cached(self, text_hash: str, text: str) -> tuple:
        """Cache split point calculation"""
        split_points = set()
        
        split_points.update(m.end() for m in self.SECTION_BREAK.finditer(text))
        
        if len(split_points) < 20:
            split_points.update(m.end() for m in self.SENTENCE_END.finditer(text))
        
        if len(split_points) < 50:
            split_points.update(m.end() for m in self.NEWLINE.finditer(text))
        
        return tuple(sorted(split_points))
    
    def create_intelligent_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Optimized chunking with caching"""
        if not self.should_chunk_text(text):
            return [{
                'chunk_id': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'text': text,
                'tokens_estimate': len(text) // 4,
                'is_single_chunk': True
            }]
        
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
        split_points = self._find_split_points_cached(text_hash, text)
        
        chunks = []
        current_start = 0
        chunk_id = 0
        
        max_chunk_length = self.config["max_chunk_length"]
        overlap_length = self.config["overlap_length"]
        
        while current_start < len(text):
            target_end = current_start + max_chunk_length
            
            if target_end >= len(text):
                chunk_text = text[current_start:]
                chunks.append({
                    'chunk_id': chunk_id,
                    'start_pos': current_start,
                    'end_pos': len(text),
                    'text': chunk_text,
                    'tokens_estimate': len(chunk_text) // 4,
                    'is_final_chunk': True
                })
                break
            
            best_split = self._find_best_split_binary(
                split_points, current_start + self.min_chunk_length, target_end
            )
            
            if best_split is None:
                best_split = target_end
            
            chunk_text = text[current_start:best_split]
            chunks.append({
                'chunk_id': chunk_id,
                'start_pos': current_start,
                'end_pos': best_split,
                'text': chunk_text,
                'tokens_estimate': len(chunk_text) // 4,
                'is_final_chunk': False
            })
            
            current_start = max(current_start + self.min_chunk_length, 
                              best_split - overlap_length)
            chunk_id += 1
        
        return chunks
    
    @staticmethod
    def _find_best_split_binary(split_points: tuple, min_pos: int, max_pos: int) -> Optional[int]:
        """Binary search for best split point"""
        import bisect
        
        idx = bisect.bisect_left(split_points, max_pos)
        
        while idx > 0:
            split = split_points[idx - 1]
            if min_pos <= split <= max_pos:
                return split
            idx -= 1
        
        return None
    
    def create_chunk_specific_prompt(self, base_prompt: str, chunk_info: Dict, 
                                    total_chunks: int, language: str) -> str:
        """Create context-aware prompt"""
        chunk_id = chunk_info['chunk_id']
        is_first = chunk_id == 0
        is_last = chunk_info.get('is_final_chunk', False)
        
        context_info = f"This is chunk {chunk_id + 1} of {total_chunks} from a longer transcript."
        
        if is_first:
            context_info += " This is the BEGINNING of the transcript."
        elif is_last:
            context_info += " This is the FINAL chunk of the transcript."
        else:
            context_info += " This is a MIDDLE section of the transcript."
        
        if language == "繁體中文" or language == "中文":
            chunk_instruction = f"""
{base_prompt}

**重要提示：**
{context_info}
請處理此部分內容，保持與整體文檔的連貫性。
"""
        else:
            chunk_instruction = f"""
{base_prompt}

**IMPORTANT CONTEXT:**
{context_info}
Process this section while maintaining coherence with the overall document.
"""
        
        return chunk_instruction

# ==================== LANGUAGE DETECTOR ====================

class LanguageDetector:
    """Optimized language detection"""
    
    @staticmethod
    @functools.lru_cache(maxsize=500)
    def detect_language(text_sample: str) -> str:
        """Cached language detection"""
        if not text_sample or len(text_sample.strip()) < 10:
            return "English"
        
        # Sample first 2000 chars for performance
        sample = text_sample[:2000]
        
        chinese_chars = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in sample if c.isalpha() and ord(c) < 128)
        total_chars = chinese_chars + english_chars
        
        if total_chars == 0:
            return "English"
        
        chinese_ratio = chinese_chars / total_chars
        
        return "Chinese" if chinese_ratio > 0.3 else "English"
    
    @staticmethod
    def get_language_code(language: str) -> str:
        """Convert language name to code"""
        if language == "Chinese":
            return "繁體中文"
        else:
            return "English"

# ==================== YOUTUBE AUDIO EXTRACTOR ====================

class YouTubeAudioExtractor:
    """Extract audio URLs from YouTube"""
    
    def __init__(self):
        self.cookie_manager = YouTubeCookieManager()
    
    def extract_audio_url(self, youtube_url: str, browser: str = 'chrome') -> Optional[str]:
        """Extract direct audio URL"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                ydl_opts['cookiefile'] = render_cookies
            else:
                env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
                if env_cookies and os.path.exists(env_cookies):
                    ydl_opts['cookiefile'] = env_cookies
            
            ydl_opts['headers'] = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                if info:
                    formats = info.get('formats', [])
                    audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                    
                    if audio_formats:
                        best_audio = max(audio_formats, key=lambda x: x.get('abr', 0) or 0)
                        return best_audio.get('url')
                    
                    formats_with_audio = [f for f in formats if f.get('acodec') != 'none']
                    if formats_with_audio:
                        best_format = max(formats_with_audio, key=lambda x: x.get('abr', 0) or 0)
                        return best_format.get('url')
            
            return None
            
        except Exception as e:
            st.error(f"Audio extraction error: {e}")
            return None

# ==================== TRANSCRIPT PROVIDERS ====================

class TranscriptProvider:
    """Base class for transcript providers"""
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class OptimizedSupadataProvider(TranscriptProvider):
    """Optimized Supadata provider"""
    
    def __init__(self, api_key: str):
        try:
            from supadata import Supadata
            self.client = Supadata(api_key=api_key)
            self.available = True
        except ImportError:
            self.available = False
            self.client = None
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.available or not self.client:
            return None
        
        language_map = {"English": "en", "中文": "zh", "zh": "zh", "en": "en"}
        supadata_lang = language_map.get(language, "en")
        
        try:
            resp = self.client.transcript(url=url, lang=supadata_lang, text=True, mode="auto")
            return self._normalize_transcript(resp)
        except Exception as e:
            st.error(f"Supadata error: {e}")
            return None
    
    @staticmethod
    @functools.lru_cache(maxsize=100)
    def _normalize_transcript_cached(resp_str: str) -> str:
        """Cached normalization"""
        if "BatchJob" in resp_str or "job_id" in resp_str:
            return ""
        return resp_str
    
    def _normalize_transcript(self, resp) -> str:
        """Normalize various response formats"""
        if resp is None:
            return ""
        
        resp_str = str(resp)
        if "BatchJob" in resp_str or "job_id" in resp_str:
            return ""

        if isinstance(resp, str):
            return self._normalize_transcript_cached(resp)

        if isinstance(resp, dict):
            if resp.get("status") == "processing" or resp.get("job_id"):
                return ""
            
            if "text" in resp:
                return resp["text"]
            
            if "chunks" in resp:
                return "\n".join(
                    chunk.get("text", chunk.get("content", str(chunk)))
                    for chunk in resp["chunks"]
                    if isinstance(chunk, (str, dict))
                )

        if isinstance(resp, list):
            return "\n".join(
                chunk.get("text", chunk.get("content", str(chunk)))
                for chunk in resp
                if isinstance(chunk, (str, dict))
            )

        if hasattr(resp, "text"):
            return resp.text

        return str(resp) if "BatchJob" not in str(resp) else ""

class OptimizedAssemblyAIProvider(TranscriptProvider):
    """Optimized AssemblyAI provider with circuit breaker"""
    
    def __init__(self, api_key: str, browser: str = 'chrome'):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
        self.browser = browser
        
        self.consecutive_failures = 0
        self.MAX_FAILURES = 5
        self.circuit_open = False
        self.circuit_open_time = None
        self.CIRCUIT_TIMEOUT = timedelta(minutes=5)
    
    def is_circuit_open(self) -> bool:
        """Check circuit breaker"""
        if self.circuit_open:
            if datetime.now() - self.circuit_open_time > self.CIRCUIT_TIMEOUT:
                self.circuit_open = False
                self.consecutive_failures = 0
                return False
            return True
        return False
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        """Get transcript with circuit breaker"""
        if self.is_circuit_open():
            st.error("AssemblyAI temporarily unavailable")
            return None
        
        try:
            result = self._transcribe_standard(url, language)
            
            if result:
                self.consecutive_failures = 0
                return result
            else:
                self._handle_failure()
                return None
                
        except Exception as e:
            self._handle_failure()
            st.error(f"AssemblyAI error: {e}")
            return None
    
    def _handle_failure(self):
        """Handle failure and update circuit breaker"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.MAX_FAILURES:
            self.circuit_open = True
            self.circuit_open_time = datetime.now()
    
    def _transcribe_standard(self, url: str, language: str) -> Optional[str]:
        """Standard transcription"""
        audio_url = self.audio_extractor.extract_audio_url(url, self.browser)
        
        if not audio_url:
            return None
        
        return self._transcribe_audio_url(audio_url, language)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def _transcribe_audio_url(self, audio_url: str, language: str) -> Optional[str]:
        """Transcribe with retry"""
        headers = {
            "authorization": self.api_key,
            "content-type": "application/json"
        }
        
        language_map = {"English": "en", "中文": "zh", "en": "en", "zh": "zh"}
        assemblyai_lang = language_map.get(language, "en")
        
        data = {
            "audio_url": audio_url,
            "language_code": assemblyai_lang,
            "speech_model": "best"
        }
        
        response = OptimizedHTTPClient.get_session().post(
            f"{self.base_url}/transcript",
            json=data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get('error', 'Unknown error')
            raise Exception(f"AssemblyAI error: {error_msg}")
        
        transcript_id = response.json()['id']
        return self._poll_for_completion(transcript_id, headers)
    
    def _poll_for_completion(self, transcript_id: str, headers: dict) -> Optional[str]:
        """Poll for completion"""
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        max_attempts = 120
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for attempt in range(max_attempts):
            response = OptimizedHTTPClient.get_session().get(
                polling_endpoint, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            status = data.get('status', 'unknown')
            
            status_text.text(f"Status: {status} ({attempt + 1}/{max_attempts})")
            progress_bar.progress((attempt + 1) / max_attempts)
            
            if status == 'completed':
                progress_bar.progress(1.0)
                return data.get('text', '')
            elif status == 'error':
                return None
            
            time.sleep(3)
        
        return None

# ==================== LLM PROVIDER ====================

@dataclass
class ProcessedChunk:
    """Memory-efficient chunk result"""
    chunk_id: int
    result: Optional[str]
    processing_time: float

class OptimizedDeepSeekProvider:
    """Optimized DeepSeek provider with async support"""
    
    def __init__(self, api_key: str, base_url: str, model: str, 
                 temperature: float, performance_mode: str = "balanced"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.performance_mode = performance_mode
        
        self.text_chunker = OptimizedPerformanceChunker(performance_mode)
        self.language_detector = LanguageDetector()
        self.config = self.text_chunker.get_processing_config()
    
    def structure_transcript_with_live_updates(self, transcript: str, system_prompt: str, 
                                              ui_language: str = "English",
                                              is_custom_prompt: bool = False,
                                              progress_callback=None) -> Optional[Dict[str, str]]:
        """Process transcript with optimizations"""
        detected_language = self.language_detector.detect_language(transcript)
        actual_language = self.language_detector.get_language_code(detected_language)
        
        mode_emoji = {"speed": "⚡", "balanced": "⚖️", "quality": "🎯"}
        st.info(f"🔍 Detected: **{actual_language}** | {mode_emoji[self.performance_mode]} {self.performance_mode.title()}")
        
        if self.performance_mode == "speed":
            return self._process_with_parallel_strategy(
                transcript, system_prompt, actual_language, is_custom_prompt, progress_callback
            )
        else:
            return self._process_with_sequential_strategy(
                transcript, system_prompt, actual_language, is_custom_prompt, progress_callback
            )
    
    def _process_with_parallel_strategy(self, transcript: str, system_prompt: str,
                                       language: str, is_custom_prompt: bool,
                                       progress_callback) -> Optional[Dict[str, str]]:
        """Parallel processing for speed mode"""
        summary_transcript = self.text_chunker.prepare_text_for_summary(transcript)
        
        summary_prompt = self._create_summary_prompt(language)
        adapted_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            st.info("⚡ Starting parallel processing...")
            
            summary_future = executor.submit(
                self._process_for_summary, summary_transcript, summary_prompt, language
            )
            detailed_future = executor.submit(
                self._process_for_detailed_structure, transcript, adapted_prompt, language
            )
            
            try:
                summary_result = summary_future.result(timeout=120)
                if summary_result and progress_callback:
                    progress_callback("executive_summary", summary_result)
            except Exception:
                summary_result = None
            
            try:
                detailed_result = detailed_future.result(timeout=300)
            except Exception:
                detailed_result = None
        
        if not summary_result and not detailed_result:
            return None
        
        return {
            'executive_summary': summary_result,
            'detailed_transcript': detailed_result,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt,
            'performance_mode': self.performance_mode
        }
    
    def _process_with_sequential_strategy(self, transcript: str, system_prompt: str,
                                         language: str, is_custom_prompt: bool,
                                         progress_callback) -> Optional[Dict[str, str]]:
        """Sequential processing"""
        st.info("Step 1: Generating executive summary...")
        summary_prompt = self._create_summary_prompt(language)
        executive_summary = self._process_for_summary(transcript, summary_prompt, language)
        
        if executive_summary and progress_callback:
            progress_callback("executive_summary", executive_summary)
        
        st.info("Step 2: Generating detailed transcript...")
        adapted_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        detailed_transcript = self._process_for_detailed_structure(transcript, adapted_prompt, language)
        
        if not detailed_transcript:
            return None
        
        return {
            'executive_summary': executive_summary,
            'detailed_transcript': detailed_transcript,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt,
            'performance_mode': self.performance_mode
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_api_request(self, text: str, system_prompt: str) -> Optional[str]:
        """API request with retry"""
        endpoint = self.base_url + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
        }

        resp = OptimizedHTTPClient.get_session().post(
            endpoint, 
            headers=headers, 
            json=payload, 
            timeout=self.config["processing_timeout"]
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}")
        
        data = resp.json()
        
        if 'choices' not in data or len(data['choices']) == 0:
            raise RuntimeError("No choices returned")
        
        return data["choices"][0]["message"]["content"].strip()
    
    def _process_for_summary(self, transcript: str, summary_prompt: str, language: str) -> Optional[str]:
        """Generate summary"""
        if len(transcript) > 12000:
            chunks = self.text_chunker.create_intelligent_chunks(transcript)
            return self._process_summary_with_chunking(chunks, summary_prompt, language)
        else:
            return self._make_api_request(transcript, summary_prompt)
    
    def _process_summary_with_chunking(self, chunks: List[Dict], summary_prompt: str, 
                                      language: str) -> Optional[str]:
        """Process summary with chunking"""
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{summary_prompt}\n\nChunk {i+1}/{len(chunks)}: Focus on key points from this section."
            result = self._make_api_request(chunk['text'], chunk_prompt)
            if result:
                chunk_summaries.append(result)
        
        if not chunk_summaries:
            return None
        
        combined = "\n\n".join(chunk_summaries)
        
        final_prompt = "Combine these summaries into one unified executive summary:" if language == "English" else "將這些摘要合併成一個統一的執行摘要："
        
        return self._make_api_request(combined, final_prompt)
    
    def _process_for_detailed_structure(self, transcript: str, system_prompt: str, 
                                       language: str) -> Optional[str]:
        """Process detailed structure"""
        if not self.text_chunker.should_chunk_text(transcript):
            return self._make_api_request(transcript, system_prompt)
        else:
            return self._process_detailed_with_chunking(transcript, system_prompt, language)
    
    def _process_detailed_with_chunking(self, transcript: str, system_prompt: str, 
                                       language: str) -> Optional[str]:
        """Process with chunking"""
        chunks = self.text_chunker.create_intelligent_chunks(transcript)
        
        if len(chunks) == 1:
            return self._make_api_request(transcript, system_prompt)
        
        if self.performance_mode == "speed" and len(chunks) > 2:
            return self._process_chunks_parallel(chunks, system_prompt, language)
        else:
            return self._process_chunks_sequential(chunks, system_prompt, language)
    
    def _process_chunks_parallel(self, chunks: List[Dict], system_prompt: str, 
                                language: str) -> Optional[str]:
        """Parallel chunk processing"""
        max_workers = self.config["max_parallel_workers"]
        
        def process_chunk(chunk_info):
            chunk_prompt = self.text_chunker.create_chunk_specific_prompt(
                system_prompt, chunk_info, len(chunks), language
            )
            return self._make_api_request(chunk_info['text'], chunk_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    pass
        
        if not results:
            return None
        
        return self._combine_processed_chunks(results, language)
    
    def _process_chunks_sequential(self, chunks: List[Dict], system_prompt: str, 
                                  language: str) -> Optional[str]:
        """Sequential chunk processing"""
        results = []
        
        for i, chunk in enumerate(chunks):
            st.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_prompt = self.text_chunker.create_chunk_specific_prompt(
                system_prompt, chunk, len(chunks), language
            )
            
            try:
                result = self._make_api_request(chunk['text'], chunk_prompt)
                if result:
                    results.append(result)
            except Exception:
                continue
        
        if not results:
            return None
        
        return self._combine_processed_chunks(results, language)
    
    def _combine_processed_chunks(self, chunks: List[str], language: str) -> str:
        """Combine processed chunks"""
        if len(chunks) == 1:
            return chunks[0]
        
        separator = "\n\n---\n\n"
        header = "# Detailed Structured Transcript\n\n" if language == "English" else "# 詳細結構化轉錄文稿\n\n"
        
        return header + separator.join(chunks)
    
    def _adapt_system_prompt_to_language(self, system_prompt: str, detected_language: str, 
                                        is_custom_prompt: bool = False) -> str:
        """Adapt prompt to language"""
        if is_custom_prompt:
            language_instruction = "\n\n**Please output in English.**" if detected_language == "English" else "\n\n**請用繁體中文輸出。**"
            return system_prompt + language_instruction
        
        if detected_language == "繁體中文":
            return """你是一個專業的YouTube影片轉錄文本分析和結構化專家。你的任務是將原始轉錄文本轉換為組織良好、易於閱讀的文檔。

請按照以下指導原則來結構化轉錄文本：

1. **創建清晰的章節和標題**，基於主題變化和內容流程
2. **提高可讀性**：
   - 修正語法和標點符號
   - 刪除填充詞（嗯、呃、那個、就是說）
   - 合併斷裂的句子
   - 添加段落分隔以改善流暢性

3. **保留所有重要信息** - 不要總結或省略內容
4. **使用markdown格式** 來設置標題、強調和結構
5. **在自然主題轉換處添加時間戳**
6. **保持說話者的語調和意思**，同時提高清晰度

將輸出格式化為一個清潔、專業的文檔，便於閱讀和參考。請用繁體中文輸出結構化的轉錄文本。"""
        
        else:
            return """You are an expert at analyzing and structuring YouTube video transcripts. Your task is to convert raw transcript text into a well-organized, readable document.

Please structure the transcript following these guidelines:

1. **Create clear sections and headings** based on topic changes and content flow
2. **Improve readability** by:
   - Fixing grammar and punctuation
   - Removing filler words (um, uh, like, you know)
   - Combining fragmented sentences
   - Adding paragraph breaks for better flow

3. **Preserve all important information** - don't summarize or omit content
4. **Use markdown formatting** for headers, emphasis, and structure
5. **Add timestamps** where natural topic transitions occur
6. **Maintain the speaker's tone and meaning** while improving clarity

Format the output as a clean, professional document that would be easy to read and reference. Please output in English."""
    
    def _create_summary_prompt(self, language: str) -> str:
        """Create summary prompt"""
        if language == "繁體中文":
            return """你是一個專業的YouTube影片轉錄文本執行摘要專家。

你的任務是創建一個全面的執行摘要，捕捉轉錄文本中的關鍵點、主要觀點和重要信息。

請按以下結構組織你的執行摘要：

1. **概述** - 簡要2-3句話描述影片內容
2. **關鍵點** - 討論的主要觀點、論證或話題（使用要點形式）
3. **重要細節** - 提到的重要事實、統計數據、例子或見解
4. **結論/要點** - 主要結論、建議或可操作的見解

要求：
- 保持簡潔但全面（目標300-500字）
- 使用要點形式便於掃描
- 捕捉人們需要了解的最重要信息
- 保持說話者的關鍵信息和語調

請用繁體中文輸出執行摘要。"""
        
        else:
            return """You are an expert at creating concise executive summaries from YouTube video transcripts.

Your task is to create a comprehensive EXECUTIVE SUMMARY that captures the key points, main ideas, and essential information from the transcript.

Please structure your executive summary with:

1. **Overview** - A brief 2-3 sentence description of what the video covers
2. **Key Points** - The main ideas, arguments, or topics discussed (use bullet points)
3. **Important Details** - Significant facts, statistics, examples, or insights mentioned
4. **Conclusions/Takeaways** - Main conclusions, recommendations, or actionable insights

Requirements:
- Keep it concise but comprehensive (aim for 300-500 words)
- Use bullet points for easy scanning
- Capture the most important information that someone would need to know

Please output the executive summary in English."""

# ==================== YOUTUBE DATA PROVIDER ====================

class YouTubeDataProvider:
    """Provider for YouTube Data API operations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.available = bool(api_key)
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        """Extract videos from playlist"""
        if not self.available:
            return []
            
        playlist_id = self.extract_playlist_id(playlist_url)
        if not playlist_id:
            return []
        
        try:
            videos = []
            next_page_token = None
            
            while True:
                params = {
                    'part': 'snippet',
                    'playlistId': playlist_id,
                    'maxResults': 50,
                    'key': self.api_key
                }
                
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                response = OptimizedHTTPClient.get_session().get(
                    f"{self.base_url}/playlistItems",
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                
                for item in data.get('items', []):
                    snippet = item.get('snippet', {})
                    resource_id = snippet.get('resourceId', {})
                    
                    video_id = resource_id.get('videoId')
                    title = snippet.get('title', 'Unknown Title')
                    
                    if video_id:
                        videos.append({
                            'title': title,
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'video_id': video_id
                        })
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos
            
        except Exception:
            return []
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from URL"""
        match = re.search(r'list=([\w-]+)', url)
        return match.group(1) if match else None

# ==================== ORCHESTRATOR ====================

class EnhancedTranscriptOrchestrator:
    """Orchestrator with optimizations"""
    
    def __init__(
        self, 
        transcript_provider: TranscriptProvider,
        asr_fallback_provider: Optional[TranscriptProvider] = None,
        llm_provider: Optional[OptimizedDeepSeekProvider] = None
    ):
        self.transcript_provider = transcript_provider
        self.asr_fallback_provider = asr_fallback_provider
        self.llm_provider = llm_provider
    
    def get_transcript(self, url: str, language: str, use_fallback: bool = False) -> Optional[str]:
        """Get transcript with fallback"""
        transcript = self.transcript_provider.get_transcript(url, language)
        
        def is_valid_transcript(text):
            if not text:
                return False
            if "BatchJob" in str(text) or "job_id" in str(text):
                return False
            cleaned = text.strip()
            if len(cleaned) < 100:
                return False
            word_count = len(cleaned.split())
            if word_count < 20:
                return False
            return True
        
        if not is_valid_transcript(transcript):
            transcript = None
        
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("Trying ASR fallback...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
            if not is_valid_transcript(transcript):
                transcript = None
        
        return transcript
    
    def structure_transcript_with_live_updates(self, transcript: str, system_prompt: str, 
                                             language: str = "English",
                                             is_custom_prompt: bool = False,
                                             progress_callback=None) -> Optional[Dict[str, str]]:
        """Structure transcript"""
        if not self.llm_provider:
            return None
        return self.llm_provider.structure_transcript_with_live_updates(
            transcript, system_prompt, language, is_custom_prompt, progress_callback
        )

# ==================== STREAMLIT UI ====================

@st.cache_resource
def get_database_manager():
    """Cached database manager"""
    return DatabaseManager()

@st.cache_resource
def get_auth_manager():
    """Cached auth manager"""
    db = get_database_manager()
    return SecureAuthManager(db)

@st.cache_resource
def get_data_manager():
    """Cached data manager"""
    db = get_database_manager()
    return OptimizedUserDataManager(db)

def login_page():
    """Login page"""
    st.title("YouTube Transcript Processor")
    st.subheader("Login Required")
    
    st.info("**Default Login:** Username: `admin` | Password: `admin123`")
    
    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
                return False
            
            auth_manager = get_auth_manager()
            success, error_msg = auth_manager.authenticate(username, password)
            
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.api_keys = auth_manager.get_user_api_keys(username)
                
                data_manager = get_data_manager()
                st.session_state.user_data_manager = data_manager
                
                youtube_key = st.session_state.api_keys.get('youtube')
                data_manager.set_youtube_extractor(youtube_key)
                
                user_settings = data_manager.get_user_settings(username)
                st.session_state.user_settings = user_settings
                
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                st.error(error_msg)
                return False
    
    return False

def show_history_page():
    """Optimized history page with pagination"""
    st.header("📚 Processing History")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    if 'history_page' not in st.session_state:
        st.session_state.history_page = 0
    if 'history_filters' not in st.session_state:
        st.session_state.history_filters = {}
    
    # Cached stats
    stats = data_manager.get_history_statistics(username)
    
    if stats["total_videos"] > 0:
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Videos", stats["total_videos"])
            with col2:
                st.metric("Avg Length", f"{stats['average_transcript_length']:,} chars")
            with col3:
                languages = ", ".join(stats["languages_used"][:2])
                st.metric("Languages", languages if languages else "N/A")
            with col4:
                st.metric("Success Rate", stats.get("success_rate", "0/0"))
        
        st.divider()
        
        # Filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input("🔍 Search", key="history_search")
            if search != st.session_state.history_filters.get('search', ''):
                st.session_state.history_filters['search'] = search
                st.session_state.history_page = 0
        
        with col2:
            status = st.selectbox("Status", ["All", "Completed", "Failed"], key="status_filter")
            if status != "All":
                st.session_state.history_filters['status'] = status
            elif 'status' in st.session_state.history_filters:
                del st.session_state.history_filters['status']
        
        with col3:
            lang_options = ["All"] + stats["languages_used"]
            lang = st.selectbox("Language", lang_options, key="lang_filter")
            if lang != "All":
                st.session_state.history_filters['language'] = lang
            elif 'language' in st.session_state.history_filters:
                del st.session_state.history_filters['language']
        
        # Paginated history
        page_size = 10
        offset = st.session_state.history_page * page_size
        
        history_page = data_manager.get_user_history_paginated(
            username, 
            limit=page_size, 
            offset=offset,
            filters=st.session_state.history_filters
        )
        
        if history_page:
            for entry in history_page:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{entry.get('video_title', 'Unknown')}**")
                        if entry.get('video_channel'):
                            st.caption(f"📺 {entry['video_channel']}")
                        
                        status = entry.get('status', 'Unknown')
                        if status == 'Completed':
                            st.success(f"✅ {status}")
                        elif status == 'Failed':
                            st.error(f"❌ {status}")
                        else:
                            st.info(f"⏳ {status}")
                    
                    with col2:
                        st.write(f"**Date:** {entry.get('timestamp', '')[:10]}")
                        st.write(f"**Lang:** {entry.get('language', 'N/A')}")
                    
                    with col3:
                        if st.button("👁️ View", key=f"view_{entry.get('id')}", type="primary"):
                            st.session_state.current_page = "history_detail"
                            st.session_state.current_entry_id = entry.get('id')
                            st.rerun()
                    
                    with col4:
                        if st.button("🗑️", key=f"del_{entry.get('id')}", type="secondary"):
                            data_manager.delete_history_entry(username, entry.get('id'))
                            st.rerun()
                
                st.divider()
            
            # Pagination
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("← Previous", disabled=st.session_state.history_page == 0):
                    st.session_state.history_page -= 1
                    st.rerun()
            
            with col2:
                st.write(f"Page {st.session_state.history_page + 1}")
            
            with col3:
                has_more = len(history_page) == page_size
                if st.button("Next →", disabled=not has_more):
                    st.session_state.history_page += 1
                    st.rerun()
        
        else:
            st.info("No videos match your filters.")
    
    else:
        st.info("No processing history yet.")

def show_history_detail_page(entry_id: str):
    """Detail page for history entry"""
    st.header("🔍 Transcript Details")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    entry = data_manager.get_history_entry_full(username, entry_id)
    
    if not entry:
        st.error("Entry not found!")
        if st.button("← Back"):
            st.session_state.current_page = "history"
            st.rerun()
        return
    
    if st.button("← Back to History", type="secondary"):
        st.session_state.current_page = "history"
        st.rerun()
    
    # Video info
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if entry.get('video_thumbnail'):
            st.image(entry['video_thumbnail'], width=200)
    
    with col2:
        st.title(entry.get('video_title', 'Unknown'))
        if entry.get('video_channel'):
            st.write(f"**Channel:** {entry['video_channel']}")
        st.write(f"**Status:** {entry.get('status', 'N/A')}")
        st.write(f"**Language:** {entry.get('language', 'N/A')}")
    
    st.divider()
    
    # Results
    if entry.get('status') == 'Completed':
        if entry.get('executive_summary'):
            st.subheader("📋 Executive Summary")
            st.markdown(entry['executive_summary'])
            st.download_button(
                "📄 Download Summary",
                entry['executive_summary'],
                file_name=f"summary_{entry_id}.md"
            )
            st.divider()
        
        if entry.get('detailed_transcript'):
            st.subheader("📝 Detailed Transcript")
            st.markdown(entry['detailed_transcript'])
            st.download_button(
                "📄 Download Transcript",
                entry['detailed_transcript'],
                file_name=f"transcript_{entry_id}.md"
            )
    
    elif entry.get('status') == 'Failed':
        st.error(f"Processing failed: {entry.get('error_message', 'Unknown error')}")

def show_settings_page():
    """Settings page"""
    st.header("⚙️ User Settings")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    current_settings = data_manager.get_user_settings(username)
    
    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Default Language",
                ["English", "中文"],
                index=0 if current_settings.get("language") == "English" else 1
            )
            
            use_asr = st.checkbox(
                "Enable ASR Fallback",
                value=current_settings.get("use_asr_fallback", True)
            )
            
            model = st.selectbox(
                "DeepSeek Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0 if current_settings.get("deepseek_model") == "deepseek-chat" else 1
            )
        
        with col2:
            temp = st.slider(
                "Temperature",
                0.0, 1.0,
                current_settings.get("temperature", 0.1),
                0.1
            )
            
            browser = st.selectbox(
                "Browser for Cookies",
                ["none", "chrome", "firefox", "edge", "safari"],
                index=["none", "chrome", "firefox", "edge", "safari"].index(
                    current_settings.get("browser_for_cookies", "none")
                )
            )
            
            perf_mode = st.selectbox(
                "Performance Mode",
                ["speed", "balanced", "quality"],
                index=["speed", "balanced", "quality"].index(
                    current_settings.get("performance_mode", "balanced")
                )
            )
        
        if st.form_submit_button("💾 Save Settings", type="primary"):
            new_settings = {
                "language": language,
                "use_asr_fallback": use_asr,
                "deepseek_model": model,
                "temperature": temp,
                "browser_for_cookies": browser,
                "performance_mode": perf_mode
            }
            
            data_manager.save_user_settings(username, new_settings)
            st.session_state.user_settings = new_settings
            st.success("Settings saved!")
            st.rerun()

def show_main_processing_page():
    """Main processing page"""
    api_keys = st.session_state.get('api_keys', {})
    user_settings = st.session_state.get('user_settings', {})
    
    st.header("Configuration")
    
    with st.expander("API Status", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("Supadata ✅") if api_keys.get('supadata') else st.error("Supadata ❌")
        with col2:
            st.success("AssemblyAI ✅") if api_keys.get('assemblyai') else st.error("AssemblyAI ❌")
        with col3:
            st.success("DeepSeek ✅") if api_keys.get('deepseek') else st.error("DeepSeek ❌")
        with col4:
            st.success("YouTube ✅") if api_keys.get('youtube') else st.error("YouTube ❌")
    
    with st.expander("Processing Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Language",
                ["English", "中文"],
                index=0 if user_settings.get("language") == "English" else 1
            )
            
            use_asr = st.checkbox(
                "Enable ASR Fallback",
                value=user_settings.get("use_asr_fallback", True)
            )
            
            perf_mode = st.selectbox(
                "Performance Mode",
                ["speed", "balanced", "quality"],
                index=["speed", "balanced", "quality"].index(
                    user_settings.get("performance_mode", "balanced")
                )
            )
        
        with col2:
            model = st.selectbox(
                "DeepSeek Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0 if user_settings.get("deepseek_model") == "deepseek-chat" else 1
            )
            
            temp = st.slider("Temperature", 0.0, 1.0, user_settings.get("temperature", 0.1), 0.1)
            
            browser = st.selectbox(
                "Browser",
                ["none", "chrome", "firefox"],
                index=["none", "chrome", "firefox"].index(
                    user_settings.get("browser_for_cookies", "none")
                )
            )
    
    st.header("Process Video")
    
    input_method = st.radio(
        "Input Method",
        ["Single Video URL", "Direct Transcript Input", "Playlist URL", "Batch URLs"]
    )
    
    videos_to_process = []
    
    if input_method == "Single Video URL":
        url = st.text_input("YouTube Video URL")
        if url:
            videos_to_process = [{"title": "Single Video", "url": url, "type": "url"}]
    
    elif input_method == "Direct Transcript Input":
        title = st.text_input("Transcript Title")
        text = st.text_area("Paste Transcript", height=300)
        
        if text:
            char_count = len(text)
            st.metric("Characters", f"{char_count:,}")
            
            if title and text:
                videos_to_process = [{
                    "title": title,
                    "url": f"direct_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "type": "direct_transcript",
                    "transcript": text
                }]
    
    elif input_method == "Playlist URL":
        playlist_url = st.text_input("Playlist URL")
        if playlist_url and api_keys.get('youtube'):
            if st.button("Load Playlist"):
                with st.spinner("Loading..."):
                    provider = YouTubeDataProvider(api_keys['youtube'])
                    videos = provider.get_playlist_videos(playlist_url)
                    if videos:
                        for v in videos:
                            v["type"] = "url"
                        st.success(f"Loaded {len(videos)} videos")
                        st.session_state.playlist_videos = videos
    
    elif input_method == "Batch URLs":
        batch_urls = st.text_area("URLs (one per line)")
        if batch_urls:
            urls = [u.strip() for u in batch_urls.split('\n') if u.strip()]
            videos_to_process = [{"title": f"Video {i+1}", "url": url, "type": "url"} 
                               for i, url in enumerate(urls)]
    
    if 'playlist_videos' in st.session_state and input_method == "Playlist URL":
        videos_to_process = st.session_state.playlist_videos
    
    if videos_to_process:
        st.subheader(f"Videos to Process ({len(videos_to_process)})")
        
        # System prompt
        default_prompt = """You are an expert at analyzing and structuring YouTube video transcripts..."""
        
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompt,
            height=200
        )
        
        mode_emoji = {"speed": "⚡", "balanced": "⚖️", "quality": "🎯"}
        if st.button(f"Start Processing {mode_emoji[perf_mode]} {perf_mode.title()}", type="primary"):
            process_videos(videos_to_process, language, use_asr, system_prompt, 
                         model, temp, api_keys, browser, perf_mode)

def process_videos(videos, language, use_asr, system_prompt, model, temp, api_keys, browser, perf_mode):
    """Process videos with optimizations"""
    username = st.session_state.username
    data_manager = get_data_manager()
    
    # Initialize providers
    supadata = OptimizedSupadataProvider(api_keys.get('supadata', ''))
    assemblyai = OptimizedAssemblyAIProvider(api_keys.get('assemblyai', ''), browser) if use_asr else None
    deepseek = OptimizedDeepSeekProvider(
        api_keys.get('deepseek', ''),
        "https://api.deepseek.com/v1",
        model,
        temp,
        perf_mode
    )
    
    orchestrator = EnhancedTranscriptOrchestrator(supadata, assemblyai, deepseek)
    
    for i, video in enumerate(videos):
        st.subheader(f"Processing {i+1}/{len(videos)}")
        st.write(f"**Title:** {video['title']}")
        
        start_time = time.time()
        
        history_entry = {
            "title": video['title'],
            "url": video['url'],
            "language": language,
            "model_used": model,
            "input_type": video.get("type", "url"),
            "status": "Processing",
            "performance_mode": perf_mode
        }
        
        data_manager.add_to_history(username, history_entry)
        entry_id = history_entry.get('id')
        
        try:
            # Get transcript
            if video.get("type") == "direct_transcript":
                transcript = video.get("transcript", "")
            else:
                with st.spinner("Getting transcript..."):
                    transcript = orchestrator.get_transcript(video['url'], language, use_asr)
            
            if not transcript:
                raise Exception("Failed to get transcript")
            
            history_entry["transcript_length"] = len(transcript)
            data_manager.update_history_entry(username, entry_id, history_entry)
            
            # Structure transcript
            with st.spinner("Structuring..."):
                summary_placeholder = st.empty()
                
                def progress_callback(component_type, content):
                    if component_type == "executive_summary":
                        with summary_placeholder.container():
                            st.subheader("📋 Executive Summary")
                            st.markdown(content)
                
                result = orchestrator.structure_transcript_with_live_updates(
                    transcript, system_prompt, language, False, progress_callback
                )
            
            if not result:
                raise Exception("Failed to structure transcript")
            
            processing_time = time.time() - start_time
            history_entry["processing_time"] = f"{processing_time:.1f}s"
            history_entry["status"] = "Completed"
            history_entry["executive_summary"] = result.get('executive_summary')
            history_entry["detailed_transcript"] = result.get('detailed_transcript')
            
            data_manager.update_history_entry(username, entry_id, history_entry)
            
            st.success("🎉 Processing completed!")
            
            # Show results
            if result.get('detailed_transcript'):
                with st.expander("📝 Detailed Transcript", expanded=False):
                    st.markdown(result['detailed_transcript'])
            
            # Downloads
            col1, col2 = st.columns(2)
            with col1:
                if result.get('executive_summary'):
                    st.download_button(
                        "Download Summary",
                        result['executive_summary'],
                        file_name=f"summary_{entry_id}.md"
                    )
            with col2:
                if result.get('detailed_transcript'):
                    st.download_button(
                        "Download Transcript",
                        result['detailed_transcript'],
                        file_name=f"transcript_{entry_id}.md"
                    )
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error: {e}")
            processing_time = time.time() - start_time
            history_entry["processing_time"] = f"{processing_time:.1f}s"
            history_entry["status"] = "Failed"
            history_entry["error"] = str(e)
            data_manager.update_history_entry(username, entry_id, history_entry)

def main_app():
    """Main app interface"""
    st.title("YouTube Transcript Processor")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    with st.sidebar:
        st.write(f"Welcome, **{st.session_state.username}**!")
        st.divider()
        
        if st.session_state.current_page != "history_detail":
            page_options = ["🎬 Process Videos", "📚 History", "⚙️ Settings"]
            current_index = 0
            
            if st.session_state.current_page == "history":
                current_index = 1
            elif st.session_state.current_page == "settings":
                current_index = 2
            
            page = st.radio("Navigation", page_options, index=current_index)
            
            if page == "🎬 Process Videos":
                st.session_state.current_page = "main"
            elif page == "📚 History":
                st.session_state.current_page = "history"
            elif page == "⚙️ Settings":
                st.session_state.current_page = "settings"
        
        st.divider()
        
        if st.button("🚪 Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Show appropriate page
    if st.session_state.current_page == "history_detail":
        entry_id = st.session_state.get('current_entry_id')
        if entry_id:
            show_history_detail_page(entry_id)
        else
        st.session_state.current_page = "history"
            st.rerun()
    elif st.session_state.current_page == "history":
        show_history_page()
    elif st.session_state.current_page == "settings":
        show_settings_page()
    else:
        show_main_processing_page()

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point"""
    st.set_page_config(
        page_title="YouTube Transcript Processor - Optimized",
        page_icon="⚡",
        layout="wide"
    )
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Show appropriate page
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
