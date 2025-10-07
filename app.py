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

import bcrypt
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
            
            # Create default admin user
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        with self.get_connection() as conn:
            result = conn.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
            if not result:
                salt = bcrypt.gensalt(rounds=12)
                password_hash = bcrypt.hashpw("admin123".encode('utf-8'), salt).decode('utf-8')
                
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    ("admin", password_hash)
                )
                
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

# ==================== HTTP CLIENT ====================

class OptimizedHTTPClient:
    """HTTP client with connection pooling"""
    
    _session = None
    _lock = threading.Lock()
    
    @classmethod
    def get_session(cls):
        """Get singleton session"""
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
    """Optimized YouTube info extractor"""
    
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
    """Optimized user data manager"""
    
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
        if not text:
            return b''
        return gzip.compress(text.encode('utf-8'))
    
    @staticmethod
    def _decompress(data: bytes) -> str:
        if not data:
            return ''
        return gzip.decompress(data).decode('utf-8')
    
    def add_to_history(self, username: str, entry: Dict):
        """Optimized history insertion"""
        if 'id' not in entry:
            entry['id'] = hashlib.md5(
                f"{entry.get('url', '')}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]
        
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
        """Get full entry"""
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
        """Optimized statistics"""
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
        """Delete entry"""
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM history WHERE username = ? AND id = ?",
                (username, entry_id)
            )
    
    def clear_user_history(self, username: str):
        """Clear all history"""
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM history WHERE username = ?", (username,))

# ==================== AUTHENTICATION ====================

class SecureAuthManager:
    """Secure authentication"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.failed_attempts = {}
        self.MAX_FAILED_ATTEMPTS = 5
        self.LOCKOUT_DURATION = timedelta(minutes=15)
        self._load_users_from_env()
    
    def _load_users_from_env(self):
        """Load users from environment"""
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
        """Check if locked"""
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
        """Authenticate user"""
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
            
            if username not in self.failed_attempts:
                self.failed_attempts[username] = (1, datetime.now())
            else:
                count, _ = self.failed_attempts[username]
                self.failed_attempts[username] = (count + 1, datetime.now())
            
            remaining = self.MAX_FAILED_ATTEMPTS - self.failed_attempts[username][0]
            if remaining > 0:
                return False, f"Invalid credentials. {remaining} attempts remaining."
            else:
                return False, "Account locked."
    
    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        """Get API keys"""
        with self.db.get_connection() as conn:
            results = conn.execute(
                "SELECT service, api_key FROM api_keys WHERE username = ?",
                (username,)
            ).fetchall()
            
            return {row['service']: row['api_key'] for row in results}

# ==================== CHUNKER ====================

class OptimizedPerformanceChunker:
    """High-performance chunking"""
    
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
        """Prepare text for summary"""
        truncate_length = self.config.get("summary_truncate_length")
        
        if truncate_length and len(text) > truncate_length:
            beginning = text[:truncate_length // 2]
            end = text[-(truncate_length // 2):]
            return beginning + "\n\n[... truncated ...]\n\n" + end
        
        return text
    
    @functools.lru_cache(maxsize=100)
    def _find_split_points_cached(self, text_hash: str, text: str) -> tuple:
        """Cache split points"""
        split_points = set()
        split_points.update(m.end() for m in self.SECTION_BREAK.finditer(text))
        
        if len(split_points) < 20:
            split_points.update(m.end() for m in self.SENTENCE_END.finditer(text))
        
        if len(split_points) < 50:
            split_points.update(m.end() for m in self.NEWLINE.finditer(text))
        
        return tuple(sorted(split_points))
    
    def create_intelligent_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks"""
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
        """Binary search"""
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
        """Create prompt for chunk"""
        chunk_id = chunk_info['chunk_id']
        is_first = chunk_id == 0
        is_last = chunk_info.get('is_final_chunk', False)
        
        context_info = f"This is chunk {chunk_id + 1} of {total_chunks}."
        
        if is_first:
            context_info += " This is the BEGINNING."
        elif is_last:
            context_info += " This is the FINAL chunk."
        else:
            context_info += " This is a MIDDLE section."
        
        if language == "繁體中文" or language == "中文":
            return f"{base_prompt}\n\n**重要提示：** {context_info}"
        else:
            return f"{base_prompt}\n\n**CONTEXT:** {context_info}"

# ==================== LANGUAGE DETECTOR ====================

class LanguageDetector:
    """Language detection"""
    
    @staticmethod
    @functools.lru_cache(maxsize=500)
    def detect_language(text_sample: str) -> str:
        """Detect language"""
        if not text_sample or len(text_sample.strip()) < 10:
            return "English"
        
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
        """Get code"""
        if language == "Chinese":
            return "繁體中文"
        else:
            return "English"

# ==================== YOUTUBE AUDIO EXTRACTOR ====================

class YouTubeAudioExtractor:
    """Extract audio URLs"""
    
    def extract_audio_url(self, youtube_url: str, browser: str = 'chrome') -> Optional[str]:
        """Extract audio URL"""
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
    """Base provider"""
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class OptimizedSupadataProvider(TranscriptProvider):
    """Supadata provider"""
    
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
    
    def _normalize_transcript(self, resp) -> str:
        """Normalize response"""
        if resp is None:
            return ""
        
        resp_str = str(resp)
        if "BatchJob" in resp_str or "job_id" in resp_str:
            return ""

        if isinstance(resp, str):
            if "BatchJob" in resp or "job_id" in resp:
                return ""
            return resp

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
    """AssemblyAI provider"""
    
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
        """Check circuit"""
        if self.circuit_open:
            if datetime.now() - self.circuit_open_time > self.CIRCUIT_TIMEOUT:
                self.circuit_open = False
                self.consecutive_failures = 0
                return False
            return True
        return False
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        """Get transcript"""
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
        """Handle failure"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.MAX_FAILURES:
            self.circuit_open = True
            self.circuit_open_time = datetime.now()
    
    def _transcribe_standard(self, url: str, language: str) -> Optional[str]:
        """Transcribe"""
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

class OptimizedDeepSeekProvider:
    """DeepSeek provider"""
    
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
        """Process transcript"""
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
        """Parallel processing"""
        summary_transcript = self.text_chunker.prepare_text_for_summary(transcript)
        
        summary_prompt = self._create_summary_prompt(language)
        adapted_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            st.info("⚡ Parallel processing...")
            
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
        st.info("Step 1: Executive summary...")
        summary_prompt = self._create_summary_prompt(language)
        executive_summary = self._process_for_summary(transcript, summary_prompt, language)
        
        if executive_summary and progress_callback:
            progress_callback("executive_summary", executive_summary)
        
        st.info("Step 2: Detailed transcript...")
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
        """API request"""
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
            raise RuntimeError("No choices")
        
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
        """Summary with chunking"""
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{summary_prompt}\n\nChunk {i+1}/{len(chunks)}"
            result = self._make_api_request(chunk['text'], chunk_prompt)
            if result:
                chunk_summaries.append(result)
        
        if not chunk_summaries:
            return None
        
        combined = "\n\n".join(chunk_summaries)
        final_prompt = "Combine into one summary:" if language == "English" else "合併成一個摘要："
        
        return self._make_api_request(combined, final_prompt)
    
    def _process_for_detailed_structure(self, transcript: str, system_prompt: str, 
                                       language: str) -> Optional[str]:
        """Detailed structure"""
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
        """Parallel chunks"""
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
        """Sequential chunks"""
        results = []
        
        for i, chunk in enumerate(chunks):
            st.info(f"Chunk {i+1}/{len(chunks)}")
            
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
        """Combine chunks"""
        if len(chunks) == 1:
            return chunks[0]
        
        separator = "\n\n---\n\n"
        header = "# Detailed Transcript\n\n" if language == "English" else "# 詳細轉錄\n\n"
        
        return header + separator.join(chunks)
    
    def _adapt_system_prompt_to_language(self, system_prompt: str, detected_language: str, 
                                        is_custom_prompt: bool = False) -> str:
        """Adapt prompt"""
        if is_custom_prompt:
            instruction = "\n\n**Output in English.**" if detected_language == "English" else "\n\n**請用繁體中文輸出。**"
            return system_prompt + instruction
        
        if detected_language == "繁體中文":
            return """你是專業的YouTube轉錄專家。請結構化轉錄：

1. 創建清晰章節
2. 提高可讀性
3. 保留所有信息
4. 使用markdown格式

用繁體中文輸出。"""
        
        else:
            return """You are a YouTube transcript expert. Structure the transcript:

1. Create clear sections
2. Improve readability
3. Preserve all information
4. Use markdown formatting

Output in English."""
    
    def _create_summary_prompt(self, language: str) -> str:
        """Summary prompt"""
        if language == "繁體中文":
            return """創建執行摘要：
1. 概述
2. 關鍵點
3. 重要細節
4. 結論

用繁體中文輸出。"""
        
        else:
            return """Create executive summary:
1. Overview
2. Key Points
3. Important Details
4. Conclusions

Output in English."""

# ==================== YOUTUBE DATA PROVIDER ====================

class YouTubeDataProvider:
    """YouTube Data API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.available = bool(api_key)
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        """Get playlist videos"""
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
                    title = snippet.get('title', 'Unknown')
                    
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
        """Extract playlist ID"""
        match = re.search(r'list=([\w-]+)', url)
        return match.group(1) if match else None

# ==================== ORCHESTRATOR ====================

class EnhancedTranscriptOrchestrator:
    """Orchestrator"""
    
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
        """Get transcript"""
        transcript = self.transcript_provider.get_transcript(url, language)
        
        def is_valid(text):
            if not text or "BatchJob" in str(text):
                return False
            if len(text.strip()) < 100 or len(text.split()) < 20:
                return False
            return True
        
        if not is_valid(transcript):
            transcript = None
        
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("ASR fallback...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
            if not is_valid(transcript):
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
    """Cached DB"""
    return DatabaseManager()

@st.cache_resource
def get_auth_manager():
    """Cached auth"""
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
    st.subheader("Login")
    
    st.info("**Default:** Username: `admin` | Password: `admin123`")
    
    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Enter credentials")
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
    """History page"""
    st.header("📚 History")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    if 'history_page' not in st.session_state:
        st.session_state.history_page = 0
    if 'history_filters' not in st.session_state:
        st.session_state.history_filters = {}
    
    stats = data_manager.get_history_statistics(username)
    
    if stats["total_videos"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Videos", stats["total_videos"])
        with col2:
            st.metric("Avg Length", f"{stats['average_transcript_length']:,}")
        with col3:
            languages = ", ".join(stats["languages_used"][:2])
            st.metric("Languages", languages if languages else "N/A")
        with col4:
            st.metric("Success", stats.get("success_rate", "0/0"))
        
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input("🔍 Search")
            if search != st.session_state.history_filters.get('search', ''):
                st.session_state.history_filters['search'] = search
                st.session_state.history_page = 0
        
        with col2:
            status = st.selectbox("Status", ["All", "Completed", "Failed"])
            if status != "All":
                st.session_state.history_filters['status'] = status
            elif 'status' in st.session_state.history_filters:
                del st.session_state.history_filters['status']
        
        with col3:
            lang = st.selectbox("Language", ["All"] + stats["languages_used"])
            if lang != "All":
                st.session_state.history_filters['language'] = lang
            elif 'language' in st.session_state.history_filters:
                del st.session_state.history_filters['language']
        
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
                    if st.button("👁️", key=f"view_{entry.get('id')}", type="primary"):
                        st.session_state.current_page = "history_detail"
                        st.session_state.current_entry_id = entry.get('id')
                        st.rerun()
                
                with col4:
                    if st.button("🗑️", key=f"del_{entry.get('id')}", type="secondary"):
                        data_manager.delete_history_entry(username, entry.get('id'))
                        st.rerun()
                
                st.divider()
            
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
            st.info("No videos match filters")
    
    else:
        st.info("No history yet")

def show_history_detail_page(entry_id: str):
    """Detail page"""
    st.header("🔍 Details")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    entry = data_manager.get_history_entry_full(username, entry_id)
    
    if not entry:
        st.error("Not found")
        if st.button("← Back"):
            st.session_state.current_page = "history"
            st.rerun()
        return
    
    if st.button("← Back", type="secondary"):
        st.session_state.current_page = "history"
        st.rerun()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if entry.get('video_thumbnail'):
            st.image(entry['video_thumbnail'], width=200)
    
    with col2:
        st.title(entry.get('video_title', 'Unknown'))
        if entry.get('video_channel'):
            st.write(f"**Channel:** {entry['video_channel']}")
        st.write(f"**Status:** {entry.get('status')}")
    
    st.divider()
    
    if entry.get('status') == 'Completed':
        if entry.get('executive_summary'):
            st.subheader("📋 Summary")
            st.markdown(entry['executive_summary'])
            st.download_button(
                "Download Summary",
                entry['executive_summary'],
                file_name=f"summary_{entry_id}.md"
            )
            st.divider()
        
        if entry.get('detailed_transcript'):
            st.subheader("📝 Transcript")
            st.markdown(entry['detailed_transcript'])
            st.download_button(
                "Download Transcript",
                entry['detailed_transcript'],
                file_name=f"transcript_{entry_id}.md"
            )
    
    elif entry.get('status') == 'Failed':
        st.error(f"Failed: {entry.get('error_message', 'Unknown')}")

def show_settings_page():
    """Settings page"""
    st.header("⚙️ Settings")
    
    username = st.session_state.username
    data_manager = get_data_manager()
    
    current = data_manager.get_user_settings(username)
    
    with st.form("settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            lang = st.selectbox("Language", ["English", "中文"], 
                              index=0 if current.get("language") == "English" else 1)
            use_asr = st.checkbox("ASR Fallback", value=current.get("use_asr_fallback", True))
            model = st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"],
                               index=0 if current.get("deepseek_model") == "deepseek-chat" else 1)
        
        with col2:
            temp = st.slider("Temperature", 0.0, 1.0, current.get("temperature", 0.1), 0.1)
            browser = st.selectbox("Browser", ["none", "chrome", "firefox"],
                                 index=["none", "chrome", "firefox"].index(current.get("browser_for_cookies", "none")))
            perf = st.selectbox("Performance", ["speed", "balanced", "quality"],
                              index=["speed", "balanced", "quality"].index(current.get("performance_mode", "balanced")))
        
        if st.form_submit_button("💾 Save", type="primary"):
            new_settings = {
                "language": lang,
                "use_asr_fallback": use_asr,
                "deepseek_model": model,
                "temperature": temp,
                "browser_for_cookies": browser,
                "performance_mode": perf
            }
            
            data_manager.save_user_settings(username, new_settings)
            st.session_state.user_settings = new_settings
            st.success("Saved!")
            st.rerun()

def show_main_processing_page():
    """Main page"""
    api_keys = st.session_state.get('api_keys', {})
    user_settings = st.session_state.get('user_settings', {})
    
    st.header("Configuration")
    
with st.expander("API Status", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if api_keys.get('supadata'):
            st.success("Supadata ✅")
        else:
            st.error("Supadata ❌")
    
    with col2:
        if api_keys.get('assemblyai'):
            st.success("AssemblyAI ✅")
        else:
            st.error("AssemblyAI ❌")
    
    with col3:
        if api_keys.get('deepseek'):
            st.success("DeepSeek ✅")
        else:
            st.error("DeepSeek ❌")
    
    with col4:
        if api_keys.get('youtube'):
            st.success("YouTube ✅")
        else:
            st.error("YouTube ❌")
    
    with st.expander("Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox("Language", ["English", "中文"],
                                  index=0 if user_settings.get("language") == "English" else 1)
            use_asr = st.checkbox("ASR", value=user_settings.get("use_asr_fallback", True))
            perf_mode = st.selectbox("Mode", ["speed", "balanced", "quality"],
                                   index=["speed", "balanced", "quality"].index(user_settings.get("performance_mode", "balanced")))
        
        with col2:
            model = st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"],
                               index=0 if user_settings.get("deepseek_model") == "deepseek-chat" else 1)
            temp = st.slider("Temp", 0.0, 1.0, user_settings.get("temperature", 0.1), 0.1)
            browser = st.selectbox("Browser", ["none", "chrome", "firefox"],
                                 index=["none", "chrome", "firefox"].index(user_settings.get("browser_for_cookies", "none")))
    
    st.header("Process")
    
    input_method = st.radio("Method", ["Single URL", "Direct Input", "Playlist", "Batch"])
    
    videos_to_process = []
    
    if input_method == "Single URL":
        url = st.text_input("YouTube URL")
        if url:
            videos_to_process = [{"title": "Video", "url": url, "type": "url"}]
    
    elif input_method == "Direct Input":
        title = st.text_input("Title")
        text = st.text_area("Transcript", height=300)
        
        if text:
            st.metric("Chars", f"{len(text):,}")
            
            if title and text:
                videos_to_process = [{
                    "title": title,
                    "url": f"direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "type": "direct_transcript",
                    "transcript": text
                }]
    
    elif input_method == "Playlist":
        playlist_url = st.text_input("Playlist URL")
        if playlist_url and api_keys.get('youtube'):
            if st.button("Load"):
                with st.spinner("Loading..."):
                    provider = YouTubeDataProvider(api_keys['youtube'])
                    videos = provider.get_playlist_videos(playlist_url)
                    if videos:
                        for v in videos:
                            v["type"] = "url"
                        st.success(f"Loaded {len(videos)}")
                        st.session_state.playlist_videos = videos
    
    elif input_method == "Batch":
        batch_urls = st.text_area("URLs (one per line)")
        if batch_urls:
            urls = [u.strip() for u in batch_urls.split('\n') if u.strip()]
            videos_to_process = [{"title": f"Video {i+1}", "url": url, "type": "url"} 
                               for i, url in enumerate(urls)]
    
    if 'playlist_videos' in st.session_state and input_method == "Playlist":
        videos_to_process = st.session_state.playlist_videos
    
    if videos_to_process:
        st.subheader(f"To Process ({len(videos_to_process)})")
        
        default_prompt = """You are an expert at structuring YouTube transcripts.

Structure the transcript:
1. Create clear sections
2. Improve readability
3. Preserve information
4. Use markdown

Output in English."""
        
        system_prompt = st.text_area("System Prompt", value=default_prompt, height=200)
        
        mode_emoji = {"speed": "⚡", "balanced": "⚖️", "quality": "🎯"}
        if st.button(f"Start {mode_emoji[perf_mode]} {perf_mode.title()}", type="primary"):
            process_videos(videos_to_process, language, use_asr, system_prompt, 
                         model, temp, api_keys, browser, perf_mode)

def process_videos(videos, language, use_asr, system_prompt, model, temp, api_keys, browser, perf_mode):
    """Process videos"""
    username = st.session_state.username
    data_manager = get_data_manager()
    
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
        st.write(f"**{video['title']}**")
        
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
            if video.get("type") == "direct_transcript":
                transcript = video.get("transcript", "")
            else:
                with st.spinner("Getting transcript..."):
                    transcript = orchestrator.get_transcript(video['url'], language, use_asr)
            
            if not transcript:
                raise Exception("No transcript")
            
            history_entry["transcript_length"] = len(transcript)
            data_manager.update_history_entry(username, entry_id, history_entry)
            
            with st.spinner("Structuring..."):
                summary_placeholder = st.empty()
                
                def progress_callback(component_type, content):
                    if component_type == "executive_summary":
                        with summary_placeholder.container():
                            st.subheader("📋 Summary")
                            st.markdown(content)
                
                result = orchestrator.structure_transcript_with_live_updates(
                    transcript, system_prompt, language, False, progress_callback
                )
            
            if not result:
                raise Exception("Failed to structure")
            
            processing_time = time.time() - start_time
            history_entry["processing_time"] = f"{processing_time:.1f}s"
            history_entry["status"] = "Completed"
            history_entry["executive_summary"] = result.get('executive_summary')
            history_entry["detailed_transcript"] = result.get('detailed_transcript')
            
            data_manager.update_history_entry(username, entry_id, history_entry)
            
st.success("🎉 Done!")

# Show executive summary if available
if result.get('executive_summary'):
    st.subheader("📋 Executive Summary")
    st.markdown(result['executive_summary'])
    st.divider()

# Show detailed transcript
if result.get('detailed_transcript'):
    st.subheader("📝 Detailed Structured Transcript")
    st.markdown(result['detailed_transcript'])
    st.divider()

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
    """Main app"""
    st.title("YouTube Transcript Processor")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    with st.sidebar:
        st.write(f"**{st.session_state.username}**")
        st.divider()
        
        if st.session_state.current_page != "history_detail":
            pages = ["🎬 Process", "📚 History", "⚙️ Settings"]
            idx = 0
            
            if st.session_state.current_page == "history":
                idx = 1
            elif st.session_state.current_page == "settings":
                idx = 2
            
            page = st.radio("Navigation", pages, index=idx)
            
            if page == "🎬 Process":
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
    
    if st.session_state.current_page == "history_detail":
        entry_id = st.session_state.get('current_entry_id')
        if entry_id:
            show_history_detail_page(entry_id)
        else:
            st.session_state.current_page = "history"
            st.rerun()
    elif st.session_state.current_page == "history":
        show_history_page()
    elif st.session_state.current_page == "settings":
        show_settings_page()
    else:
        show_main_processing_page()

# ==================== MAIN ====================

def main():
    """Main entry"""
    st.set_page_config(
        page_title="YouTube Transcript Processor",
        page_icon="⚡",
        layout="wide"
    )
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
