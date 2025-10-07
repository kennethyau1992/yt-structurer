import os
import re
import json
import time
import hashlib
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from functools import wraps
import requests
import streamlit as st
import threading
from queue import Queue

# ==================== OPTIMIZATION 1: CONNECTION POOLING ====================

class ConnectionPool:
    """Singleton connection pool for HTTP requests"""
    _instance = None
    _session = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionPool, cls).__new__(cls)
            cls._session = requests.Session()
            cls._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
        return cls._instance
    
    @classmethod
    def get_session(cls) -> requests.Session:
        if cls._session is None:
            cls()
        return cls._session

# ==================== OPTIMIZATION 2: PRECOMPILED REGEX ====================

class RegexPatterns:
    """Precompiled regex patterns"""
    VIDEO_ID = re.compile(r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)')
    VIDEO_ID_ALT = re.compile(r'youtube\.com/v/([^&\n?#]+)')
    PLAYLIST_ID = re.compile(r'list=([\w-]+)')
    SECTION_BREAK = re.compile(r'\n\s*\n')
    SENTENCE_END = re.compile(r'[.!?]\s+')
    NEWLINE = re.compile(r'\n')

# ==================== OPTIMIZATION 3: TTL CACHING ====================

def cache_with_ttl(seconds=300):
    """Cache decorator with TTL"""
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, str(args), str(sorted(kwargs.items())))
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < seconds:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator

# ==================== OPTIMIZATION 4: SESSION CACHING ====================

class SessionCache:
    """Session-level caching"""
    
    @staticmethod
    def get_user_data(username: str, user_data_manager) -> Dict:
        cache_key = f"user_data_{username}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = user_data_manager.load_user_data(username)
        return st.session_state[cache_key]
    
    @staticmethod
    def invalidate_user_data(username: str):
        cache_key = f"user_data_{username}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]
    
    @staticmethod
    def get_user_settings(username: str, user_data_manager) -> Dict:
        cache_key = f"user_settings_{username}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = user_data_manager.get_user_settings(username)
        return st.session_state[cache_key]
    
    @staticmethod
    def invalidate_user_settings(username: str):
        cache_key = f"user_settings_{username}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]

# ==================== YOUTUBE INFO EXTRACTOR ====================

class YouTubeInfoExtractor:
    """Extract YouTube video information with optimizations"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.session = ConnectionPool.get_session()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        match = RegexPatterns.VIDEO_ID.search(url)
        if match:
            return match.group(1)
        match = RegexPatterns.VIDEO_ID_ALT.search(url)
        if match:
            return match.group(1)
        return None
    
    @cache_with_ttl(seconds=3600)
    def get_video_info(self, url: str) -> Dict[str, str]:
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}
        
        if self.api_key:
            api_info = self._get_info_from_api(video_id)
            if api_info:
                return api_info
        
        return self._get_info_from_ytdlp(url)
    
    def _get_info_from_api(self, video_id: str) -> Optional[Dict[str, str]]:
        try:
            params = {
                'part': 'snippet,contentDetails',
                'id': video_id,
                'key': self.api_key
            }
            
            response = self.session.get(
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
                    return {
                        "title": snippet.get('title', 'Unknown Video'),
                        "description": description[:500] + "..." if len(description) > 500 else description,
                        "duration": content_details.get('duration', ''),
                        "channel": snippet.get('channelTitle', ''),
                        "published_at": snippet.get('publishedAt', ''),
                        "thumbnail": snippet.get('thumbnails', {}).get('medium', {}).get('url', '')
                    }
        except Exception as e:
            st.warning(f"YouTube API failed: {e}")
        return None
    
    def _get_info_from_ytdlp(self, url: str) -> Dict[str, str]:
        try:
            import yt_dlp
            ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                description = (info.get('description', '') or '')
                return {
                    "title": info.get('title', 'Unknown Video'),
                    "description": description[:500] + "..." if len(description) > 500 else description,
                    "duration": str(info.get('duration', '')),
                    "channel": info.get('uploader', ''),
                    "published_at": info.get('upload_date', ''),
                    "thumbnail": info.get('thumbnail', '')
                }
        except Exception as e:
            st.warning(f"yt-dlp info extraction failed: {e}")
        return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}

# ==================== USER DATA MANAGER ====================

class UserDataManager:
    """Manages user settings and history"""
    
    def __init__(self, base_dir: str = "user_data"):
        self.base_dir = base_dir
        self.ensure_data_directory()
        self.youtube_extractor = None
    
    def set_youtube_extractor(self, youtube_api_key: str = None):
        self.youtube_extractor = YouTubeInfoExtractor(youtube_api_key)
    
    def ensure_data_directory(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def get_user_data_path(self, username: str) -> str:
        return os.path.join(self.base_dir, f"{username}_data.json")
    
    def load_user_data(self, username: str) -> Dict:
        data_path = self.get_user_data_path(username)
        default_data = {
            "settings": {
                "language": "English",
                "use_asr_fallback": True,
                "deepseek_model": "deepseek-reasoner",
                "temperature": 0.1,
                "browser_for_cookies": "none",
                "performance_mode": "balanced"
            },
            "history": [],
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key in default_data:
                    if key not in data:
                        data[key] = default_data[key]
                for setting_key in default_data["settings"]:
                    if setting_key not in data["settings"]:
                        data["settings"][setting_key] = default_data["settings"][setting_key]
                return data
            else:
                return default_data
        except Exception as e:
            st.warning(f"Could not load user data: {e}. Using defaults.")
            return default_data
    
    def save_user_data(self, username: str, data: Dict):
        data_path = self.get_user_data_path(username)
        data["last_updated"] = datetime.now().isoformat()
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            SessionCache.invalidate_user_data(username)
        except Exception as e:
            st.error(f"Could not save user data: {e}")
    
    def save_user_settings(self, username: str, settings: Dict):
        data = SessionCache.get_user_data(username, self)
        data["settings"].update(settings)
        self.save_user_data(username, data)
        SessionCache.invalidate_user_settings(username)
    
    def get_user_settings(self, username: str) -> Dict:
        data = SessionCache.get_user_data(username, self)
        return data["settings"]
    
    def add_to_history(self, username: str, entry: Dict):
        data = SessionCache.get_user_data(username, self)
        
        if self.youtube_extractor and entry.get('url'):
            video_info = self.youtube_extractor.get_video_info(entry['url'])
            entry.update({
                "video_title": video_info.get("title", entry.get("title", "Unknown Video")),
                "video_description": video_info.get("description", ""),
                "video_duration": video_info.get("duration", ""),
                "video_channel": video_info.get("channel", ""),
                "video_published": video_info.get("published_at", ""),
                "video_thumbnail": video_info.get("thumbnail", "")
            })
        else:
            entry["video_title"] = entry.get("title", "Unknown Video")
        
        entry["timestamp"] = datetime.now().isoformat()
        entry["id"] = hashlib.md5(f"{entry['url']}_{entry['timestamp']}".encode()).hexdigest()[:8]
        data["history"].insert(0, entry)
        data["history"] = data["history"][:100]
        self.save_user_data(username, data)
    
    def update_history_entry(self, username: str, entry_id: str, updates: Dict):
        data = SessionCache.get_user_data(username, self)
        for entry in data["history"]:
            if entry.get("id") == entry_id:
                entry.update(updates)
                self.save_user_data(username, data)
                return True
        return False
    
    def get_user_history(self, username: str) -> List[Dict]:
        data = SessionCache.get_user_data(username, self)
        return data["history"]
    
    def get_history_entry(self, username: str, entry_id: str) -> Optional[Dict]:
        history = self.get_user_history(username)
        for entry in history:
            if entry.get("id") == entry_id:
                return entry
        return None
    
    def delete_history_entry(self, username: str, entry_id: str):
        data = SessionCache.get_user_data(username, self)
        data["history"] = [entry for entry in data["history"] if entry.get("id") != entry_id]
        self.save_user_data(username, data)
    
    def clear_user_history(self, username: str):
        data = SessionCache.get_user_data(username, self)
        data["history"] = []
        self.save_user_data(username, data)
    
    def export_user_data(self, username: str) -> str:
        data = SessionCache.get_user_data(username, self)
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @cache_with_ttl(seconds=60)
    def get_history_statistics(self, username: str) -> Dict:
        history = self.get_user_history(username)
        if not history:
            return {
                "total_videos": 0,
                "total_transcript_length": 0,
                "average_transcript_length": 0,
                "most_recent": None,
                "languages_used": [],
                "models_used": []
            }
        
        total_length = sum(entry.get("transcript_length", 0) for entry in history)
        languages = list(set(entry.get("language", "Unknown") for entry in history))
        models = list(set(entry.get("model_used", "Unknown") for entry in history))
        
        return {
            "total_videos": len(history),
            "total_transcript_length": total_length,
            "average_transcript_length": total_length // len(history) if history else 0,
            "most_recent": history[0].get("timestamp") if history else None,
            "languages_used": languages,
            "models_used": models
        }

# ==================== AUTH MANAGER ====================

class AuthManager:
    def __init__(self):
        self.users = {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "api_keys": {
                    "supadata": os.getenv("ADMIN_SUPADATA_KEY", ""),
                    "assemblyai": os.getenv("ADMIN_ASSEMBLYAI_KEY", ""),
                    "deepseek": os.getenv("ADMIN_DEEPSEEK_KEY", ""),
                    "youtube": os.getenv("ADMIN_YOUTUBE_KEY", "")
                }
            }
        }
        self.user_data_manager = UserDataManager()
        self._load_users_from_env()
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users_from_env(self):
        env_users = {}
        for key, value in os.environ.items():
            if key.endswith('_PASSWORD'):
                username = key[:-9].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["password_hash"] = self._hash_password(value)
            elif key.endswith('_SUPADATA_KEY'):
                username = key[:-13].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["supadata"] = value
            elif key.endswith('_ASSEMBLYAI_KEY'):
                username = key[:-14].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["assemblyai"] = value
            elif key.endswith('_DEEPSEEK_KEY'):
                username = key[:-12].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["deepseek"] = value
            elif key.endswith('_YOUTUBE_KEY'):
                username = key[:-12].lower()
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["youtube"] = value
        
        for username, user_data in env_users.items():
            if username in self.users:
                self.users[username]["api_keys"].update(user_data["api_keys"])
            else:
                self.users[username] = user_data
    
    def authenticate(self, username: str, password: str) -> bool:
        if username in self.users:
            return self.users[username]["password_hash"] == self._hash_password(password)
        return False
    
    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        if username in self.users:
            return self.users[username]["api_keys"]
        return {}
    
    def get_user_data_manager(self) -> UserDataManager:
        return self.user_data_manager
    
    def initialize_user_data_manager_with_youtube(self, youtube_api_key: str = None):
        self.user_data_manager.set_youtube_extractor(youtube_api_key)

# ==================== YOUTUBE COOKIE MANAGER ====================

class YouTubeCookieManager:
    @staticmethod
    def get_ydl_opts(use_cookies: bool = True, browser: str = 'chrome') -> dict:
        base_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False}
        
        if use_cookies:
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                import tempfile, shutil
                try:
                    temp_cookies = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    temp_cookies_path = temp_cookies.name
                    temp_cookies.close()
                    shutil.copy2(render_cookies, temp_cookies_path)
                    base_opts['cookiefile'] = temp_cookies_path
                except Exception:
                    pass
            else:
                cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
                if cookies_file and os.path.exists(cookies_file):
                    base_opts['cookiefile'] = cookies_file
        
        base_opts['headers'] = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        return base_opts

# ==================== CHUNKING ====================

class PerformanceOptimizedChunker:
    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.configs = {
            "speed": {"max_chunk_length": 6000, "overlap_length": 100, "max_parallel_workers": 4, 
                     "summary_truncate_length": 15000, "processing_timeout": 45},
            "balanced": {"max_chunk_length": 8000, "overlap_length": 200, "max_parallel_workers": 3,
                        "summary_truncate_length": 25000, "processing_timeout": 90},
            "quality": {"max_chunk_length": 12000, "overlap_length": 400, "max_parallel_workers": 2,
                       "summary_truncate_length": None, "processing_timeout": 180}
        }
        self.config = self.configs.get(performance_mode, self.configs["balanced"])
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
        truncate_length = self.config.get("summary_truncate_length")
        if truncate_length and len(text) > truncate_length:
            half = truncate_length // 2
            beginning = text[:half]
            end = text[-half:]
            truncated_text = beginning + "\n\n[... truncated ...]\n\n" + end
            st.info(f"🚀 Smart truncation ({len(text):,} → {len(truncated_text):,} chars)")
            return truncated_text
        return text
    
    def create_intelligent_chunks(self, text: str) -> List[Dict[str, Any]]:
        if not self.should_chunk_text(text):
            return [{'chunk_id': 0, 'start_pos': 0, 'end_pos': len(text), 'text': text, 
                    'tokens_estimate': len(text) // 4, 'is_single_chunk': True}]
        
        split_points = self._find_split_points(text)
        chunks = []
        current_start = 0
        chunk_id = 0
        
        while current_start < len(text):
            target_end = current_start + self.config["max_chunk_length"]
            
            if target_end >= len(text):
                chunk_text = text[current_start:]
                chunks.append({'chunk_id': chunk_id, 'start_pos': current_start, 'end_pos': len(text),
                             'text': chunk_text, 'tokens_estimate': len(chunk_text) // 4, 'is_final_chunk': True})
                break
            
            best_split = target_end
            for split_point in reversed(split_points):
                if current_start + self.min_chunk_length <= split_point <= target_end:
                    best_split = split_point
                    break
            
            chunk_text = text[current_start:best_split]
            chunks.append({'chunk_id': chunk_id, 'start_pos': current_start, 'end_pos': best_split,
                         'text': chunk_text, 'tokens_estimate': len(chunk_text) // 4, 'is_final_chunk': False})
            
            current_start = max(current_start + self.min_chunk_length, best_split - self.config["overlap_length"])
            chunk_id += 1
        
        return chunks
    
    def _find_split_points(self, text: str) -> List[int]:
        split_points = []
        for match in RegexPatterns.SECTION_BREAK.finditer(text):
            split_points.append(match.end())
        for match in RegexPatterns.SENTENCE_END.finditer(text):
            split_points.append(match.end())
        for match in RegexPatterns.NEWLINE.finditer(text):
            split_points.append(match.end())
        return sorted(list(set(split_points)))
    
    def create_chunk_specific_prompt(self, base_prompt: str, chunk_info: Dict, total_chunks: int, language: str) -> str:
        chunk_id = chunk_info['chunk_id']
        is_first = chunk_id == 0
        is_last = chunk_info.get('is_final_chunk', False)
        
        context_info = f"This is chunk {chunk_id + 1} of {total_chunks}."
        if is_first:
            context_info += " BEGINNING of transcript."
        elif is_last:
            context_info += " FINAL chunk."
        else:
            context_info += " MIDDLE section."
        
        if language in ["繁體中文", "中文"]:
            return f"{base_prompt}\n\n**重要：**{context_info}"
        else:
            return f"{base_prompt}\n\n**CONTEXT:**{context_info}"

# ==================== AUDIO CHUNKER ====================

class AudioChunker:
    def __init__(self, chunk_duration_minutes: int = 12):
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.max_parallel_chunks = 3
        self.cookie_manager = YouTubeCookieManager()
    
    def get_video_duration(self, youtube_url: str, browser: str = 'chrome') -> Optional[int]:
        try:
            import yt_dlp
            ydl_opts = self.cookie_manager.get_ydl_opts(use_cookies=True, browser=browser)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                duration = info.get('duration')
                return int(duration) if duration else None
        except Exception:
            return None
    
    def transcribe_chunks_parallel(self, chunks: List[Dict], assemblyai_provider, language: str) -> Optional[str]:
        if not chunks:
            return None
        
        def transcribe_single_chunk(chunk_info):
            try:
                transcript = assemblyai_provider._transcribe_audio_url(chunk_info['audio_url'], language)
                return {'chunk_id': chunk_info['chunk_id'], 'transcript': transcript, 'success': transcript is not None}
            except Exception as e:
                return {'chunk_id': chunk_info['chunk_id'], 'transcript': None, 'success': False, 'error': str(e)}
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
            future_to_chunk = {executor.submit(transcribe_single_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    results.append(future.result())
                except Exception:
                    pass
        
        results.sort(key=lambda x: x['chunk_id'])
        successful = [r['transcript'] for r in results if r['success'] and r['transcript']]
        
        if not successful:
            return None
        
        return "\n\n".join(successful)

# ==================== PROVIDERS ====================

class TranscriptProvider:
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class SupadataTranscriptProvider(TranscriptProvider):
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
        except Exception:
            return None
    
    def _normalize_transcript(self, resp) -> str:
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
            t = resp.get("text")
            if isinstance(t, str):
                return t
        return ""

class YouTubeAudioExtractor:
    def __init__(self):
        self.cookie_manager = YouTubeCookieManager()
    
    def extract_audio_url(self, youtube_url: str, browser: str = 'chrome') -> Optional[str]:
        try:
            import yt_dlp
            ydl_opts = {'format': 'bestaudio/best', 'noplaylist': True, 'quiet': True}
            
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                ydl_opts['cookiefile'] = render_cookies
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if info:
                    formats = info.get('formats', [])
                    audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                    if audio_formats:
                        return max(audio_formats, key=lambda x: x.get('abr', 0) or 0).get('url')
            return None
        except Exception:
            return None

class ImprovedAssemblyAITranscriptProvider(TranscriptProvider):
    def __init__(self, api_key: str, browser: str = 'chrome'):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
        self.chunker = AudioChunker()
        self.browser = browser
        self.session = ConnectionPool.get_session()
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            return None
        audio_url = self.audio_extractor.extract_audio_url(url, self.browser)
        if not audio_url:
            return None
        return self._transcribe_audio_url(audio_url, language)
    
    def _transcribe_audio_url(self, audio_url: str, language: str) -> Optional[str]:
        try:
            headers = {"authorization": self.api_key, "content-type": "application/json"}
            language_map = {"English": "en", "中文": "zh", "en": "en", "zh": "zh"}
            data = {"audio_url": audio_url, "language_code": language_map.get(language, "en"), "speech_model": "best"}
            
            response = self.session.post(f"{self.base_url}/transcript", json=data, headers=headers, timeout=30)
            if response.status_code != 200:
                return None
            
            transcript_id = response.json()['id']
            return self._poll_for_completion(transcript_id, headers)
        except Exception:
            return None
    
    def _poll_for_completion(self, transcript_id: str, headers: dict) -> Optional[str]:
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        for _ in range(120):
            try:
                response = self.session.get(polling_endpoint, headers=headers, timeout=30)
                if response.status_code != 200:
                    return None
                data = response.json()
                status = data.get('status')
                if status == 'completed':
                    return data.get('text', '')
                elif status == 'error':
                    return None
                time.sleep(3)
            except Exception:
                return None
        return None

class LanguageDetector:
    @staticmethod
    def detect_language(text: str) -> str:
        if not text or len(text.strip()) < 10:
            return "English"
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total = chinese + english
        if total == 0:
            return "English"
        return "Chinese" if chinese / total > 0.3 else "English"
    
    @staticmethod
    def get_language_code(language: str) -> str:
        return "繁體中文" if language == "Chinese" else "English"

# ==================== DEEPSEEK PROVIDER ====================

class EnhancedDeepSeekProvider:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, performance_mode: str = "balanced"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.performance_mode = performance_mode
        self.session = ConnectionPool.get_session()
        self.text_chunker = PerformanceOptimizedChunker(performance_mode)
        self.language_detector = LanguageDetector()
        self.config = self.text_chunker.get_processing_config()
        self.processing_timeout = self.config["processing_timeout"]
    
    def structure_transcript_with_live_updates(self, transcript: str, system_prompt: str, ui_language: str = "English", 
                                             is_custom_prompt: bool = False, progress_callback=None) -> Optional[Dict[str, str]]:
        try:
            detected_language = self.language_detector.detect_language(transcript)
            actual_language = self.language_detector.get_language_code(detected_language)
            
            if self.performance_mode == "speed":
                return self._process_with_parallel_strategy(transcript, system_prompt, actual_language, 
                                                          is_custom_prompt, progress_callback)
            else:
                return self._process_with_sequential_strategy(transcript, system_prompt, actual_language, 
                                                            is_custom_prompt, progress_callback)
        except Exception as e:
            raise RuntimeError(f"DeepSeek error: {str(e)}")
    
    def _process_with_parallel_strategy(self, transcript: str, system_prompt: str, language: str, 
                                      is_custom_prompt: bool, progress_callback) -> Optional[Dict[str, str]]:
        summary_transcript = self.text_chunker.prepare_text_for_summary(transcript)
        summary_prompt = self._create_summary_prompt(language)
        adapted_system_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            summary_future = executor.submit(self._process_for_summary, summary_transcript, summary_prompt, language)
            detailed_future = executor.submit(self._process_for_detailed_structure, transcript, adapted_system_prompt, language)
            
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
    
    def _process_with_sequential_strategy(self, transcript: str, system_prompt: str, language: str, 
                                        is_custom_prompt: bool, progress_callback) -> Optional[Dict[str, str]]:
        summary_prompt = self._create_summary_prompt(language)
        executive_summary = self._process_for_summary(transcript, summary_prompt, language)
        
        if executive_summary and progress_callback:
            progress_callback("executive_summary", executive_summary)
        
        adapted_system_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        detailed_transcript = self._process_for_detailed_structure(transcript, adapted_system_prompt, language)
        
        if not detailed_transcript:
            return None
        
        return {
            'executive_summary': executive_summary,
            'detailed_transcript': detailed_transcript,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt,
            'performance_mode': self.performance_mode
        }
    
    def _make_api_request(self, text: str, system_prompt: str, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                endpoint = self.base_url + "/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                    "temperature": self.temperature
                }
                
                resp = self.session.post(endpoint, headers=headers, json=payload, timeout=self.processing_timeout)
                
                if resp.status_code != 200:
                    if resp.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 2)
                        continue
                    raise RuntimeError(f"API error {resp.status_code}")
                
                data = resp.json()
                if 'choices' not in data or len(data['choices']) == 0:
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 2)
                        continue
                    raise RuntimeError("No choices in response")
                
                content = data["choices"][0]["message"]["content"]
                if not content:
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 2)
                        continue
                    raise RuntimeError("Empty content")
                
                return content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                raise
        raise RuntimeError("All retries failed")
    
    def _process_for_summary(self, transcript: str, summary_prompt: str, language: str) -> Optional[str]:
        if self.text_chunker.estimate_tokens(transcript) > 12000:
            return self._process_summary_with_chunking(transcript, summary_prompt, language)
        return self._make_api_request(transcript, summary_prompt)
    
    def _process_summary_with_chunking(self, transcript: str, summary_prompt: str, language: str) -> Optional[str]:
        large_chunker = PerformanceOptimizedChunker("quality")
        chunks = large_chunker.create_intelligent_chunks(transcript)
        if len(chunks) == 1:
            return self._make_api_request(transcript, summary_prompt)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{summary_prompt}\n\nCHUNK {i+1}/{len(chunks)}: Summarize key points."
            result = self._make_api_request(chunk['text'], chunk_prompt)
            if result:
                chunk_summaries.append(result)
        
        if not chunk_summaries:
            return None
        
        combined = "\n\n".join(chunk_summaries)
        final_prompt = "Combine these summaries into one unified executive summary." if language == "English" else "將這些摘要合併成一個統一的執行摘要。"
        return self._make_api_request(combined, final_prompt)
    
    def _process_for_detailed_structure(self, transcript: str, system_prompt: str, language: str) -> Optional[str]:
        if not self.text_chunker.should_chunk_text(transcript):
            return self._make_api_request(transcript, system_prompt)
        return self._process_detailed_with_chunking(transcript, system_prompt, language)
    
    def _process_detailed_with_chunking(self, transcript: str, system_prompt: str, language: str) -> Optional[str]:
        chunks = self.text_chunker.create_intelligent_chunks(transcript)
        if len(chunks) == 1:
            return self._make_api_request(transcript, system_prompt)
        
        processed_chunks = []
        for chunk_info in chunks:
            chunk_prompt = self.text_chunker.create_chunk_specific_prompt(system_prompt, chunk_info, len(chunks), language)
            result = self._make_api_request(chunk_info['text'], chunk_prompt)
            if result:
                processed_chunks.append(result)
        
        if not processed_chunks:
            return None
        
        separator = "\n\n---\n\n"
        return separator.join(processed_chunks)
    
    def _adapt_system_prompt_to_language(self, system_prompt: str, detected_language: str, is_custom_prompt: bool = False) -> str:
        if is_custom_prompt:
            suffix = "\n\n**重要：請用繁體中文輸出。**" if detected_language == "繁體中文" else "\n\n**Important: Output in English.**"
            return system_prompt + suffix
        
        if detected_language == "繁體中文":
            return """你是YouTube轉錄文本結構化專家。請：
1. 創建清晰章節和標題
2. 提高可讀性：修正語法、刪除填充詞、合併句子
3. 保留所有重要信息
4. 使用markdown格式
5. 添加時間戳
請用繁體中文輸出。"""
        else:
            return """You are an expert at structuring YouTube transcripts. Please:
1. Create clear sections and headings
2. Improve readability: fix grammar, remove filler words
3. Preserve all important information
4. Use markdown formatting
5. Add timestamps
Output in English."""
    
    def _create_summary_prompt(self, language: str) -> str:
        if language == "繁體中文":
            return """創建執行摘要：
1. 概述（2-3句）
2. 關鍵點（要點形式）
3. 重要細節
4. 結論/要點
請用繁體中文輸出。"""
        else:
            return """Create an executive summary:
1. Overview (2-3 sentences)
2. Key Points (bullet points)
3. Important Details
4. Conclusions/Takeaways
Output in English."""

class YouTubeDataProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.available = bool(api_key)
        self.session = ConnectionPool.get_session()
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        if not self.available:
            return []
        
        match = RegexPatterns.PLAYLIST_ID.search(playlist_url)
        playlist_id = match.group(1) if match else None
        if not playlist_id:
            return []
        
        try:
            videos = []
            next_page_token = None
            
            while True:
                params = {'part': 'snippet', 'playlistId': playlist_id, 'maxResults': 50, 'key': self.api_key}
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                response = self.session.get(f"{self.base_url}/playlistItems", params=params, timeout=30)
                if response.status_code != 200:
                    return []
                
                data = response.json()
                for item in data.get('items', []):
                    video_id = item.get('snippet', {}).get('resourceId', {}).get('videoId')
                    if video_id:
                        videos.append({
                            'title': item.get('snippet', {}).get('title', 'Unknown'),
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'video_id': video_id
                        })
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos
        except Exception:
            return []

# ==================== ORCHESTRATOR ====================

class EnhancedTranscriptOrchestrator:
    def __init__(self, transcript_provider: TranscriptProvider, asr_fallback_provider: Optional[TranscriptProvider] = None,
                 llm_provider: Optional[EnhancedDeepSeekProvider] = None):
        self.transcript_provider = transcript_provider
        self.asr_fallback_provider = asr_fallback_provider
        self.llm_provider = llm_provider
    
    def get_transcript(self, url: str, language: str, use_fallback: bool = False) -> Optional[str]:
        transcript = self.transcript_provider.get_transcript(url, language)
        
        def is_valid(text):
            if not text or "BatchJob" in str(text) or "job_id" in str(text):
                return False
            return len(text.strip()) >= 100 and len(text.split()) >= 20
        
        if not is_valid(transcript):
            transcript = None
            if use_fallback and self.asr_fallback_provider:
                transcript = self.asr_fallback_provider.get_transcript(url, language)
                if not is_valid(transcript):
                    transcript = None
        
        return transcript
    
    def structure_transcript_with_live_updates(self, transcript: str, system_prompt: str, language: str = "English", 
                                             is_custom_prompt: bool = False, progress_callback=None) -> Optional[Dict[str, str]]:
        if not self.llm_provider:
            return None
        return self.llm_provider.structure_transcript_with_live_updates(
            transcript, system_prompt, language, is_custom_prompt, progress_callback
        )

# ==================== UI FUNCTIONS ====================

def login_page():
    st.title("YouTube Transcript Processor")
    st.subheader("Login Required")
    st.info("**Default:** Username: `admin` | Password: `admin123`")
    
    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Enter both username and password")
                return False
            
            auth = AuthManager()
            if auth.authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.api_keys = auth.get_user_api_keys(username)
                st.session_state.user_data_manager = auth.get_user_data_manager()
                
                youtube_key = st.session_state.api_keys.get('youtube')
                auth.initialize_user_data_manager_with_youtube(youtube_key)
                
                user_settings = SessionCache.get_user_settings(username, st.session_state.user_data_manager)
                st.session_state.user_settings = user_settings
                
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                st.error("Invalid credentials")
    return False

def show_history_page():
    st.header("📚 Processing History")
    username = st.session_state.username
    mgr = st.session_state.user_data_manager
    
    history = mgr.get_user_history(username)
    stats = mgr.get_history_statistics(username)
    
    if stats["total_videos"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Videos", stats["total_videos"])
        col2.metric("Avg Length", f"{stats['average_transcript_length']:,}")
        col3.metric("Languages", ", ".join(stats["languages_used"]))
        col4.metric("Success Rate", f"{len([h for h in history if h.get('status') == 'Completed'])}/{len(history)}")
        
        st.divider()
        
        if st.button("🔄 Refresh"):
            SessionCache.invalidate_user_data(username)
            st.rerun()
        
        for i, entry in enumerate(history[:20]):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{entry.get('video_title', 'Unknown')}**")
                    if entry.get('status') == 'Completed':
                        st.success(f"✅ {entry.get('status')}")
                with col2:
                    st.write(f"**Date:** {entry.get('timestamp', '')[:10]}")
                with col3:
                    if st.button("🗑️", key=f"del_{i}"):
                        mgr.delete_history_entry(username, entry.get('id'))
                        st.rerun()
                st.divider()
    else:
        st.info("No history yet.")

def show_settings_page():
    st.header("⚙️ Settings")
    username = st.session_state.username
    mgr = st.session_state.user_data_manager
    current = SessionCache.get_user_settings(username, mgr)
    
    with st.form("settings"):
        col1, col2 = st.columns(2)
        with col1:
            lang = st.selectbox("Language", ["English", "中文"], 
                               index=0 if current.get("language") == "English" else 1)
            asr = st.checkbox("ASR Fallback", value=current.get("use_asr_fallback", True))
        with col2:
            model = st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"],
                                index=0 if current.get("deepseek_model") == "deepseek-chat" else 1)
            mode = st.selectbox("Mode", ["speed", "balanced", "quality"],
                               index=["speed", "balanced", "quality"].index(current.get("performance_mode", "balanced")))
        
        if st.form_submit_button("💾 Save"):
            new_settings = {"language": lang, "use_asr_fallback": asr, 
                          "deepseek_model": model, "performance_mode": mode, "temperature": 0.1, "browser_for_cookies": "none"}
            mgr.save_user_settings(username, new_settings)
            st.session_state.user_settings = new_settings
            SessionCache.invalidate_user_settings(username)
            st.success("Saved!")
            st.rerun()

def show_main_processing_page():
    api_keys = st.session_state.get('api_keys', {})
    settings = SessionCache.get_user_settings(st.session_state.username, st.session_state.user_data_manager)
    
    st.header("Process Video")
    
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if url and st.button("Process"):
        if not api_keys.get('deepseek'):
            st.error("DeepSeek API key required")
            return
        
        videos = [{"title": "Video", "url": url, "type": "url"}]
        process_videos(videos, settings.get("language", "English"), settings.get("use_asr_fallback", True),
                      "", settings.get("deepseek_model", "deepseek-reasoner"), 0.1, api_keys, 
                      "chrome", settings.get("performance_mode", "balanced"))

def process_videos(videos, language, use_asr, system_prompt, model, temp, api_keys, browser, perf_mode):
    username = st.session_state.username
    mgr = st.session_state.user_data_manager
    
    supadata = SupadataTranscriptProvider(api_keys.get('supadata', ''))
    assemblyai = ImprovedAssemblyAITranscriptProvider(api_keys.get('assemblyai', ''), browser) if use_asr else None
    deepseek = EnhancedDeepSeekProvider(api_keys.get('deepseek', ''), "https://api.deepseek.com/v1", model, temp, perf_mode)
    orch = EnhancedTranscriptOrchestrator(supadata, assemblyai, deepseek)
    
    for video in videos:
        st.subheader(f"Processing: {video['title']}")
        
        entry = {"title": video['title'], "url": video['url'], "language": language, "model_used": model,
                "status": "Processing", "performance_mode": perf_mode}
        mgr.add_to_history(username, entry)
        entry_id = entry.get('id')
        
        try:
            with st.spinner("Getting transcript..."):
                transcript = orch.get_transcript(video['url'], language, use_asr)
            
            if not transcript:
                entry["status"] = "Failed"
                mgr.update_history_entry(username, entry_id, entry)
                st.error("Failed to get transcript")
                continue
            
            entry["transcript_length"] = len(transcript)
            mgr.update_history_entry(username, entry_id, entry)
            
            with st.spinner("Structuring..."):
                default_prompt = "Structure this transcript clearly with sections and markdown formatting."
                result = orch.structure_transcript_with_live_updates(transcript, default_prompt, language, False, None)
            
            if result:
                entry["status"] = "Completed"
                entry["executive_summary"] = result.get('executive_summary')
                entry["detailed_transcript"] = result.get('detailed_transcript')
                mgr.update_history_entry(username, entry_id, entry)
                st.success("✅ Completed!")
                
                if result.get('executive_summary'):
                    with st.expander("Summary"):
                        st.markdown(result['executive_summary'])
                if result.get('detailed_transcript'):
                    with st.expander("Transcript"):
                        st.markdown(result['detailed_transcript'])
            else:
                entry["status"] = "Failed"
                mgr.update_history_entry(username, entry_id, entry)
                st.error("Failed to structure")
        except Exception as e:
            entry["status"] = "Failed"
            entry["error"] = str(e)
            mgr.update_history_entry(username, entry_id, entry)
            st.error(f"Error: {e}")

def main_app():
    st.title("YouTube Transcript Processor")
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    with st.sidebar:
        st.write(f"**{st.session_state.username}**")
        st.divider()
        
        page = st.radio("Navigation", ["🎬 Process", "📚 History", "⚙️ Settings"])
        
        if page == "🎬 Process":
            st.session_state.current_page = "main"
        elif page == "📚 History":
            st.session_state.current_page = "history"
        elif page == "⚙️ Settings":
            st.session_state.current_page = "settings"
        
        st.divider()
        
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    if st.session_state.current_page == "history":
        show_history_page()
    elif st.session_state.current_page == "settings":
        show_settings_page()
    else:
        show_main_processing_page()

def main():
    st.set_page_config(page_title="YT Transcript Processor", page_icon="⚡", layout="wide")
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
