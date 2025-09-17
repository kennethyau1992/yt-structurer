import os
import re
import json
import time
import hashlib
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import requests
import streamlit as st

# ==================== PERSISTENCE LAYER ====================

class YouTubeInfoExtractor:
    """Helper class to extract YouTube video information"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, url: str) -> Dict[str, str]:
        """Get video title, description, and other metadata"""
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}
        
        # Try YouTube Data API first
        if self.api_key:
            api_info = self._get_info_from_api(video_id)
            if api_info:
                return api_info
        
        # Fallback to yt-dlp
        return self._get_info_from_ytdlp(url)
    
    def _get_info_from_api(self, video_id: str) -> Optional[Dict[str, str]]:
        """Get video info using YouTube Data API"""
        try:
            params = {
                'part': 'snippet,contentDetails',
                'id': video_id,
                'key': self.api_key
            }
            
            response = requests.get(
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
                    
                    return {
                        "title": snippet.get('title', 'Unknown Video'),
                        "description": snippet.get('description', '')[:500] + "..." if len(snippet.get('description', '')) > 500 else snippet.get('description', ''),
                        "duration": content_details.get('duration', ''),
                        "channel": snippet.get('channelTitle', ''),
                        "published_at": snippet.get('publishedAt', ''),
                        "thumbnail": snippet.get('thumbnails', {}).get('medium', {}).get('url', '')
                    }
        except Exception as e:
            st.warning(f"YouTube API failed: {e}")
        
        return None
    
    def _get_info_from_ytdlp(self, url: str) -> Dict[str, str]:
        """Get video info using yt-dlp as fallback"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get('title', 'Unknown Video'),
                    "description": (info.get('description', '') or '')[:500] + "..." if len(info.get('description', '') or '') > 500 else info.get('description', ''),
                    "duration": str(info.get('duration', '')),
                    "channel": info.get('uploader', ''),
                    "published_at": info.get('upload_date', ''),
                    "thumbnail": info.get('thumbnail', '')
                }
        except Exception as e:
            st.warning(f"yt-dlp info extraction failed: {e}")
            
        return {"title": "Unknown Video", "description": "", "duration": "", "channel": ""}

class UserDataManager:
    """Manages persistent storage of user settings and history"""
    
    def __init__(self, base_dir: str = "user_data"):
        """
        Initialize user data manager
        
        Args:
            base_dir: Base directory for storing user data
        """
        self.base_dir = base_dir
        self.ensure_data_directory()
        self.youtube_extractor = None
    
    def set_youtube_extractor(self, youtube_api_key: str = None):
        """Set YouTube info extractor for enhanced video metadata"""
        self.youtube_extractor = YouTubeInfoExtractor(youtube_api_key)
    
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def get_user_data_path(self, username: str) -> str:
        """Get the path to user's data file"""
        return os.path.join(self.base_dir, f"{username}_data.json")
    
    def load_user_data(self, username: str) -> Dict:
        """Load user data from file"""
        data_path = self.get_user_data_path(username)
        
        default_data = {
            "settings": {
                "language": "English",
                "use_asr_fallback": True,
                "deepseek_model": "deepseek-reasoner",
                "temperature": 0.1,
                "browser_for_cookies": "none"
            },
            "history": [],
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Ensure all required keys exist
                for key in default_data:
                    if key not in data:
                        data[key] = default_data[key]
                # Ensure all settings exist
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
        """Save user data to file"""
        data_path = self.get_user_data_path(username)
        data["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Could not save user data: {e}")
    
    def save_user_settings(self, username: str, settings: Dict):
        """Save user settings"""
        data = self.load_user_data(username)
        data["settings"].update(settings)
        self.save_user_data(username, data)
    
    def get_user_settings(self, username: str) -> Dict:
        """Get user settings"""
        data = self.load_user_data(username)
        return data["settings"]
    
    def add_to_history(self, username: str, entry: Dict):
        """Add entry to user's processing history with enhanced video info"""
        data = self.load_user_data(username)
        
        # Enhance entry with YouTube video information
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
            # Fallback to provided title
            entry["video_title"] = entry.get("title", "Unknown Video")
        
        # Add timestamp and unique ID
        entry["timestamp"] = datetime.now().isoformat()
        entry["id"] = hashlib.md5(f"{entry['url']}_{entry['timestamp']}".encode()).hexdigest()[:8]
        
        # Add to beginning of history
        data["history"].insert(0, entry)
        
        # Keep only last 100 entries
        data["history"] = data["history"][:100]
        
        self.save_user_data(username, data)
    
    def get_user_history(self, username: str) -> List[Dict]:
        """Get user's processing history"""
        data = self.load_user_data(username)
        return data["history"]
    
    def get_history_entry(self, username: str, entry_id: str) -> Optional[Dict]:
        """Get a specific history entry by ID"""
        history = self.get_user_history(username)
        for entry in history:
            if entry.get("id") == entry_id:
                return entry
        return None
    
    def delete_history_entry(self, username: str, entry_id: str):
        """Delete a specific history entry"""
        data = self.load_user_data(username)
        data["history"] = [entry for entry in data["history"] if entry.get("id") != entry_id]
        self.save_user_data(username, data)
    
    def clear_user_history(self, username: str):
        """Clear all user history"""
        data = self.load_user_data(username)
        data["history"] = []
        self.save_user_data(username, data)
    
    def export_user_data(self, username: str) -> str:
        """Export user data as JSON string"""
        data = self.load_user_data(username)
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def get_history_statistics(self, username: str) -> Dict:
        """Get statistics about user's history"""
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

# ==================== AUTHENTICATION LAYER ====================

class AuthManager:
    """Simple authentication manager for storing user credentials"""
    
    def __init__(self):
        # In production, use environment variables or a secure database
        self.users = {
            # Default admin user - change these credentials!
            "admin": {
                "password_hash": self._hash_password("admin123"),  # Change this!
                "api_keys": {
                    "supadata": os.getenv("ADMIN_SUPADATA_KEY", ""),
                    "assemblyai": os.getenv("ADMIN_ASSEMBLYAI_KEY", ""),
                    "deepseek": os.getenv("ADMIN_DEEPSEEK_KEY", ""),
                    "youtube": os.getenv("ADMIN_YOUTUBE_KEY", "")
                }
            }
        }
        
        # Initialize user data manager
        self.user_data_manager = UserDataManager()
        
        # Load additional users from environment variables if available
        self._load_users_from_env()
    
    def _hash_password(self, password: str) -> str:
        """Simple password hashing - in production use proper hashing like bcrypt"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users_from_env(self):
        """Load users from environment variables with debug output"""
        env_users = {}
        
        for key, value in os.environ.items():
            if key.endswith('_PASSWORD'):
                username = key[:-9].lower()  # Remove '_PASSWORD' and lowercase
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["password_hash"] = self._hash_password(value)
                
            elif key.endswith('_SUPADATA_KEY'):
                username = key[:-13].lower()  # Remove '_SUPADATA_KEY'
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["supadata"] = value
                
            elif key.endswith('_ASSEMBLYAI_KEY'):
                username = key[:-14].lower()  # Remove '_ASSEMBLYAI_KEY'
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["assemblyai"] = value
                
            elif key.endswith('_DEEPSEEK_KEY'):
                username = key[:-12].lower()  # Remove '_DEEPSEEK_KEY'
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["deepseek"] = value
                
            elif key.endswith('_YOUTUBE_KEY'):
                username = key[:-12].lower()  # Remove '_YOUTUBE_KEY'
                if username not in env_users:
                    env_users[username] = {"api_keys": {}}
                env_users[username]["api_keys"]["youtube"] = value
        
        # Merge with existing users
        for username, user_data in env_users.items():
            if username in self.users:
                # User already exists, update API keys
                self.users[username]["api_keys"].update(user_data["api_keys"])
            else:
                # New user
                self.users[username] = user_data
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        if username in self.users:
            return self.users[username]["password_hash"] == self._hash_password(password)
        return False
    
    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        """Get API keys for authenticated user"""
        if username in self.users:
            return self.users[username]["api_keys"]
        return {}
    
    def add_user(self, username: str, password: str, api_keys: Dict[str, str] = None):
        """Add a new user (for admin functionality)"""
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "api_keys": api_keys or {"supadata": "", "assemblyai": "", "deepseek": "", "youtube": ""}
        }
    
    def get_user_data_manager(self) -> UserDataManager:
        """Get the user data manager instance"""
        return self.user_data_manager
    
    def initialize_user_data_manager_with_youtube(self, youtube_api_key: str = None):
        """Initialize user data manager with YouTube API access"""
        self.user_data_manager.set_youtube_extractor(youtube_api_key)

# ==================== YOUTUBE COOKIE MANAGER ====================

class YouTubeCookieManager:
    """Manages YouTube cookies for yt-dlp to bypass bot detection"""
    
    @staticmethod
    def get_ydl_opts(use_cookies: bool = True, browser: str = 'chrome') -> dict:
        """
        Get yt-dlp options with cookie configuration
        
        Args:
            use_cookies: Whether to use browser cookies
            browser: Which browser to extract cookies from
        
        Returns:
            Dictionary of yt-dlp options
        """
        base_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        if use_cookies:
            # Check for Render secret file first
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                st.info(f"Using Render secret cookies file: {render_cookies}")
                base_opts['cookiefile'] = render_cookies
            else:
                # Check for environment variable
                cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
                if cookies_file and os.path.exists(cookies_file):
                    st.info(f"Using cookies file from env: {cookies_file}")
                    base_opts['cookiefile'] = cookies_file
                else:
                    # Try browser cookies (may not work on Linux/containers)
                    import platform
                    if platform.system() != 'Linux':
                        base_opts['cookiesfrombrowser'] = (browser,)
                    else:
                        # On Linux, especially in containers, browser cookie extraction often fails
                        st.warning("No cookies file found. Trying without authentication...")
        
        # Add user agent to avoid some bot detection
        base_opts['headers'] = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        return base_opts

# ==================== CHUNKING LAYER ====================

class AudioChunker:
    """Handles chunking of long audio files for transcription"""
    
    def __init__(self, chunk_duration_minutes: int = 12):
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.max_parallel_chunks = 3  # Limit parallel processing to avoid rate limits
        self.cookie_manager = YouTubeCookieManager()
    
    def get_video_duration(self, youtube_url: str, browser: str = 'chrome') -> Optional[int]:
        """Get video duration in seconds using yt-dlp"""
        try:
            import yt_dlp
            
            ydl_opts = self.cookie_manager.get_ydl_opts(use_cookies=True, browser=browser)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                duration = info.get('duration')
                return int(duration) if duration else None
                
        except Exception as e:
            st.warning(f"Could not get video duration: {e}")
            return None
    
    def extract_chunked_audio_urls(self, youtube_url: str, browser: str = 'chrome') -> List[Dict[str, Any]]:
        """Extract audio URLs for video chunks"""
        try:
            import yt_dlp
            
            # First, get video duration
            duration = self.get_video_duration(youtube_url, browser)
            if not duration:
                st.error("Could not determine video duration for chunking")
                return []
            
            if duration <= self.chunk_duration_seconds:
                st.info(f"Video is {duration//60}m {duration%60}s - no chunking needed")
                return []
            
            st.info(f"Video is {duration//60}m {duration%60}s - will chunk into {self.chunk_duration_seconds//60}-minute segments")
            
            # Calculate number of chunks needed
            num_chunks = (duration + self.chunk_duration_seconds - 1) // self.chunk_duration_seconds
            
            chunks = []
            for i in range(num_chunks):
                start_time = i * self.chunk_duration_seconds
                end_time = min((i + 1) * self.chunk_duration_seconds, duration)
                
                # Use yt-dlp to get audio URL
                ydl_opts = self.cookie_manager.get_ydl_opts(use_cookies=True, browser=browser)
                ydl_opts.update({
                    'format': 'bestaudio/best',
                    'noplaylist': True,
                })
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    formats = info.get('formats', [])
                    
                    # Find best audio format
                    audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                    if not audio_formats:
                        audio_formats = [f for f in formats if f.get('acodec') != 'none']
                    
                    if audio_formats:
                        best_audio = max(audio_formats, key=lambda x: x.get('abr', 0) or 0)
                        chunks.append({
                            'chunk_id': i,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'audio_url': best_audio.get('url'),
                            'format_note': best_audio.get('format_note', 'unknown')
                        })
            
            return chunks
            
        except Exception as e:
            st.error(f"Error extracting chunked audio URLs: {e}")
            return []
    
    def transcribe_chunks_parallel(self, chunks: List[Dict], assemblyai_provider, language: str) -> Optional[str]:
        """Transcribe multiple chunks in parallel and combine results"""
        if not chunks:
            return None
        
        st.info(f"Processing {len(chunks)} chunks in parallel...")
        
        def transcribe_single_chunk(chunk_info):
            """Transcribe a single chunk"""
            chunk_id = chunk_info['chunk_id']
            start_time = chunk_info['start_time']
            end_time = chunk_info['end_time']
            audio_url = chunk_info['audio_url']
            
            try:
                st.info(f"Chunk {chunk_id + 1}: {start_time//60}m{start_time%60}s - {end_time//60}m{end_time%60}s")
                
                # For now, we'll transcribe the full audio URL and note the time range
                transcript = assemblyai_provider._transcribe_audio_url(audio_url, language)
                
                return {
                    'chunk_id': chunk_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': transcript,
                    'success': transcript is not None
                }
                
            except Exception as e:
                st.error(f"Chunk {chunk_id + 1} failed: {e}")
                return {
                    'chunk_id': chunk_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Process chunks in parallel with limited concurrency
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(transcribe_single_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        st.success(f"Chunk {result['chunk_id'] + 1} completed")
                    else:
                        st.error(f"Chunk {result['chunk_id'] + 1} failed")
                        
                except Exception as e:
                    st.error(f"Chunk processing error: {e}")
        
        # Sort results by chunk_id and combine transcripts
        results.sort(key=lambda x: x['chunk_id'])
        
        successful_transcripts = []
        failed_chunks = []
        
        for result in results:
            if result['success'] and result['transcript']:
                # Add timestamp marker
                start_min, start_sec = divmod(result['start_time'], 60)
                end_min, end_sec = divmod(result['end_time'], 60)
                timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
                
                successful_transcripts.append(f"{timestamp}\n{result['transcript']}")
            else:
                failed_chunks.append(result['chunk_id'] + 1)
        
        if not successful_transcripts:
            st.error("All chunks failed to transcribe")
            return None
        
        if failed_chunks:
            st.warning(f"Chunks {failed_chunks} failed, but continuing with successful chunks")
        
        # Combine all successful transcripts
        combined_transcript = "\n\n".join(successful_transcripts)
        
        st.success(f"Successfully combined {len(successful_transcripts)} chunks into final transcript")
        return combined_transcript

# ==================== LLM TEXT CHUNKER ====================

class LLMTextChunker:
    """Handles chunking of long transcripts for LLM processing with intelligent text splitting"""
    
    def __init__(self, max_chunk_length: int = 8000, overlap_length: int = 200):
        """
        Initialize text chunker for LLM processing
        
        Args:
            max_chunk_length: Maximum characters per chunk (conservative for token limits)
            overlap_length: Number of characters to overlap between chunks for context preservation
        """
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
        self.min_chunk_length = 1000  # Minimum viable chunk size
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for most languages)"""
        return len(text) // 4
    
    def should_chunk_text(self, text: str) -> bool:
        """Determine if text needs chunking based on length"""
        estimated_tokens = self.estimate_tokens(text)
        # Conservative threshold to avoid context window issues
        return estimated_tokens > 6000  # ~24k characters
    
    def find_split_points(self, text: str) -> List[int]:
        """Find intelligent split points in text (paragraphs, sections, sentences)"""
        split_points = []
        
        # Priority 1: Look for section breaks (double newlines)
        for match in re.finditer(r'\n\s*\n', text):
            split_points.append(match.end())
        
        # Priority 2: Look for sentence endings
        sentence_endings = r'[.!?]\s+'
        for match in re.finditer(sentence_endings, text):
            split_points.append(match.end())
        
        # Priority 3: Look for any newlines
        for match in re.finditer(r'\n', text):
            split_points.append(match.end())
        
        # Remove duplicates and sort
        split_points = sorted(list(set(split_points)))
        
        return split_points
    
    def create_intelligent_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks with intelligent splitting"""
        if not self.should_chunk_text(text):
            return [{
                'chunk_id': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'text': text,
                'tokens_estimate': self.estimate_tokens(text),
                'is_single_chunk': True
            }]
        
        st.info(f"Text is {len(text):,} characters (~{self.estimate_tokens(text):,} tokens) - chunking for better LLM processing...")
        
        chunks = []
        split_points = self.find_split_points(text)
        
        current_start = 0
        chunk_id = 0
        
        while current_start < len(text):
            # Find the end position for this chunk
            target_end = current_start + self.max_chunk_length
            
            if target_end >= len(text):
                # Last chunk
                chunk_text = text[current_start:]
                chunks.append({
                    'chunk_id': chunk_id,
                    'start_pos': current_start,
                    'end_pos': len(text),
                    'text': chunk_text,
                    'tokens_estimate': self.estimate_tokens(chunk_text),
                    'is_final_chunk': True
                })
                break
            
            # Find the best split point before target_end
            best_split = target_end
            for split_point in reversed(split_points):
                if current_start + self.min_chunk_length <= split_point <= target_end:
                    best_split = split_point
                    break
            
            # Create chunk
            chunk_text = text[current_start:best_split]
            chunks.append({
                'chunk_id': chunk_id,
                'start_pos': current_start,
                'end_pos': best_split,
                'text': chunk_text,
                'tokens_estimate': self.estimate_tokens(chunk_text),
                'is_final_chunk': False
            })
            
            # Calculate next start with overlap
            next_start = max(current_start + self.min_chunk_length, best_split - self.overlap_length)
            current_start = next_start
            chunk_id += 1
        
        st.info(f"Created {len(chunks)} intelligent chunks with overlap for context preservation")
        return chunks
    
    def create_chunk_specific_prompt(self, base_system_prompt: str, chunk_info: Dict, total_chunks: int, language: str) -> str:
        """Create a context-aware system prompt for each chunk"""
        chunk_id = chunk_info['chunk_id']
        is_single = chunk_info.get('is_single_chunk', False)
        is_final = chunk_info.get('is_final_chunk', False)
        
        if is_single:
            return base_system_prompt
        
        # Multi-chunk processing instructions
        if "English" in base_system_prompt or language == "English":
            chunk_context = f"""
CHUNK PROCESSING CONTEXT:
- This is chunk {chunk_id + 1} of {total_chunks} chunks
- Process ONLY the content in this chunk
- Maintain consistency with overall document structure
- {"This is the FINAL chunk - ensure proper conclusion" if is_final else "More chunks will follow - maintain section flow"}

IMPORTANT CHUNK RULES:
- Create sections/headings based ONLY on content in THIS chunk
- If chunk starts mid-sentence/mid-topic, begin appropriately
- If chunk ends mid-topic, end appropriately (will continue in next chunk)
- Use section numbers that make sense for THIS chunk's content
- Don't reference "previous" or "next" chunks in the output

"""
        else:  # Chinese
            chunk_context = f"""
分块处理上下文：
- 这是第 {chunk_id + 1} 块，共 {total_chunks} 块
- 仅处理此块中的内容
- 保持与整体文档结构的一致性
- {"这是最后一块 - 确保适当的结论" if is_final else "后续还有更多块 - 保持章节流畅"}

重要分块规则：
- 仅基于此块中的内容创建章节/标题
- 如果块从句子/主题中间开始，请适当开始
- 如果块在主题中间结束，请适当结束（将在下一块中继续）
- 使用适合此块内容的章节编号
- 不要在输出中引用"前一个"或"下一个"块

"""
        
        return chunk_context + base_system_prompt

# ==================== PROVIDER LAYER ====================

class TranscriptProvider:
    """Base class for transcript providers"""
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class SupadataTranscriptProvider(TranscriptProvider):
    def __init__(self, api_key: str):
        try:
            from supadata import Supadata, SupadataError
            self.client = Supadata(api_key=api_key)
            self.available = True
        except ImportError:
            self.available = False
            self.client = None
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.available or not self.client:
            st.warning("Supadata client not available")
            return None
        
        # FIX: Map display language names to Supadata language codes
        language_map = {
            "English": "en",
            "中文": "zh",
            "zh": "zh",
            "en": "en"
        }
        
        supadata_lang = language_map.get(language, "en")
        
        try:
            resp = self.client.transcript(url=url, lang=supadata_lang, text=True, mode="auto")
            return self._normalize_transcript(resp)
        except Exception as e:
            st.error(f"Supadata error: {e}")
            return None
    
    def _normalize_transcript(self, resp) -> str:
        """
        Convert various Supadata transcript response shapes to a single plain-text string.
        """
        if resp is None:
            return ""
        
        # Check if it's a BatchJob object (async job that's not ready)
        resp_str = str(resp)
        if "BatchJob" in resp_str or "job_id" in resp_str:
            st.warning("Supadata returned a BatchJob - transcript may be processing or unavailable")
            return ""

        # Plain string already
        if isinstance(resp, str):
            # Additional check for error strings
            if "BatchJob" in resp or "job_id" in resp:
                return ""
            return resp

        # Dict shapes: {"text": "..."} or {"chunks": [...]}
        if isinstance(resp, dict):
            # Check for error or job status
            if resp.get("status") == "processing" or resp.get("job_id"):
                st.warning("Transcript is still processing")
                return ""
            
            t = resp.get("text")
            if isinstance(t, str):
                return t

            chunks = resp.get("chunks")
            if isinstance(chunks, list):
                parts = []
                for ch in chunks:
                    if isinstance(ch, str):
                        parts.append(ch)
                    elif isinstance(ch, dict):
                        v = ch.get("text") or ch.get("content")
                        if isinstance(v, str):
                            parts.append(v)
                    else:
                        v = getattr(ch, "text", None)
                        if isinstance(v, str):
                            parts.append(v)
                return "\n".join(parts)

        # List shapes
        if isinstance(resp, list):
            parts = []
            for ch in resp:
                if isinstance(ch, str):
                    parts.append(ch)
                elif isinstance(ch, dict):
                    v = ch.get("text") or ch.get("content")
                    if isinstance(v, str):
                        parts.append(v)
                else:
                    v = getattr(ch, "text", None)
                    if isinstance(v, str):
                        parts.append(v)
            return "\n".join(parts)

        # Object with .text
        v = getattr(resp, "text", None)
        if isinstance(v, str):
            return v

        # Fallback stringify - but check for BatchJob pattern
        try:
            result_str = str(resp)
            if "BatchJob" not in result_str and "job_id" not in result_str:
                return result_str
            return ""
        except Exception:
            return ""

class YouTubeAudioExtractor:
    """Helper class to extract audio URLs from YouTube videos for ASR"""
    
    def __init__(self):
        self.cookie_manager = YouTubeCookieManager()
    
    def extract_audio_url(self, youtube_url: str, browser: str = 'chrome') -> Optional[str]:
        """
        Extract direct audio URL from YouTube video using yt-dlp.
        Returns the best available audio stream URL for AssemblyAI.
        
        Args:
            youtube_url: YouTube video URL
            browser: Browser to extract cookies from ('chrome', 'firefox', 'edge', 'safari')
        """
        try:
            import yt_dlp
            import platform
            
            st.info(f"Extracting audio stream URL using yt-dlp...")
            
            # Base options
            ydl_opts = {
                'format': 'bestaudio/best',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            }
            
            # Check for Render secret cookies file first
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            cookies_file = None
            
            if os.path.exists(render_cookies):
                st.success(f"Found Render secret cookies file!")
                cookies_file = render_cookies
            else:
                # Check environment variable
                env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
                if env_cookies and os.path.exists(env_cookies):
                    st.info(f"Using cookies file from environment variable")
                    cookies_file = env_cookies
            
            # Add cookies file if found
            if cookies_file:
                ydl_opts['cookiefile'] = cookies_file
                st.info(f"Using cookies for authentication: {cookies_file}")
            else:
                st.warning("No cookies file found - attempting without authentication")
            
            # Try to extract audio
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    
                    if info:
                        formats = info.get('formats', [])
                        
                        # Try to find audio-only streams
                        audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                        
                        if audio_formats:
                            audio_formats.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                            best_audio = audio_formats[0]
                            st.success(f"Found audio-only stream: {best_audio.get('format_note', 'unknown quality')}")
                            return best_audio.get('url')
                        
                        # Fallback: find formats with audio
                        formats_with_audio = [f for f in formats if f.get('acodec') != 'none']
                        
                        if formats_with_audio:
                            formats_with_audio.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                            best_format = formats_with_audio[0]
                            st.success(f"Found audio stream: {best_format.get('format_note', 'unknown quality')}")
                            return best_format.get('url')
                    
                    st.error("No audio formats found in video")
                    return None
            
            except Exception as e:
                error_msg = str(e)
                if "Sign in to confirm" in error_msg:
                    st.error("YouTube requires authentication - cookies may be expired or invalid")
                    st.info("Please update your youtube_cookies.txt file in Render secrets:\n"
                           "1. Log into YouTube in your browser\n"
                           "2. Use 'Get cookies.txt' extension to export fresh cookies\n"
                           "3. Update the youtube_cookies.txt secret file in Render")
                elif "Video unavailable" in error_msg:
                    st.error("Video is unavailable or private")
                else:
                    st.error(f"Error extracting audio: {error_msg}")
                return None
                
        except ImportError:
            st.error("yt-dlp not installed. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            st.error(f"Audio extraction error: {e}")
            return None

class ImprovedAssemblyAITranscriptProvider(TranscriptProvider):
    """Improved ASR provider that extracts audio URLs using yt-dlp with chunking support"""
    def __init__(self, api_key: str, browser: str = 'chrome'):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
        self.chunker = AudioChunker()
        self.max_duration_minutes = 60  # Chunk videos longer than 60 minutes
        self.browser = browser
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            st.warning("AssemblyAI API key not provided")
            return None
        
        # Step 1: Check video duration for chunking decision
        duration = self.chunker.get_video_duration(url, self.browser)
        
        if duration and duration > (self.max_duration_minutes * 60):
            st.info(f"Long video detected ({duration//60}m {duration%60}s) - using chunked transcription")
            return self._transcribe_with_chunking(url, language)
        else:
            st.info(f"Standard transcription for video ({duration//60 if duration else 'unknown'}m)")
            return self._transcribe_standard(url, language)
    
    def _transcribe_standard(self, url: str, language: str) -> Optional[str]:
        """Standard transcription without chunking"""
        # Step 1: Extract audio URL from YouTube video
        st.info("Step 1: Extracting audio URL from YouTube video...")
        audio_url = self.audio_extractor.extract_audio_url(url, self.browser)
        
        if not audio_url:
            st.error("Cannot extract audio URL from YouTube video. ASR fallback failed.")
            return None
        
        # Step 2: Proceed with AssemblyAI transcription using the extracted audio URL
        st.info("Step 2: Starting AssemblyAI transcription...")
        return self._transcribe_audio_url(audio_url, language)
    
    def _transcribe_with_chunking(self, url: str, language: str) -> Optional[str]:
        """Transcription with chunking for long videos"""
        st.info("Step 1: Preparing video chunks...")
        chunks = self.chunker.extract_chunked_audio_urls(url, self.browser)
        
        if not chunks:
            st.warning("Could not create chunks, falling back to standard transcription")
            return self._transcribe_standard(url, language)
        
        st.info("Step 2: Starting parallel chunk transcription...")
        return self.chunker.transcribe_chunks_parallel(chunks, self, language)
    
    def _transcribe_audio_url(self, audio_url: str, language: str) -> Optional[str]:
        """Transcribe using a direct audio URL"""
        try:
            headers = {
                "authorization": self.api_key,
                "content-type": "application/json"
            }
            
            # Map language codes for AssemblyAI
            language_map = {
                "English": "en",
                "中文": "zh",
                "en": "en",
                "zh": "zh"
            }
            
            assemblyai_lang = language_map.get(language, "en")
            
            # Prepare transcription request
            data = {
                "audio_url": audio_url,
                "language_code": assemblyai_lang,
                "speech_model": "best"  # Use best available model
            }
            
            st.info("Submitting transcription request to AssemblyAI...")
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', 'Unknown error')
                st.error(f"AssemblyAI submission error ({response.status_code}): {error_msg}")
                return None
                
            transcript_id = response.json()['id']
            st.success(f"Transcription submitted with ID: {transcript_id}")
            
            # Step 3: Poll for completion with progress indication
            return self._poll_for_completion(transcript_id, headers)
            
        except requests.exceptions.Timeout:
            st.error("Request timed out while submitting to AssemblyAI")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Network error communicating with AssemblyAI: {e}")
            return None
        except Exception as e:
            st.error(f"AssemblyAI transcription error: {e}")
            return None
    
    def _poll_for_completion(self, transcript_id: str, headers: dict) -> Optional[str]:
        """Poll AssemblyAI for transcription completion with progress updates"""
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        max_attempts = 120  # 6 minutes max (3 second intervals)
        attempt = 0
        
        st.info("Polling for transcription completion...")
        
        # Create a progress bar and status display
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while attempt < max_attempts:
            try:
                polling_response = requests.get(polling_endpoint, headers=headers, timeout=30)
                
                if polling_response.status_code != 200:
                    st.error(f"Polling error ({polling_response.status_code})")
                    return None
                
                polling_data = polling_response.json()
                status = polling_data.get('status', 'unknown')
                
                # Update status display
                status_text.text(f"Status: {status} (attempt {attempt + 1}/{max_attempts})")
                progress_bar.progress((attempt + 1) / max_attempts)
                
                if status == 'completed':
                    status_text.text("Transcription completed successfully!")
                    progress_bar.progress(1.0)
                    
                    transcript_text = polling_data.get('text', '')
                    if transcript_text:
                        st.success(f"Received transcript ({len(transcript_text)} characters)")
                        return transcript_text
                    else:
                        st.error("Transcription completed but no text returned")
                        return None
                        
                elif status == 'error':
                    error_msg = polling_data.get('error', 'Unknown transcription error')
                    st.error(f"AssemblyAI transcription error: {error_msg}")
                    return None
                    
                elif status in ['queued', 'processing']:
                    # Continue polling
                    time.sleep(3)
                    attempt += 1
                    
                else:
                    st.warning(f"Unknown status: {status}")
                    time.sleep(3)
                    attempt += 1
            
            except requests.exceptions.Timeout:
                st.error("Polling request timed out")
                return None
            except Exception as e:
                st.error(f"Polling error: {e}")
                return None
        
        st.error("Transcription polling timed out after 6 minutes")
        return None

class LLMProvider:
    """Base class for LLM providers"""
    def structure_transcript(self, transcript: str, system_prompt: str, language: str = "English") -> Optional[Dict[str, str]]:
        raise NotImplementedError

class LanguageDetector:
    """Detects language of transcript content"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect if text is primarily Chinese or English
        Returns 'Chinese' or 'English'
        """
        if not text or len(text.strip()) < 10:
            return "English"  # Default fallback
        
        # Count Chinese characters (CJK ranges)
        chinese_chars = 0
        english_chars = 0
        total_chars = 0
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf' or '\uf900' <= char <= '\ufaff':
                # Chinese characters (including Traditional and Simplified)
                chinese_chars += 1
                total_chars += 1
            elif char.isalpha() and ord(char) < 128:
                # English/Latin characters
                english_chars += 1
                total_chars += 1
        
        if total_chars == 0:
            return "English"  # Default if no detectable characters
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # If more than 30% Chinese characters, consider it Chinese
        if chinese_ratio > 0.3:
            return "Chinese"
        # If more than 50% English characters, consider it English
        elif english_ratio > 0.5:
            return "English"
        else:
            # Ambiguous case, use character count
            return "Chinese" if chinese_chars > english_chars else "English"
    
    @staticmethod
    def get_language_code(language: str) -> str:
        """Convert language name to code - using Traditional Chinese"""
        if language == "Chinese":
            return "繁體中文"  # Traditional Chinese
        else:
            return "English"

class PerformanceOptimizedTextChunker:
    """High-performance text chunker with parallel processing support"""
    
    def __init__(self, performance_mode: str = "Balanced"):
        self.performance_mode = performance_mode
        
        # Adjust parameters based on performance mode
        if performance_mode == "Speed Optimized":
            self.max_chunk_length = 6000  # Smaller chunks for faster processing
            self.overlap_length = 100     # Less overlap
            self.max_parallel_workers = 4 # More parallel workers
        elif performance_mode == "Quality Optimized":
            self.max_chunk_length = 12000 # Larger chunks for better context
            self.overlap_length = 300     # More overlap
            self.max_parallel_workers = 2 # Fewer workers to avoid rate limits
        else:  # Balanced
            self.max_chunk_length = 8000
            self.overlap_length = 200
            self.max_parallel_workers = 3
            
        self.min_chunk_length = 800
    
    def should_chunk_text(self, text: str) -> bool:
        """Determine if text needs chunking"""
        estimated_tokens = len(text) // 4
        
        # Adjust threshold based on performance mode
        if self.performance_mode == "Speed Optimized":
            return estimated_tokens > 4000  # Chunk earlier for speed
        elif self.performance_mode == "Quality Optimized":
            return estimated_tokens > 8000  # Chunk later for quality
        else:
            return estimated_tokens > 6000  # Balanced
    
    def create_performance_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks optimized for the selected performance mode"""
        if not self.should_chunk_text(text):
            return [{
                'chunk_id': 0,
                'text': text,
                'tokens_estimate': len(text) // 4,
                'is_single_chunk': True
            }]
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        # Simple but fast chunking strategy
        while current_pos < len(text):
            end_pos = min(current_pos + self.max_chunk_length, len(text))
            
            # Find sentence boundary near end
            if end_pos < len(text):
                sentence_end = text.rfind('.', current_pos + self.min_chunk_length, end_pos)
                if sentence_end > current_pos + self.min_chunk_length:
                    end_pos = sentence_end + 1
            
            chunk_text = text[current_pos:end_pos]
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'tokens_estimate': len(chunk_text) // 4,
                'is_final_chunk': end_pos >= len(text)
            })
            
            # Calculate next position with overlap
            current_pos = max(current_pos + self.min_chunk_length, end_pos - self.overlap_length)
            chunk_id += 1
        
        return chunks

class FastDeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, performance_mode: str = "Balanced"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.performance_mode = performance_mode
        
        # Initialize optimized text chunker
        self.text_chunker = PerformanceOptimizedTextChunker(performance_mode)
        self.language_detector = LanguageDetector()
        
        # Adjust timeouts based on model and performance mode
        if model == "deepseek-chat":
            self.api_timeout = 60  # Faster model, shorter timeout
        else:
            self.api_timeout = 180  # Reasoning model needs more time
    
    def structure_transcript(self, transcript: str, system_prompt: str, ui_language: str = "English", is_custom_prompt: bool = False) -> Optional[Dict[str, str]]:
        """Performance-optimized transcript structuring"""
        try:
            # Detect language
            detected_language = self.language_detector.detect_language(transcript)
            actual_language = self.language_detector.get_language_code(detected_language)
            
            # Show performance info
            if self.performance_mode == "Speed Optimized":
                st.info(f"🚀 Speed mode: Detected {actual_language} - Using parallel processing for faster results")
            elif self.performance_mode == "Quality Optimized":
                st.info(f"🎯 Quality mode: Detected {actual_language} - Using sequential processing for best results")
            else:
                st.info(f"⚖️ Balanced mode: Detected {actual_language} - Optimizing for speed and quality")
            
            # Choose processing strategy based on performance mode
            if self.performance_mode == "Speed Optimized":
                return self._process_parallel_optimized(transcript, system_prompt, actual_language, is_custom_prompt)
            else:
                return self._process_sequential_optimized(transcript, system_prompt, actual_language, is_custom_prompt)
                
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def _process_parallel_optimized(self, transcript: str, system_prompt: str, language: str, is_custom_prompt: bool) -> Optional[Dict[str, str]]:
        """Parallel processing for speed optimization"""
        
        # Process summary and detailed transcript in parallel
        import concurrent.futures
        
        def process_summary():
            st.info("⚡ Step 1: Generating executive summary (parallel)...")
            summary_prompt = self._create_summary_prompt(language)
            return self._process_for_summary_fast(transcript, summary_prompt)
        
        def process_detailed():
            st.info("⚡ Step 2: Generating detailed transcript (parallel)...")
            adapted_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
            return self._process_for_detailed_structure_fast(transcript, adapted_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            summary_future = executor.submit(process_summary)
            detailed_future = executor.submit(process_detailed)
            
            # Get results
            executive_summary = summary_future.result()
            detailed_transcript = detailed_future.result()
        
        if not executive_summary or not detailed_transcript:
            st.error("Parallel processing failed")
            return None
            
        return {
            'executive_summary': executive_summary,
            'detailed_transcript': detailed_transcript,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt,
            'processing_mode': 'parallel'
        }
    
    def _process_sequential_optimized(self, transcript: str, system_prompt: str, language: str, is_custom_prompt: bool) -> Optional[Dict[str, str]]:
        """Sequential processing with optimizations"""
        
        # Step 1: Generate Executive Summary
        st.info(f"Step 1: Generating executive summary in {language}...")
        summary_prompt = self._create_summary_prompt(language)
        executive_summary = self._process_for_summary_fast(transcript, summary_prompt)
        
        if not executive_summary:
            st.error("Failed to generate executive summary")
            return None
        
        # Step 2: Generate Detailed Structured Transcript
        st.info(f"Step 2: Generating detailed structured transcript...")
        adapted_system_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        detailed_transcript = self._process_for_detailed_structure_fast(transcript, adapted_system_prompt)
        
        if not detailed_transcript:
            st.error("Failed to generate detailed transcript")
            return None
        
        return {
            'executive_summary': executive_summary,
            'detailed_transcript': detailed_transcript,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt,
            'processing_mode': 'sequential'
        }
    
    def _process_for_summary_fast(self, transcript: str, summary_prompt: str) -> Optional[str]:
        """Fast summary processing"""
        
        # For speed mode, use smaller context windows
        if self.performance_mode == "Speed Optimized" and len(transcript) > 15000:
            # Truncate very long transcripts for summary
            st.info("⚡ Using truncated content for faster summary generation")
            transcript_sample = transcript[:15000] + "\n\n[Content truncated for speed]"
            return self._make_api_request_fast(transcript_sample, summary_prompt)
        else:
            return self._make_api_request_fast(transcript, summary_prompt)
    
    def _process_for_detailed_structure_fast(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Fast detailed structure processing"""
        
        if not self.text_chunker.should_chunk_text(transcript):
            return self._make_api_request_fast(transcript, system_prompt)
        
        # Use performance-optimized chunking
        chunks = self.text_chunker.create_performance_chunks(transcript)
        
        if len(chunks) == 1:
            return self._make_api_request_fast(transcript, system_prompt)
        
        if self.performance_mode == "Speed Optimized":
            return self._process_chunks_parallel(chunks, system_prompt)
        else:
            return self._process_chunks_sequential(chunks, system_prompt)
    
    def _process_chunks_parallel(self, chunks: List[Dict], system_prompt: str) -> Optional[str]:
        """Process chunks in parallel for speed"""
        import concurrent.futures
        
        def process_chunk(chunk_info):
            return self._make_api_request_fast(chunk_info['text'], system_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.text_chunker.max_parallel_workers) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        # Filter successful results
        successful_results = [r for r in results if r is not None]
        
        if not successful_results:
            return None
        
        return "\n\n---\n\n".join(successful_results)
    
    def _process_chunks_sequential(self, chunks: List[Dict], system_prompt: str) -> Optional[str]:
        """Process chunks sequentially with progress updates"""
        results = []
        
        for i, chunk in enumerate(chunks):
            st.info(f"Processing chunk {i+1}/{len(chunks)}")
            result = self._make_api_request_fast(chunk['text'], system_prompt)
            if result:
                results.append(result)
        
        if not results:
            return None
            
        return "\n\n---\n\n".join(results)
    
    def _make_api_request_fast(self, text: str, system_prompt: str) -> Optional[str]:
        """Optimized API request with performance monitoring"""
        start_time = time.time()
        
        try:
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

            resp = requests.post(
                endpoint, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=self.api_timeout
            )
            
            if resp.status_code != 200:
                error_detail = ""
                try:
                    error_data = resp.json()
                    error_detail = error_data.get('error', {}).get('message', resp.text)
                except:
                    error_detail = resp.text
                
                raise RuntimeError(f"DeepSeek API error {resp.status_code}: {error_detail}")
            
            data = resp.json()
            
            if 'choices' not in data or len(data['choices']) == 0:
                raise RuntimeError("DeepSeek API returned no choices")
            
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise RuntimeError("DeepSeek API returned empty content")
            
            # Show performance info
            elapsed = time.time() - start_time
            st.success(f"✅ API call completed in {elapsed:.1f}s")
                
            return content.strip()
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"DeepSeek API request timed out after {self.api_timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def structure_transcript(self, transcript: str, system_prompt: str, ui_language: str = "English", is_custom_prompt: bool = False) -> Optional[Dict[str, str]]:
        """Enhanced transcript structuring with automatic language detection"""
        try:
            # Detect the actual language of the transcript content
            detected_language = self.language_detector.detect_language(transcript)
            actual_language = self.language_detector.get_language_code(detected_language)
            
            # Log the detection for user awareness
            if is_custom_prompt:
                st.info(f"🔍 Detected transcript language: **{actual_language}** (Using your custom system prompt)")
            else:
                st.info(f"🔍 Detected transcript language: **{actual_language}** (Processing will use detected language)")
            
            # Generate both executive summary and detailed structured transcript using detected language
            return self._process_transcript_with_summary(transcript, system_prompt, actual_language, is_custom_prompt)
                
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def _process_transcript_with_summary(self, transcript: str, system_prompt: str, language: str, is_custom_prompt: bool = False) -> Optional[Dict[str, str]]:
        """Process transcript to generate both executive summary and detailed structure"""
        
        # Step 1: Generate Executive Summary in detected language
        st.info(f"Step 1: Generating executive summary in {language}...")
        summary_prompt = self._create_summary_prompt(language)
        executive_summary = self._process_for_summary(transcript, summary_prompt, language)
        
        if not executive_summary:
            st.error("Failed to generate executive summary")
            return None
        
        # Step 2: Generate Detailed Structured Transcript in detected language
        if is_custom_prompt:
            st.info(f"Step 2: Generating detailed structured transcript using your custom prompt (output in {language})...")
        else:
            st.info(f"Step 2: Generating detailed structured transcript in {language}...")
        
        # Adapt the system prompt to the detected language, preserving user customizations
        adapted_system_prompt = self._adapt_system_prompt_to_language(system_prompt, language, is_custom_prompt)
        detailed_transcript = self._process_for_detailed_structure(transcript, adapted_system_prompt, language)
        
        if not detailed_transcript:
            st.error("Failed to generate detailed transcript")
            return None
        
        return {
            'executive_summary': executive_summary,
            'detailed_transcript': detailed_transcript,
            'detected_language': language,
            'used_custom_prompt': is_custom_prompt
        }
    
    def _adapt_system_prompt_to_language(self, system_prompt: str, detected_language: str, is_custom_prompt: bool = False) -> str:
        """Adapt the system prompt to match the detected language - but preserve user customizations"""
        
        # If user has customized the prompt, use it as-is and just add language instruction
        if is_custom_prompt:
            language_instruction = ""
            if detected_language == "繁體中文":
                language_instruction = "\n\n**重要提醒：請用繁體中文輸出結構化的轉錄文本。**"
            else:
                language_instruction = "\n\n**Important: Please output the structured transcript in English.**"
            
            return system_prompt + language_instruction
        
        # Otherwise, use predefined language-specific prompts
        if detected_language == "繁體中文":
            # Traditional Chinese prompt
            chinese_prompt = """你是一個專業的YouTube影片轉錄文本分析和結構化專家。你的任務是將原始轉錄文本轉換為組織良好、易於閱讀的文檔。

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
            
            return chinese_prompt
        
        else:  # English
            # If content is English, ensure English prompt
            english_prompt = """You are an expert at analyzing and structuring YouTube video transcripts. Your task is to convert raw transcript text into a well-organized, readable document.

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

Format the output as a clean, professional document that would be easy to read and reference. Please output the structured transcript in English."""
            
            return english_prompt
    
    def _create_summary_prompt(self, language: str) -> str:
        """Create a specialized prompt for generating executive summary in Traditional Chinese"""
        
        if language == "繁體中文":
            summary_prompt = """你是一個專業的YouTube影片轉錄文本執行摘要專家。

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
- 專注於實質內容而非填充內容
- 使用清晰、專業的語言

使用markdown格式，包含標題和要點，以獲得最大可讀性。請用繁體中文輸出執行摘要。"""
            
        else:  # English
            summary_prompt = """You are an expert at creating concise executive summaries from YouTube video transcripts.

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
- Maintain the speaker's key messages and tone
- Focus on substance over filler content
- Use clear, professional language

Format using markdown with headers and bullet points for maximum readability. Please output the executive summary in English."""
        
        return summary_prompt
    
    def _process_for_summary(self, transcript: str, summary_prompt: str, language: str) -> Optional[str]:
        """Process transcript to generate executive summary"""
        
        # For summary, we can handle longer text since we're condensing it
        if self.text_chunker.estimate_tokens(transcript) > 10000:  # ~40k characters
            st.info("Long transcript detected - using intelligent chunking for summary generation...")
            return self._process_summary_with_chunking(transcript, summary_prompt, language)
        else:
            return self._make_api_request(transcript, summary_prompt)
    
    def _process_summary_with_chunking(self, transcript: str, summary_prompt: str, language: str) -> Optional[str]:
        """Process long transcript with chunking for summary generation"""
        
        # Create larger chunks for summary since we're condensing
        large_chunker = LLMTextChunker(max_chunk_length=12000, overlap_length=300)
        chunks = large_chunker.create_intelligent_chunks(transcript)
        
        if len(chunks) == 1:
            return self._make_api_request(transcript, summary_prompt)
        
        st.info(f"Processing {len(chunks)} chunks for summary generation...")
        
        # Generate summary for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            st.info(f"Generating summary for chunk {i+1}/{len(chunks)}")
            
            chunk_summary_prompt = f"""
{summary_prompt}

CHUNK CONTEXT: This is chunk {i+1} of {len(chunks)} from a longer transcript. 
Focus on summarizing the key points from THIS specific chunk.
"""
            
            chunk_summary = self._make_api_request(chunk['text'], chunk_summary_prompt)
            if chunk_summary:
                chunk_summaries.append(chunk_summary)
        
        if not chunk_summaries:
            st.error("Failed to generate any chunk summaries")
            return None
        
        # Combine chunk summaries into final executive summary
        st.info("Combining chunk summaries into final executive summary...")
        
        combined_summaries = "\n\n".join(chunk_summaries)
        
        if language == "English":
            final_summary_prompt = """You are an expert at synthesizing multiple summary chunks into a single, coherent executive summary.

Below are summary chunks from different parts of a video transcript. Your task is to:

1. Combine these chunks into one unified executive summary
2. Remove redundancy and overlapping points
3. Organize information logically
4. Maintain the structure: Overview, Key Points, Important Details, Conclusions/Takeaways
5. Keep it concise but comprehensive

Create a polished, professional executive summary that flows naturally."""
            
        else:  # Chinese
            final_summary_prompt = """你是一个专业的多摘要块综合专家。

以下是视频转录文本不同部分的摘要块。你的任务是：

1. 将这些块合并成一个统一的执行摘要
2. 去除冗余和重叠的点
3. 逻辑性地组织信息
4. 保持结构：概述、关键点、重要细节、结论/要点
5. 保持简洁但全面

创建一个精美、专业的执行摘要，自然流畅。"""
        
        return self._make_api_request(combined_summaries, final_summary_prompt)
    
    def _process_for_detailed_structure(self, transcript: str, system_prompt: str, language: str) -> Optional[str]:
        """Process transcript for detailed structure"""
        
        # Determine if chunking is needed
        if not self.text_chunker.should_chunk_text(transcript):
            # Process as single chunk
            return self._make_api_request(transcript, system_prompt)
        else:
            # Process with intelligent chunking
            return self._process_detailed_with_chunking(transcript, system_prompt, language)
    
    def _process_detailed_with_chunking(self, transcript: str, system_prompt: str, language: str) -> Optional[str]:
        """Process long transcript with intelligent chunking for detailed structure"""
        # Create intelligent chunks
        chunks = self.text_chunker.create_intelligent_chunks(transcript)
        
        if len(chunks) == 1:
            return self._make_api_request(transcript, system_prompt)
        
        st.info(f"Processing {len(chunks)} chunks for detailed structuring...")
        
        # Process chunks in sequence (not parallel to maintain order)
        processed_chunks = []
        
        for i, chunk_info in enumerate(chunks):
            st.info(f"Processing detailed chunk {i+1}/{len(chunks)} (~{chunk_info['tokens_estimate']} tokens)")
            
            # Create context-aware prompt for this chunk
            chunk_prompt = self.text_chunker.create_chunk_specific_prompt(
                system_prompt, chunk_info, len(chunks), language
            )
            
            # Process chunk
            result = self._make_api_request(chunk_info['text'], chunk_prompt)
            
            if result:
                processed_chunks.append(result)
                st.success(f"Chunk {i+1} processed successfully")
            else:
                st.error(f"Chunk {i+1} failed")
                return None
        
        # Combine chunks intelligently
        st.info("Combining processed chunks into final detailed transcript...")
        combined_result = self._combine_processed_chunks(processed_chunks, language)
        
        return combined_result
    
    def _make_api_request(self, text: str, system_prompt: str) -> Optional[str]:
        """Make API request to DeepSeek"""
        try:
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

            # Increased timeout for better reliability with longer transcripts
            resp = requests.post(
                endpoint, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=180  # Increased to 180 seconds for deepseek-reasoner
            )
            
            if resp.status_code != 200:
                error_detail = ""
                try:
                    error_data = resp.json()
                    error_detail = error_data.get('error', {}).get('message', resp.text)
                except:
                    error_detail = resp.text
                
                raise RuntimeError(f"DeepSeek API error {resp.status_code}: {error_detail}")
            
            data = resp.json()
            
            if 'choices' not in data or len(data['choices']) == 0:
                raise RuntimeError("DeepSeek API returned no choices")
            
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise RuntimeError("DeepSeek API returned empty content")
                
            return content.strip()
            
        except requests.exceptions.Timeout:
            raise RuntimeError("DeepSeek API request timed out after 180 seconds")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Failed to connect to DeepSeek API - check internet connection")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API network error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def _combine_processed_chunks(self, chunk_results: List[str], language: str) -> str:
        """Intelligently combine processed chunks into final document"""
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Simple combination with section separation
        if language == "English":
            separator = "\n\n---\n\n"
            header = "# Detailed Structured Transcript\n\n"
            footer = "\n\n---\n*This document was processed in multiple chunks for optimal LLM performance.*"
        else:  # Chinese
            separator = "\n\n---\n\n"
            header = "# 详细结构化转录文稿\n\n"
            footer = "\n\n---\n*本文档经过多块处理以获得最佳LLM性能。*"
        
        # Combine chunks
        combined = header + separator.join(chunk_results) + footer
        
        return combined

class YouTubeDataProvider:
    """Provider for YouTube Data API operations using direct HTTP requests"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.available = bool(api_key)  # Simple check - just need API key
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        """
        Extract video URLs from a YouTube playlist using YouTube Data API v3.
        Returns list of dicts with 'title', 'url', and 'video_id' keys.
        """
        if not self.available or not self.api_key:
            st.error("YouTube Data API key not available.")
            return []
            
        playlist_id = self.extract_playlist_id(playlist_url)
        if not playlist_id:
            st.error("Could not extract playlist ID from URL")
            return []
        
        try:
            videos = []
            next_page_token = None
            
            while True:
                # Build the API request URL
                params = {
                    'part': 'snippet',
                    'playlistId': playlist_id,
                    'maxResults': 50,
                    'key': self.api_key
                }
                
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                # Make the HTTP request
                response = requests.get(
                    f"{self.base_url}/playlistItems",
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    error_msg = f"YouTube API error ({response.status_code})"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                    except:
                        error_msg += f": {response.text[:200]}"
                    
                    st.error(error_msg)
                    return []
                
                playlist_response = response.json()
                
                # Process the items
                for item in playlist_response.get('items', []):
                    snippet = item.get('snippet', {})
                    resource_id = snippet.get('resourceId', {})
                    
                    video_id = resource_id.get('videoId')
                    title = snippet.get('title', 'Unknown Title')
                    
                    if video_id:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        videos.append({
                            'title': title,
                            'url': video_url,
                            'video_id': video_id
                        })
                
                # Check for next page
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos
            
        except requests.exceptions.Timeout:
            st.error("YouTube API request timed out")
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"Network error accessing YouTube API: {e}")
            return []
        except Exception as e:
            st.error(f"Error processing YouTube API response: {e}")
            return []
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from YouTube playlist URL"""
        match = re.search(r'list=([\w-]+)', url)
        return match.group(1) if match else None

# ==================== ORCHESTRATOR LAYER ====================

class TranscriptOrchestrator:
    def __init__(
        self, 
        transcript_provider: TranscriptProvider,
        asr_fallback_provider: Optional[TranscriptProvider] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.transcript_provider = transcript_provider
        self.asr_fallback_provider = asr_fallback_provider
        self.llm_provider = llm_provider
    
    def get_transcript(self, url: str, language: str, use_fallback: bool = False) -> Optional[str]:
        # Try primary provider first
        st.info("Trying primary transcript provider (Supadata)...")
        transcript = self.transcript_provider.get_transcript(url, language)
        
        # Validate transcript quality
        def is_valid_transcript(text):
            """Check if transcript is valid and substantial enough"""
            if not text:
                return False
            
            # Check if it's just an error object string
            if "BatchJob" in str(text) or "job_id" in str(text):
                st.warning("Received job object instead of transcript - transcript not ready or unavailable")
                return False
            
            # Remove whitespace and check length
            cleaned = text.strip()
            if len(cleaned) < 100:  # Less than 100 characters is too short
                st.warning(f"Transcript too short ({len(cleaned)} characters) - likely invalid")
                return False
            
            # Check word count
            word_count = len(cleaned.split())
            if word_count < 20:  # Less than 20 words is suspiciously short
                st.warning(f"Transcript has only {word_count} words - likely invalid")
                return False
            
            return True
        
        # Check if transcript is valid
        if not is_valid_transcript(transcript):
            transcript = None
            st.warning("Primary provider returned invalid or too short transcript")
        
        # If no valid transcript and fallback is enabled, try ASR
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("No valid transcript from primary provider. Trying ASR fallback (AssemblyAI with chunking)...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
            # Validate ASR transcript too
            if not is_valid_transcript(transcript):
                st.error("ASR also failed to produce a valid transcript")
                transcript = None
        
        if not transcript:
            st.warning("No transcript available from any provider")
        else:
            st.success(f"Transcript obtained successfully ({len(transcript)} characters, {len(transcript.split())} words)")
            
        return transcript
    
    def structure_transcript(self, transcript: str, system_prompt: str, language: str = "English", is_custom_prompt: bool = False) -> Optional[Dict[str, str]]:
        if not self.llm_provider:
            return None
        return self.llm_provider.structure_transcript(transcript, system_prompt, language, is_custom_prompt)

# ==================== STREAMLIT UI ====================

def login_page():
    """Display login page"""
    st.title("YouTube Transcript Processor")
    st.subheader("Login Required")
    
    # Show default credentials info
    st.info("**Default Login:** Username: `admin` | Password: `admin123`")
    
    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
                return False
            
            auth_manager = AuthManager()
            if auth_manager.authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.api_keys = auth_manager.get_user_api_keys(username)
                st.session_state.user_data_manager = auth_manager.get_user_data_manager()
                
                # Initialize YouTube extractor for enhanced video metadata
                youtube_api_key = st.session_state.api_keys.get('youtube')
                auth_manager.initialize_user_data_manager_with_youtube(youtube_api_key)
                
                # Load user settings
                user_settings = st.session_state.user_data_manager.get_user_settings(username)
                st.session_state.user_settings = user_settings
                
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                st.error("Invalid credentials")
                return False
    
    return False

def show_history_detail_page(entry_id: str):
    """Display detailed view of a specific history entry"""
    st.header("📝 Transcript Details")
    
    username = st.session_state.username
    user_data_manager = st.session_state.user_data_manager
    
    # Get the specific entry
    entry = user_data_manager.get_history_entry(username, entry_id)
    
    if not entry:
        st.error("Entry not found!")
        if st.button("← Back to History"):
            st.session_state.current_page = "history"
            st.rerun()
        return
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to History", type="secondary"):
            st.session_state.current_page = "history"
            st.rerun()
    
    # Video Information Header
    with st.container():
        # Video thumbnail and basic info
        col1, col2 = st.columns([1, 3])
        
        with col1:
            thumbnail_url = entry.get('video_thumbnail')
            if thumbnail_url:
                st.image(thumbnail_url, width=200)
        
        with col2:
            st.title(entry.get('video_title', entry.get('title', 'Unknown Video')))
            
            # Video metadata
            if entry.get('video_channel'):
                st.write(f"**Channel:** {entry['video_channel']}")
            
            if entry.get('video_duration'):
                st.write(f"**Duration:** {entry['video_duration']}")
            
            if entry.get('video_published'):
                st.write(f"**Published:** {entry['video_published'][:10]}")
            
            st.write(f"**Processed:** {entry.get('timestamp', '')[:10]}")
            st.write(f"**Language:** {entry.get('language', 'N/A')}")
            st.write(f"**Model:** {entry.get('model_used', 'N/A')}")
            st.write(f"**Status:** {entry.get('status', 'N/A')}")
            st.write(f"**Processing Time:** {entry.get('processing_time', 'N/A')}")
    
    # Video description
    if entry.get('video_description'):
        with st.expander("📄 Video Description", expanded=False):
            st.write(entry['video_description'])
    
    # URL
    with st.expander("🔗 Video URL", expanded=False):
        st.write(entry.get('url', 'N/A'))
        if entry.get('url'):
            st.link_button("🎬 Open in YouTube", entry['url'])
    
    st.divider()
    
    # Processing Results
    if entry.get('status') == 'Completed':
        # Executive Summary
        if entry.get('executive_summary'):
            st.subheader("📋 Executive Summary")
            st.markdown(entry['executive_summary'])
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    "📄 Download Summary",
                    entry['executive_summary'],
                    file_name=f"summary_{entry.get('video_title', 'video')}.md",
                    mime="text/markdown"
                )
            
            st.divider()
        
        # Detailed Transcript
        if entry.get('detailed_transcript'):
            st.subheader("📝 Detailed Structured Transcript")
            
            # Search within transcript
            search_term = st.text_input("🔍 Search within transcript", placeholder="Enter keywords to highlight...")
            
            transcript_content = entry['detailed_transcript']
            
            # Highlight search terms if provided
            if search_term:
                # Simple highlighting (case-insensitive)
                highlighted_content = re.sub(
                    f"({re.escape(search_term)})", 
                    r"**\1**", 
                    transcript_content, 
                    flags=re.IGNORECASE
                )
                st.markdown(highlighted_content)
            else:
                st.markdown(transcript_content)
            
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                st.download_button(
                    "📄 Download Transcript",
                    entry['detailed_transcript'],
                    file_name=f"transcript_{entry.get('video_title', 'video')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                # Combined download
                if entry.get('executive_summary'):
                    combined_content = f"""# {entry.get('video_title', 'Video Analysis')}

**Video URL:** {entry.get('url', '')}
**Channel:** {entry.get('video_channel', 'Unknown')}
**Processed:** {entry.get('timestamp', '')[:10]}

## Executive Summary

{entry['executive_summary']}

---

## Detailed Transcript

{entry['detailed_transcript']}
"""
                    st.download_button(
                        "📦 Download Complete Analysis",
                        combined_content,
                        file_name=f"complete_analysis_{entry.get('video_title', 'video')}.md",
                        mime="text/markdown"
                    )
    
    elif entry.get('status') == 'Failed':
        st.error(f"Processing failed: {entry.get('error', 'Unknown error')}")
        
        # Option to retry processing
        if st.button("🔄 Retry Processing", type="primary"):
            st.info("Redirecting to processing page...")
            st.session_state.retry_url = entry.get('url')
            st.session_state.current_page = "main"
            st.rerun()
    
    else:
        st.info("Processing status: " + entry.get('status', 'Unknown'))

def show_history_page():
    """Display enhanced user's processing history"""
    st.header("📚 Processing History")
    
    username = st.session_state.username
    user_data_manager = st.session_state.user_data_manager
    
    # Get history and statistics
    history = user_data_manager.get_user_history(username)
    stats = user_data_manager.get_history_statistics(username)
    
    # Show statistics
    if stats["total_videos"] > 0:
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Videos", stats["total_videos"])
            
            with col2:
                st.metric("Avg Length", f"{stats['average_transcript_length']:,} chars")
            
            with col3:
                st.metric("Languages", ", ".join(stats["languages_used"]))
            
            with col4:
                st.metric("Success Rate", f"{len([h for h in history if h.get('status') == 'Completed'])}/{len(history)}")
        
        st.divider()
        
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search videos", placeholder="Search by title, channel, or URL...")
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All", "Completed", "Failed", "Processing"])
        with col3:
            language_filter = st.selectbox("Filter by Language", ["All"] + stats["languages_used"])
        
        # History management
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("Recent Videos")
        with col2:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
        with col3:
            if st.button("🗑️ Clear All", type="secondary"):
                if st.session_state.get('confirm_clear_history', False):
                    user_data_manager.clear_user_history(username)
                    st.success("History cleared!")
                    st.session_state.confirm_clear_history = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear_history = True
                    st.warning("Click again to confirm")
        
        # Filter history
        filtered_history = history
        
        if search_query:
            filtered_history = [
                entry for entry in filtered_history
                if search_query.lower() in entry.get('video_title', '').lower() or
                   search_query.lower() in entry.get('video_channel', '').lower() or
                   search_query.lower() in entry.get('url', '').lower()
            ]
        
        if status_filter != "All":
            filtered_history = [entry for entry in filtered_history if entry.get('status') == status_filter]
        
        if language_filter != "All":
            filtered_history = [entry for entry in filtered_history if entry.get('language') == language_filter]
        
        # Display history entries in a more visual way
        if filtered_history:
            for i, entry in enumerate(filtered_history[:20]):  # Show last 20 entries
                with st.container():
                    # Create a card-like layout
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    
                    with col1:
                        # Video title and channel
                        video_title = entry.get('video_title', entry.get('title', 'Unknown Video'))
                        st.markdown(f"**{video_title}**")
                        
                        if entry.get('video_channel'):
                            st.caption(f"📺 {entry['video_channel']}")
                        
                        # Status badge
                        status = entry.get('status', 'Unknown')
                        if status == 'Completed':
                            st.success(f"✅ {status}")
                        elif status == 'Failed':
                            st.error(f"❌ {status}")
                        else:
                            st.info(f"⏳ {status}")
                    
                    with col2:
                        st.write(f"**Processed:** {entry.get('timestamp', '')[:10]}")
                        st.write(f"**Language:** {entry.get('language', 'N/A')}")
                        st.write(f"**Length:** {entry.get('transcript_length', 0):,} chars")
                        if entry.get('processing_time'):
                            st.write(f"**Time:** {entry['processing_time']}")
                    
                    with col3:
                        # View details button
                        if st.button("👁️ View Details", key=f"view_{entry.get('id', i)}", type="primary"):
                            st.session_state.current_page = "history_detail"
                            st.session_state.current_entry_id = entry.get('id')
                            st.rerun()
                        
                        # Quick download for completed items
                        if entry.get('status') == 'Completed' and entry.get('executive_summary'):
                            st.download_button(
                                "📋 Summary",
                                entry['executive_summary'],
                                file_name=f"summary_{entry.get('id', i)}.md",
                                mime="text/markdown",
                                key=f"download_summary_{entry.get('id', i)}"
                            )
                    
                    with col4:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{entry.get('id', i)}", type="secondary", help="Delete this entry"):
                            user_data_manager.delete_history_entry(username, entry.get('id'))
                            st.success("Entry deleted!")
                            st.rerun()
                        
                        # Retry button for failed entries
                        if entry.get('status') == 'Failed':
                            if st.button("🔄", key=f"retry_{entry.get('id', i)}", type="secondary", help="Retry processing"):
                                st.session_state.retry_url = entry.get('url')
                                st.session_state.current_page = "main"
                                st.rerun()
                
                st.divider()
        
        else:
            if search_query or status_filter != "All" or language_filter != "All":
                st.info("No videos match your search criteria.")
            else:
                st.info("No processing history found.")
        
        # Export option
        st.divider()
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Export All Data", type="secondary"):
                exported_data = user_data_manager.export_user_data(username)
                st.download_button(
                    "📤 Download Complete Export",
                    exported_data,
                    file_name=f"{username}_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        st.info("No processing history yet. Start by processing some YouTube videos!")
        if st.button("➡️ Go to Video Processing"):
            st.session_state.current_page = "main"
            st.rerun()

def show_settings_page():
    """Display user settings management"""
    st.header("⚙️ User Settings")
    
    username = st.session_state.username
    user_data_manager = st.session_state.user_data_manager
    
    # Load current settings
    current_settings = user_data_manager.get_user_settings(username)
    
    with st.form("settings_form"):
        st.subheader("Default Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Default Language",
                ["English", "中文"],
                index=0 if current_settings.get("language", "English") == "English" else 1,
                help="Default language for transcription"
            )
            
            use_asr_fallback = st.checkbox(
                "Enable ASR Fallback by Default",
                value=current_settings.get("use_asr_fallback", True),
                help="Use AssemblyAI when official captions are not available"
            )
            
            deepseek_model = st.selectbox(
                "Default DeepSeek Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0 if current_settings.get("deepseek_model", "deepseek-reasoner") == "deepseek-chat" else 1,
                help="Default model for processing"
            )
        
        with col2:
            temperature = st.slider(
                "Default Temperature",
                min_value=0.0,
                max_value=1.0,
                value=current_settings.get("temperature", 0.1),
                step=0.1,
                help="Controls randomness in LLM responses"
            )
            
            browser_for_cookies = st.selectbox(
                "Default Browser for Cookies",
                ["none", "chrome", "firefox", "edge", "safari"],
                index=["none", "chrome", "firefox", "edge", "safari"].index(
                    current_settings.get("browser_for_cookies", "none")
                ),
                help="Default browser for YouTube cookies"
            )
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            save_settings = st.form_submit_button("💾 Save Settings", type="primary")
        
        with col2:
            reset_settings = st.form_submit_button("🔄 Reset to Defaults", type="secondary")
        
        if save_settings:
            new_settings = {
                "language": language,
                "use_asr_fallback": use_asr_fallback,
                "deepseek_model": deepseek_model,
                "temperature": temperature,
                "browser_for_cookies": browser_for_cookies
            }
            
            user_data_manager.save_user_settings(username, new_settings)
            st.session_state.user_settings = new_settings
            st.success("Settings saved successfully!")
            st.rerun()
        
        if reset_settings:
            default_settings = {
                "language": "English",
                "use_asr_fallback": True,
                "deepseek_model": "deepseek-reasoner",
                "temperature": 0.1,
                "browser_for_cookies": "none"
            }
            
            user_data_manager.save_user_settings(username, default_settings)
            st.session_state.user_settings = default_settings
            st.success("Settings reset to defaults!")
            st.rerun()

def main_app():
    """Main application interface with enhanced persistence and navigation"""
    st.title("YouTube Transcript Processor")
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    # Sidebar navigation
    with st.sidebar:
        st.write(f"Welcome, **{st.session_state.username}**!")
        st.divider()
        
        # Navigation - only update if not on a detail page
        page_options = ["🎬 Process Videos", "📚 History", "⚙️ Settings"]
        current_index = 0
        
        # Map current page to index, treating detail pages as their parent
        if st.session_state.current_page in ["history", "history_detail"]:
            current_index = 1
        elif st.session_state.current_page == "settings":
            current_index = 2
        
        # Only process radio button changes if we're not on a detail page
        if st.session_state.current_page != "history_detail":
            page = st.radio(
                "Navigation",
                page_options,
                index=current_index,
                key="navigation"
            )
            
            # Update current page based on navigation
            if page == "🎬 Process Videos":
                st.session_state.current_page = "main"
            elif page == "📚 History":
                st.session_state.current_page = "history"
            elif page == "⚙️ Settings":
                st.session_state.current_page = "settings"
        else:
            # Show navigation but don't allow changes when on detail page
            st.radio(
                "Navigation",
                page_options,
                index=current_index,
                disabled=True,
                help="Return to History to change navigation"
            )
        
        st.divider()
        
        # Quick stats
        if 'user_data_manager' in st.session_state:
            try:
                stats = st.session_state.user_data_manager.get_history_statistics(st.session_state.username)
                st.metric("Videos Processed", stats["total_videos"])
                st.metric("Total Characters", f"{stats['total_transcript_length']:,}")
            except Exception as e:
                st.caption("Stats unavailable")
        
        st.divider()
        
        # Logout button
        if st.button("🚪 Logout", type="secondary"):
            for key in ['authenticated', 'username', 'api_keys', 'user_data_manager', 'user_settings', 'current_page', 'current_entry_id', 'retry_url']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Main content based on current page
    if st.session_state.current_page == "history_detail":
        entry_id = st.session_state.get('current_entry_id')
        if entry_id:
            show_history_detail_page(entry_id)
        else:
            st.error("No entry selected")
            st.session_state.current_page = "history"
            st.rerun()
    elif st.session_state.current_page == "history":
        show_history_page()
    elif st.session_state.current_page == "settings":
        show_settings_page()
    else:  # Default to Process Videos
        show_main_processing_page()

def show_main_processing_page():
    """Show the main video processing interface"""
    
    # Get API keys and user settings
    api_keys = st.session_state.get('api_keys', {})
    user_settings = st.session_state.get('user_settings', {})
    
    # Check if there's a retry URL from failed processing
    retry_url = st.session_state.get('retry_url')
    if retry_url:
        st.info(f"Retrying processing for: {retry_url}")
        st.session_state.retry_url = None  # Clear it
    
    # Configuration Section
    st.header("Configuration")
    
    with st.expander("API Status", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if api_keys.get('supadata'):
                st.success("Supadata")
            else:
                st.error("Supadata")
        
        with col2:
            if api_keys.get('assemblyai'):
                st.success("AssemblyAI")
            else:
                st.error("AssemblyAI")
        
        with col3:
            if api_keys.get('deepseek'):
                st.success("DeepSeek")
            else:
                st.error("DeepSeek")
        
        with col4:
            if api_keys.get('youtube'):
                st.success("YouTube Data")
            else:
                st.error("YouTube Data")
    
    # Settings - Load from user preferences
    with st.expander("Processing Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            language = st.selectbox(
                "Language",
                ["English", "中文"],
                index=0 if user_settings.get("language", "English") == "English" else 1,
                help="Select the language for transcription"
            )
            
            use_asr_fallback = st.checkbox(
                "Enable ASR Fallback",
                value=user_settings.get("use_asr_fallback", True),
                help="Use AssemblyAI when official captions are not available"
            )
        
        with col2:
            deepseek_model = st.selectbox(
                "DeepSeek Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0 if user_settings.get("deepseek_model", "deepseek-reasoner") == "deepseek-chat" else 1,
                help="deepseek-chat: Faster processing (~5-15s per call), deepseek-reasoner: Higher quality but slower (~30-60s per call)"
            )
            
            # Performance mode selection
            performance_mode = st.selectbox(
                "Processing Mode",
                ["Balanced", "Speed Optimized", "Quality Optimized"],
                index=0,
                help="Speed: Parallel processing, smaller chunks, faster model. Quality: Sequential processing, larger context, reasoning model. Balanced: Good compromise."
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=user_settings.get("temperature", 0.1),
                step=0.1,
                help="Controls randomness in LLM responses"
            )
        
        with col3:
            browser_for_cookies = st.selectbox(
                "Browser for YouTube Cookies",
                ["none", "chrome", "firefox", "edge", "safari"],
                index=["none", "chrome", "firefox", "edge", "safari"].index(
                    user_settings.get("browser_for_cookies", "none")
                ),
                help="Select browser for cookies (use 'none' on Linux/servers)"
            )
            
            # Auto-save settings option
            if st.checkbox("Auto-save these settings", value=True):
                # Save current settings automatically when changed
                current_form_settings = {
                    "language": language,
                    "use_asr_fallback": use_asr_fallback,
                    "deepseek_model": deepseek_model,
                    "temperature": temperature,
                    "browser_for_cookies": browser_for_cookies
                }
                
                # Check if settings changed
                if current_form_settings != user_settings:
                    st.session_state.user_data_manager.save_user_settings(
                        st.session_state.username, 
                        current_form_settings
                    )
                    st.session_state.user_settings = current_form_settings
                    st.success("✅ Settings auto-saved!")
    
    # Cookie file upload section (for Linux/containers)
    with st.expander("YouTube Authentication (for ASR)", expanded=False):
        # Check for Render secret file
        render_cookies = '/etc/secrets/youtube_cookies.txt'
        if os.path.exists(render_cookies):
            st.success(f"✅ Render secret cookies file found: {render_cookies}")
            st.info("The app will automatically use this file for YouTube authentication")
        else:
            # Check for environment variable
            env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
            if env_cookies and os.path.exists(env_cookies):
                st.success(f"✅ Cookies file from environment: {env_cookies}")
            else:
                st.warning("⚠️ No cookies file found in /etc/secrets/youtube_cookies.txt")
                st.info("**To fix this on Render:**\n"
                       "1. Export cookies from YouTube (logged in)\n"
                       "2. Add as Secret File in Render Dashboard\n"
                       "3. Name it: youtube_cookies.txt")
        
        st.markdown("---")
        st.info("**Manual Cookie Upload (Alternative):**")
        st.markdown("""
        1. Install browser extension: 'Get cookies.txt' or 'cookies.txt'
        2. Log into YouTube in your browser
        3. Export cookies using the extension
        4. Upload the cookies.txt file below
        """)
        
        uploaded_cookies = st.file_uploader(
            "Upload cookies.txt file (optional)",
            type=['txt'],
            help="Upload YouTube cookies exported from browser"
        )
        
        if uploaded_cookies:
            # Save uploaded cookies to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                content = uploaded_cookies.read().decode('utf-8')
                f.write(content)
                cookies_path = f.name
            
            os.environ['YOUTUBE_COOKIES_FILE'] = cookies_path
            st.success(f"Cookies file uploaded and set: {cookies_path}")
    
    # Main Processing Section
    st.header("Process Video")
    
    # Input methods - updated with direct transcript option
    input_method = st.radio(
        "Input Method",
        ["Single Video URL", "Direct Transcript Input", "Playlist URL", "Batch URLs"],
        help="Choose how to input videos or transcripts"
    )
    
    videos_to_process = []
    direct_transcript = None
    
    if input_method == "Single Video URL":
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            value=retry_url if retry_url else "",
            help="Enter a single YouTube video URL"
        )
        if video_url:
            videos_to_process = [{"title": "Single Video", "url": video_url, "type": "url"}]
    
    elif input_method == "Direct Transcript Input":
        st.subheader("📝 Direct Transcript Input")
        st.info("💡 **Tip:** This option allows you to directly paste a transcript for structuring without needing a YouTube URL.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Title input for the transcript
            transcript_title = st.text_input(
                "Transcript Title",
                placeholder="e.g., 'AI Conference Keynote' or 'Product Launch Presentation'",
                help="Give your transcript a descriptive title for easy identification"
            )
        
        with col2:
            # Optional source URL
            source_url = st.text_input(
                "Source URL (Optional)",
                placeholder="https://example.com/source",
                help="Optional: Add source URL for reference"
            )
        
        # Transcript text input
        transcript_text = st.text_area(
            "Paste Your Transcript Here",
            placeholder="""Paste your transcript text here...

Example:
Hello everyone, welcome to today's presentation. 
We'll be covering three main topics: artificial intelligence, 
machine learning, and their applications in business...

[Paste your complete transcript text]""",
            height=300,
            help="Paste the complete transcript you want to structure"
        )
        
        # Character count and language detection preview
        if transcript_text:
            char_count = len(transcript_text)
            word_count = len(transcript_text.split())
            estimated_tokens = char_count // 4
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{char_count:,}")
            with col2:
                st.metric("Words", f"{word_count:,}")
            with col3:
                st.metric("Est. Tokens", f"{estimated_tokens:,}")
            
            # Preview language detection
            from datetime import datetime
            detector = LanguageDetector()
            detected_lang = detector.detect_language(transcript_text)
            preview_lang = detector.get_language_code(detected_lang)
            
            st.info(f"🔍 **Detected Language:** {preview_lang}")
            
            if transcript_title and transcript_text:
                # Create a pseudo-video entry for processing
                videos_to_process = [{
                    "title": transcript_title,
                    "url": source_url if source_url else f"direct_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "type": "direct_transcript",
                    "transcript": transcript_text
                }]
                
                st.success(f"✅ Ready to process: **{transcript_title}** ({char_count:,} characters)")
    
    elif input_method == "Playlist URL":
        playlist_url = st.text_input(
            "YouTube Playlist URL",
            placeholder="https://www.youtube.com/playlist?list=...",
            help="Enter a YouTube playlist URL"
        )
        if playlist_url and api_keys.get('youtube'):
            if st.button("Load Playlist"):
                with st.spinner("Loading playlist videos..."):
                    youtube_provider = YouTubeDataProvider(api_keys['youtube'])
                    videos_to_process = youtube_provider.get_playlist_videos(playlist_url)
                    if videos_to_process:
                        # Mark as URL type
                        for video in videos_to_process:
                            video["type"] = "url"
                        st.success(f"Loaded {len(videos_to_process)} videos from playlist")
                        st.session_state.playlist_videos = videos_to_process
    
    elif input_method == "Batch URLs":
        batch_urls = st.text_area(
            "YouTube Video URLs (one per line)",
            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
            help="Enter multiple YouTube video URLs, one per line"
        )
        if batch_urls:
            urls = [url.strip() for url in batch_urls.split('\n') if url.strip()]
            videos_to_process = [{"title": f"Video {i+1}", "url": url, "type": "url"} for i, url in enumerate(urls)]
    
    # Show loaded videos
    if input_method == "Playlist URL" and 'playlist_videos' in st.session_state:
        videos_to_process = st.session_state.playlist_videos
    
    if videos_to_process:
        st.subheader(f"Videos to Process ({len(videos_to_process)})")
        
        # Show video selection for playlists/batch
        if len(videos_to_process) > 1:
            with st.expander("Select Videos to Process", expanded=True):
                selected_videos = []
                select_all = st.checkbox("Select All", value=True)
                
                for i, video in enumerate(videos_to_process):
                    if select_all:
                        selected = True
                    else:
                        selected = st.checkbox(f"{video['title'][:50]}...", key=f"video_{i}")
                    
                    if selected:
                        selected_videos.append(video)
                
                videos_to_process = selected_videos
        
        # Custom system prompt - dynamic based on detected language
        st.subheader("System Prompt")
        
        # Default prompts
        default_prompts = {
            "English": """You are an expert at analyzing and structuring YouTube video transcripts. Your task is to convert raw transcript text into a well-organized, readable document.

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

Format the output as a clean, professional document that would be easy to read and reference.""",
            
            "繁體中文": """你是一個專業的YouTube影片轉錄文本分析和結構化專家。你的任務是將原始轉錄文本轉換為組織良好、易於閱讀的文檔。

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

將輸出格式化為一個清潔、專業的文檔，便於閱讀和參考。"""
        }
        
        # Determine which prompt to show based on content language detection
        prompt_language = language  # Default to UI language
        
        # For direct transcript input, detect language from content
        if input_method == "Direct Transcript Input" and 'transcript_text' in locals() and transcript_text:
            detector = LanguageDetector()
            detected_lang = detector.detect_language(transcript_text)
            if detected_lang == "Chinese":
                prompt_language = "繁體中文"
                st.info(f"🔍 **System prompt adapted to detected language:** {prompt_language}")
            else:
                prompt_language = "English"
                st.info(f"🔍 **System prompt adapted to detected language:** {prompt_language}")
        
        # For URL-based input, use UI language selection but show hint
        elif input_method != "Direct Transcript Input":
            if language == "中文":
                prompt_language = "繁體中文"
            st.info(f"💡 **System prompt language:** {prompt_language} (Will auto-adapt based on video content during processing)")
        
        system_prompt = st.text_area(
            "System Prompt for Detailed Transcript",
            value=default_prompts[prompt_language],
            height=200,
            help="Customize how the AI should structure the detailed transcript. This prompt will be used as-is, so modify it according to your needs."
        )
        
        # Show warning if user modified the prompt
        if system_prompt != default_prompts[prompt_language]:
            st.success("✅ **Custom system prompt detected** - Your modifications will be used for processing!")
        
        # Reset prompt button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Reset to Default", help="Reset to default system prompt"):
                st.rerun()
        
        # Processing button
        if st.button("Start Processing", type="primary"):
            # Check requirements based on input method
            if input_method == "Direct Transcript Input":
                # For direct input, only need DeepSeek
                if not api_keys.get('deepseek'):
                    st.error("DeepSeek API key required for structuring transcripts. Please configure API keys.")
                    return
                
                if not videos_to_process:
                    st.error("Please provide a transcript title and paste your transcript content.")
                    return
                    
                st.info("🚀 Processing direct transcript input - skipping video extraction...")
                
            else:
                # For URL-based processing, need transcript providers
                if not api_keys.get('supadata') and not api_keys.get('assemblyai'):
                    st.error("No transcript providers available. Please configure API keys.")
                    return
                
                if not api_keys.get('deepseek'):
                    st.error("DeepSeek API key required for structuring. Please configure API keys.")
                    return
            
            process_videos_with_history(videos_to_process, language, use_asr_fallback, system_prompt, 
                          deepseek_model, temperature, api_keys, browser_for_cookies, performance_mode)

def process_videos_with_history(videos, language, use_asr_fallback, system_prompt, deepseek_model, temperature, api_keys, browser='chrome', performance_mode="Balanced"):
    """Process multiple videos/transcripts with performance optimization"""
    
    # Get user data manager
    username = st.session_state.username
    user_data_manager = st.session_state.user_data_manager
    
    # Initialize providers only if needed
    supadata_provider = None
    assemblyai_provider = None
    
    # Check if we need transcript extraction providers
    needs_extraction = any(video.get("type") == "url" for video in videos)
    
    if needs_extraction:
        supadata_provider = SupadataTranscriptProvider(api_keys.get('supadata', ''))
        assemblyai_provider = ImprovedAssemblyAITranscriptProvider(
            api_keys.get('assemblyai', ''), 
            browser=browser
        ) if use_asr_fallback else None
    
# Use performance-optimized DeepSeek provider
deepseek_provider = FastDeepSeekProvider(
    api_keys.get('deepseek', ''),
    "https://api.deepseek.com/v1",
    deepseek_model,
    temperature,
    performance_mode
)
    
    # Show performance info
    if performance_mode == "Speed Optimized":
        st.info(f"🚀 **Speed Mode Active**: Using {deepseek_model} with parallel processing and optimized chunking")
        if deepseek_model == "deepseek-reasoner":
            st.warning("💡 **Tip**: Switch to 'deepseek-chat' model for even faster processing in Speed mode")
    elif performance_mode == "Quality Optimized":
        st.info(f"🎯 **Quality Mode Active**: Using {deepseek_model} with sequential processing and large context windows")
        if deepseek_model == "deepseek-chat":
            st.info("💡 **Note**: Consider 'deepseek-reasoner' for highest quality output in Quality mode")
    else:
        st.info(f"⚖️ **Balanced Mode Active**: Using {deepseek_model} with optimized processing")
    
    orchestrator = TranscriptOrchestrator(
        supadata_provider,
        assemblyai_provider,
        deepseek_provider
    )
    
    # Process each video/transcript
    total_start_time = time.time()
    
    for i, video in enumerate(videos):
        st.subheader(f"Processing Item {i+1}/{len(videos)}")
        st.write(f"**Title:** {video['title']}")
        
        # Show different info based on type
        if video.get("type") == "direct_transcript":
            st.write(f"**Type:** Direct Transcript Input")
            if video.get("url") and not video["url"].startswith("direct_input_"):
                st.write(f"**Source URL:** {video['url']}")
        else:
            st.write(f"**URL:** {video['url']}")
        
        # Initialize history entry
        start_time = time.time()
        history_entry = {
            "title": video['title'],
            "url": video['url'],
            "language": language,
            "model_used": deepseek_model,
            "performance_mode": performance_mode,
            "use_asr_fallback": use_asr_fallback,
            "input_type": video.get("type", "url"),
            "status": "Processing",
            "transcript_length": 0,
            "processing_time": "0s",
            "executive_summary": None,
            "detailed_transcript": None,
            "error": None
        }
        
        try:
            # Step 1: Get transcript (either from URL or direct input)
            extraction_start = time.time()
            
            if video.get("type") == "direct_transcript":
                # Use provided transcript directly
                transcript = video.get("transcript", "")
                
                if not transcript:
                    st.error("No transcript content provided")
                    history_entry["status"] = "Failed"
                    history_entry["error"] = "No transcript content provided"
                    user_data_manager.add_to_history(username, history_entry)
                    continue
                
                st.success(f"✅ Using provided transcript ({len(transcript)} characters)")
                
            else:
                # Extract transcript from URL
                with st.spinner("Getting transcript from video..."):
                    transcript = orchestrator.get_transcript(video['url'], language, use_asr_fallback)
                
                if not transcript:
                    st.error("Failed to get transcript")
                    history_entry["status"] = "Failed"
                    history_entry["error"] = "Failed to get transcript from video"
                    user_data_manager.add_to_history(username, history_entry)
                    continue
                
                extraction_time = time.time() - extraction_start
                st.info(f"⏱️ Transcript extraction completed in {extraction_time:.1f}s")
            
            # Update history entry with transcript info
            history_entry["transcript_length"] = len(transcript)
            
            # Step 2: Structure transcript with executive summary (with performance monitoring)
            processing_start = time.time()
            
            with st.spinner(f"Processing with {performance_mode} mode..."):
                # Check if user has customized the system prompt
                default_prompts = {
                    "English": """You are an expert at analyzing and structuring YouTube video transcripts. Your task is to convert raw transcript text into a well-organized, readable document.

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

Format the output as a clean, professional document that would be easy to read and reference.""",
                    
                    "繁體中文": """你是一個專業的YouTube影片轉錄文本分析和結構化專家。你的任務是將原始轉錄文本轉換為組織良好、易於閱讀的文檔。

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

將輸出格式化為一個清潔、專業的文檔，便於閱讀和參考。"""
                }
                
                # Determine if the system prompt has been customized
                is_custom_prompt = (system_prompt not in default_prompts.values())
                
                result = orchestrator.structure_transcript(transcript, system_prompt, language, is_custom_prompt)
            
            processing_time = time.time() - processing_start
            st.success(f"🎉 LLM processing completed in {processing_time:.1f}s")
            
            if not result:
                st.error("Failed to structure transcript")
                history_entry["status"] = "Failed"
                history_entry["error"] = "Failed to structure transcript"
                user_data_manager.add_to_history(username, history_entry)
                continue
            
            # Update history entry with results
            total_processing_time = time.time() - start_time
            history_entry["processing_time"] = f"{total_processing_time:.1f}s"
            history_entry["llm_processing_time"] = f"{processing_time:.1f}s"
            history_entry["status"] = "Completed"
            history_entry["executive_summary"] = result.get('executive_summary')
            history_entry["detailed_transcript"] = result.get('detailed_transcript')
            history_entry["detected_language"] = result.get('detected_language', language)
            history_entry["processing_mode"] = result.get('processing_mode', 'unknown')
            
            # Save to history
            user_data_manager.add_to_history(username, history_entry)
            
            # Display results
            st.success(f"✅ Processing completed! Total time: {total_processing_time:.1f}s")
            
            # Show performance breakdown
            if video.get("type") != "direct_transcript":
                st.info(f"📊 **Performance**: Extraction: {extraction_time:.1f}s | LLM Processing: {processing_time:.1f}s")
            else:
                st.info(f"📊 **Performance**: LLM Processing: {processing_time:.1f}s")
            
            # Show executive summary first
            if result.get('executive_summary'):
                with st.expander("📋 Executive Summary", expanded=True):
                    st.markdown(result['executive_summary'])
            
            # Show detailed structured transcript
            if result.get('detailed_transcript'):
                with st.expander("📝 Detailed Structured Transcript", expanded=False):
                    st.markdown(result['detailed_transcript'])
            
            # Download options
            safe_title = "".join(c for c in video['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if video.get("type") != "direct_transcript":  # Only show raw transcript download for URL-based
                    st.download_button(
                        "Download Raw Transcript",
                        transcript,
                        file_name=f"raw_transcript_{safe_title}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if result.get('executive_summary'):
                    st.download_button(
                        "Download Executive Summary",
                        result['executive_summary'],
                        file_name=f"executive_summary_{safe_title}.md",
                        mime="text/markdown"
                    )
            
            with col3:
                if result.get('detailed_transcript'):
                    st.download_button(
                        "Download Detailed Transcript",
                        result['detailed_transcript'],
                        file_name=f"detailed_transcript_{safe_title}.md",
                        mime="text/markdown"
                    )
            
            # Combined download option
            if result.get('executive_summary') and result.get('detailed_transcript'):
                combined_content = f"""# Complete Transcript Analysis: {video['title']}

{"**Source URL:** " + video['url'] if video.get("url") and not video["url"].startswith("direct_input_") else "**Type:** Direct Transcript Input"}
**Processed:** {history_entry.get('timestamp', '')[:10]}
**Model:** {deepseek_model} ({performance_mode} mode)
**Processing Time:** {total_processing_time:.1f}s
**Detected Language:** {result.get('detected_language', 'Unknown')}

## Executive Summary

{result['executive_summary']}

---

## Detailed Transcript

{result['detailed_transcript']}
"""
                st.download_button(
                    "📄 Download Complete Analysis",
                    combined_content,
                    file_name=f"complete_analysis_{safe_title}.md",
                    mime="text/markdown",
                    type="primary"
                )
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing item: {str(e)}")
            
            # Update history entry with error
            processing_time = time.time() - start_time
            history_entry["processing_time"] = f"{processing_time:.1f}s"
            history_entry["status"] = "Failed"
            history_entry["error"] = str(e)
            user_data_manager.add_to_history(username, history_entry)
            continue
    
    # Show completion message with performance summary
    total_batch_time = time.time() - total_start_time
    completed_count = len([v for v in videos])
    avg_time = total_batch_time / completed_count if completed_count > 0 else 0
    
    st.success(f"🎉 Batch processing completed! {completed_count} items processed in {total_batch_time:.1f}s (avg: {avg_time:.1f}s per item)")
    st.info("💡 Check your **History** page to review all processed content and re-download anytime!")

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point with enhanced persistence"""
    st.set_page_config(
        page_title="YouTube Transcript Processor",
        page_icon="🎬",
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
