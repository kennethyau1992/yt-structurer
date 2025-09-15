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
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                st.error("Invalid credentials")
                return False
    
    # Debug section for login issues
    with st.expander("Login Debug", expanded=False):
        st.subheader("Available Users")
        auth_manager = AuthManager()
        
        for user in auth_manager.users.keys():
            st.text(f"User: {user}")
        
        # Check admin password hash
        admin_user = auth_manager.users.get("admin", {})
        expected_hash = auth_manager._hash_password("admin123")
        actual_hash = admin_user.get("password_hash", "")
        
        st.text(f"Expected hash (admin123): {expected_hash[:16]}...")
        st.text(f"Actual hash: {actual_hash[:16]}...")
        st.text(f"Hashes match: {expected_hash == actual_hash}")
        
        # Check if ADMIN_PASSWORD env var is set
        env_admin_password = os.getenv("ADMIN_PASSWORD")
        if env_admin_password:
            st.text(f"ADMIN_PASSWORD env var is set to: {env_admin_password[:4]}****")
        else:
            st.text("ADMIN_PASSWORD env var not set - using default 'admin123'")
    
    return False

def main_app():
    """Main application interface"""
    st.title("YouTube Transcript Processor")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            for key in ['authenticated', 'username', 'api_keys']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.write(f"Welcome, **{st.session_state.username}**!")
    
    # Get API keys from session
    api_keys = st.session_state.get('api_keys', {})
    
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
    
    # Improved debug section
    with st.expander("Debug Info", expanded=False):
        st.subheader("Environment Variables Check")
        
        # Check only the env vars that your app actually uses
        user_prefix = st.session_state.username.upper()
        expected_keys = [
            f"{user_prefix}_SUPADATA_KEY",
            f"{user_prefix}_ASSEMBLYAI_KEY", 
            f"{user_prefix}_DEEPSEEK_KEY",
            f"{user_prefix}_YOUTUBE_KEY"
        ]
        
        for key in expected_keys:
            value = os.getenv(key)
            if value:
                masked_value = value[:4] + "*" * max(0, len(value) - 8) + value[-4:] if len(value) > 8 else "****"
                st.success(f"{key}: {masked_value}")
            else:
                st.error(f"{key}: Not found")
        
        st.subheader("Loaded API Keys for Current User")
        for key, value in api_keys.items():
            if value:
                masked_value = value[:4] + "*" * max(0, len(value) - 8) + value[-4:] if len(value) > 8 else "****" 
                st.success(f"{key}: {masked_value}")
            else:
                st.error(f"{key}: Not loaded")
        
        # Additional debugging info
        st.subheader("Authentication Status")
        st.info(f"Current user: {st.session_state.username}")
        st.info(f"Keys loaded: {len([k for k, v in api_keys.items() if v])}/4")
        
        # Test provider availability
        st.subheader("Provider Availability")
        
        # Test Supadata
        if api_keys.get('supadata'):
            supadata_provider = SupadataTranscriptProvider(api_keys['supadata'])
            if supadata_provider.available:
                st.success("Supadata: Ready")
            else:
                st.warning("Supadata: API key set but client not available (missing supadata package?)")
        else:
            st.error("Supadata: No API key")
        
        # Test AssemblyAI
        if api_keys.get('assemblyai'):
            st.success("AssemblyAI: Ready")
        else:
            st.error("AssemblyAI: No API key")
        
        # Test DeepSeek
        if api_keys.get('deepseek'):
            st.success("DeepSeek: Ready")
        else:
            st.error("DeepSeek: No API key")
        
        # Test YouTube Data API
        if api_keys.get('youtube'):
            youtube_provider = YouTubeDataProvider(api_keys['youtube'])
            if youtube_provider.available:
                st.success("YouTube Data API: Ready")
            else:
                st.error("YouTube Data API: Not available")
        else:
            st.error("YouTube Data API: No API key")
    
    # Settings
    with st.expander("Processing Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            language = st.selectbox(
                "Language",
                ["English", "中文"],
                help="Select the language for transcription"
            )
            
            use_asr_fallback = st.checkbox(
                "Enable ASR Fallback",
                value=True,
                help="Use AssemblyAI when official captions are not available"
            )
        
        with col2:
            deepseek_model = st.selectbox(
                "DeepSeek Model",
                ["deepseek-chat", "deepseek-reasoner"],
                index=1,
                help="Select the DeepSeek model to use"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Controls randomness in LLM responses"
            )
        
        with col3:
            browser_for_cookies = st.selectbox(
                "Browser for YouTube Cookies",
                ["chrome", "firefox", "edge", "safari"],
                help="Select which browser to extract YouTube cookies from (for ASR)"
            )
    
    # Main Processing Section
    st.header("Process Video")
    
    # Input methods
    input_method = st.radio(
        "Input Method",
        ["Single Video URL", "Playlist URL", "Batch URLs"],
        help="Choose how to input videos"
    )
    
    videos_to_process = []
    
    if input_method == "Single Video URL":
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a single YouTube video URL"
        )
        if video_url:
            videos_to_process = [{"title": "Single Video", "url": video_url}]
    
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
            videos_to_process = [{"title": f"Video {i+1}", "url": url} for i, url in enumerate(urls)]
    
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
        
        # Custom system prompt
        st.subheader("System Prompt")
        
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
            
            "中文": """你是一个专业的YouTube视频转录文本分析和结构化专家。你的任务是将原始转录文本转换为组织良好、易于阅读的文档。

请按照以下指导原则来结构化转录文本：

1. **创建清晰的章节和标题**，基于主题变化和内容流程
2. **提高可读性**：
   - 修正语法和标点符号
   - 删除填充词（嗯、呃、那个、就是说）
   - 合并断裂的句子
   - 添加段落分隔以改善流畅性

3. **保留所有重要信息** - 不要总结或省略内容
4. **使用markdown格式** 来设置标题、强调和结构
5. **在自然主题转换处添加时间戳**
6. **保持说话者的语调和意思**，同时提高清晰度

将输出格式化为一个清洁、专业的文档，便于阅读和参考。"""
        }
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
            # Try to use cookies from browser
            base_opts['cookiesfrombrowser'] = (browser,)
            
            # Alternative: Use a cookies file if provided
            cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
            if cookies_file and os.path.exists(cookies_file):
                base_opts['cookiefile'] = cookies_file
                del base_opts['cookiesfrombrowser']  # Use file instead of browser
        
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

        # Plain string already
        if isinstance(resp, str):
            return resp

        # Dict shapes: {"text": "..."} or {"chunks": [...]}
        if isinstance(resp, dict):
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

        # Fallback stringify
        try:
            return str(resp)
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
            
            st.info(f"Extracting audio stream URL using yt-dlp (trying cookies from {browser})...")
            
            # Configure yt-dlp options with cookies
            ydl_opts = self.cookie_manager.get_ydl_opts(use_cookies=True, browser=browser)
            ydl_opts.update({
                'format': 'bestaudio/best',  # Get best audio quality
                'noplaylist': True,          # Only process single video
            })
            
            # Try different browsers if the first one fails
            browsers_to_try = [browser, 'firefox', 'edge', 'safari', 'chrome']
            browsers_to_try = list(dict.fromkeys(browsers_to_try))  # Remove duplicates while preserving order
            
            last_error = None
            for browser_attempt in browsers_to_try:
                try:
                    ydl_opts['cookiesfrombrowser'] = (browser_attempt,)
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Extract video info without downloading
                        info = ydl.extract_info(youtube_url, download=False)
                        
                        if not info:
                            continue
                        
                        # Look for the best audio stream URL
                        formats = info.get('formats', [])
                        
                        # First, try to find audio-only streams
                        audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                        
                        if audio_formats:
                            # Sort by audio bitrate (highest first)
                            audio_formats.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                            best_audio = audio_formats[0]
                            st.success(f"Found audio-only stream using {browser_attempt} cookies: {best_audio.get('format_note', 'unknown quality')}")
                            return best_audio.get('url')
                        
                        # Fallback: find formats with audio (including video+audio)
                        formats_with_audio = [f for f in formats if f.get('acodec') != 'none']
                        
                        if formats_with_audio:
                            # Sort by audio bitrate
                            formats_with_audio.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                            best_format = formats_with_audio[0]
                            st.success(f"Found audio stream using {browser_attempt} cookies: {best_format.get('format_note', 'unknown quality')}")
                            return best_format.get('url')
                
                except Exception as e:
                    last_error = e
                    if "Sign in to confirm" in str(e):
                        st.warning(f"Cookie extraction from {browser_attempt} didn't work, trying next browser...")
                    continue
            
            # If all browsers failed, show the last error
            if last_error:
                st.error(f"Audio extraction failed. YouTube requires authentication. Error: {last_error}")
                st.info("💡 **Solutions:**\n"
                       "1. Make sure you're logged into YouTube in Chrome/Firefox\n"
                       "2. Try a different browser in settings\n"
                       "3. Export cookies manually (see yt-dlp documentation)")
            
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
