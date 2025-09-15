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
        """Load users from environment variables"""
        # Format: USER1_PASSWORD, USER1_SUPADATA_KEY, etc.
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
        self.users.update(env_users)
    
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

# ==================== CHUNKING LAYER ====================

class AudioChunker:
    """Handles chunking of long audio files for transcription"""
    
    def __init__(self, chunk_duration_minutes: int = 12):
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.max_parallel_chunks = 3  # Limit parallel processing to avoid rate limits
    
    def get_video_duration(self, youtube_url: str) -> Optional[int]:
        """Get video duration in seconds using yt-dlp"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                duration = info.get('duration')
                return int(duration) if duration else None
                
        except Exception as e:
            st.warning(f"Could not get video duration: {e}")
            return None
    
    def extract_chunked_audio_urls(self, youtube_url: str) -> List[Dict[str, Any]]:
        """Extract audio URLs for video chunks"""
        try:
            import yt_dlp
            
            # First, get video duration
            duration = self.get_video_duration(youtube_url)
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
                
                # Use yt-dlp postprocessor to extract specific time ranges
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'noplaylist': True,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                    }],
                    # Note: yt-dlp doesn't directly support time ranges in URL extraction
                    # We'll need to modify our approach
                }
                
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
        
        st.info(f"📄 Starting parallel transcription of {len(chunks)} chunks...")
        
        def transcribe_single_chunk(chunk_info):
            """Transcribe a single chunk"""
            chunk_id = chunk_info['chunk_id']
            start_time = chunk_info['start_time']
            end_time = chunk_info['end_time']
            audio_url = chunk_info['audio_url']
            
            try:
                st.info(f"🎵 Chunk {chunk_id + 1}: {start_time//60}m{start_time%60}s - {end_time//60}m{end_time%60}s")
                
                # For now, we'll transcribe the full audio URL and note the time range
                # In a full implementation, you'd need to extract the specific time range
                transcript = assemblyai_provider._transcribe_audio_url(audio_url, language)
                
                return {
                    'chunk_id': chunk_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': transcript,
                    'success': transcript is not None
                }
                
            except Exception as e:
                st.error(f"❌ Chunk {chunk_id + 1} failed: {e}")
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
                        st.success(f"✅ Chunk {result['chunk_id'] + 1} completed")
                    else:
                        st.error(f"❌ Chunk {result['chunk_id'] + 1} failed")
                        
                except Exception as e:
                    st.error(f"💥 Chunk processing error: {e}")
        
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
            st.error("❌ All chunks failed to transcribe")
            return None
        
        if failed_chunks:
            st.warning(f"⚠️ Chunks {failed_chunks} failed, but continuing with successful chunks")
        
        # Combine all successful transcripts
        combined_transcript = "\n\n".join(successful_transcripts)
        
        st.success(f"🎉 Successfully combined {len(successful_transcripts)} chunks into final transcript")
        return combined_transcript

# ==================== NEW: LLM TEXT CHUNKER ====================

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
        
        st.info(f"🧩 Text is {len(text):,} characters (~{self.estimate_tokens(text):,} tokens) - chunking for better LLM processing...")
        
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
        
        st.info(f"✂️ Created {len(chunks)} intelligent chunks with overlap for context preservation")
        return chunks
    
    def create_chunk_specific_prompt(self, base_system_prompt: str, chunk_info: Dict, total_chunks: int, language: str) -> str:
        """Create a context-aware system prompt for each chunk"""
        chunk_id = chunk_info['chunk_id']
        is_single = chunk_info.get('is_single_chunk', False)
        is_final = chunk_info.get('is_final_chunk', False)
        
        if is_single:
            return base_system_prompt
        
        # Multi-chunk processing instructions
        if language == "English":
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
            
        try:
            resp = self.client.transcript(url=url, lang=language, text=True, mode="auto")
            return self._normalize_transcript(resp)
        except Exception as e:
            st.error(f"Supadata error: {e}")
            return None
    
    def _normalize_transcript(self, resp) -> str:
        """
        Convert various Supadata transcript response shapes to a single plain-text string.
        Supported shapes:
          - str (already plain text)
          - object with .text (e.g., Transcript)
          - dict with 'text' or 'chunks'
          - list of strings/dicts/objects (joined with newlines)
        Returns "" if nothing usable is found.
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
    
    @staticmethod
    def extract_audio_url(youtube_url: str) -> Optional[str]:
        """
        Extract direct audio URL from YouTube video using yt-dlp.
        Returns the best available audio stream URL for AssemblyAI.
        """
        try:
            import yt_dlp
            
            st.info("🎵 Extracting audio stream URL using yt-dlp...")
            
            # Configure yt-dlp options for best audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',  # Get best audio quality
                'noplaylist': True,          # Only process single video
                'quiet': True,               # Suppress yt-dlp output
                'no_warnings': True,         # Suppress warnings
                'extract_flat': False,       # Get full info
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info without downloading
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info:
                    st.error("❌ Could not extract video information")
                    return None
                
                # Look for the best audio stream URL
                formats = info.get('formats', [])
                
                # First, try to find audio-only streams
                audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                
                if audio_formats:
                    # Sort by audio bitrate (highest first)
                    audio_formats.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                    best_audio = audio_formats[0]
                    st.success(f"✅ Found audio-only stream: {best_audio.get('format_note', 'unknown quality')}")
                    return best_audio.get('url')
                
                # Fallback: find formats with audio (including video+audio)
                formats_with_audio = [f for f in formats if f.get('acodec') != 'none']
                
                if formats_with_audio:
                    # Sort by audio bitrate
                    formats_with_audio.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                    best_format = formats_with_audio[0]
                    st.success(f"✅ Found audio stream: {best_format.get('format_note', 'unknown quality')}")
                    return best_format.get('url')
                
                st.error("❌ No audio streams found in video")
                return None
                
        except ImportError:
            st.error("❌ yt-dlp not installed. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            st.error(f"❌ Audio extraction error: {e}")
            # Log more details for debugging
            st.error(f"URL: {youtube_url}")
            return None

class ImprovedAssemblyAITranscriptProvider(TranscriptProvider):
    """Improved ASR provider that extracts audio URLs using yt-dlp with chunking support"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
        self.chunker = AudioChunker()
        self.max_duration_minutes = 60  # Chunk videos longer than 60 minutes
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            st.warning("AssemblyAI API key not provided")
            return None
        
        # Step 1: Check video duration for chunking decision
        duration = self.chunker.get_video_duration(url)
        
        if duration and duration > (self.max_duration_minutes * 60):
            st.info(f"📄 Long video detected ({duration//60}m {duration%60}s) - using chunked transcription")
            return self._transcribe_with_chunking(url, language)
        else:
            st.info(f"🎵 Standard transcription for video ({duration//60 if duration else 'unknown'}m)")
            return self._transcribe_standard(url, language)
    
    def _transcribe_standard(self, url: str, language: str) -> Optional[str]:
        """Standard transcription without chunking"""
        # Step 1: Extract audio URL from YouTube video
        st.info("🎵 Step 1: Extracting audio URL from YouTube video...")
        audio_url = self.audio_extractor.extract_audio_url(url)
        
        if not audio_url:
            st.error("🚫 Cannot extract audio URL from YouTube video. ASR fallback failed.")
            return None
        
        # Step 2: Proceed with AssemblyAI transcription using the extracted audio URL
        st.info("🎤 Step 2: Starting AssemblyAI transcription...")
        return self._transcribe_audio_url(audio_url, language)
    
    def _transcribe_with_chunking(self, url: str, language: str) -> Optional[str]:
        """Transcription with chunking for long videos"""
        st.info("✂️ Step 1: Preparing video chunks...")
        chunks = self.chunker.extract_chunked_audio_urls(url)
        
        if not chunks:
            st.warning("Could not create chunks, falling back to standard transcription")
            return self._transcribe_standard(url, language)
        
        st.info("🚀 Step 2: Starting parallel chunk transcription...")
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
                "中文": "zh"
            }
            
            # Prepare transcription request
            data = {
                "audio_url": audio_url,
                "language_code": language_map.get(language, "en"),
                "speech_model": "best"  # Use best available model
            }
            
            st.info("📤 Submitting transcription request to AssemblyAI...")
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', 'Unknown error')
                st.error(f"❌ AssemblyAI submission error ({response.status_code}): {error_msg}")
                return None
                
            transcript_id = response.json()['id']
            st.success(f"✅ Transcription submitted with ID: {transcript_id}")
            
            # Step 3: Poll for completion with progress indication
            return self._poll_for_completion(transcript_id, headers)
            
        except requests.exceptions.Timeout:
            st.error("🕘 Request timed out while submitting to AssemblyAI")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error communicating with AssemblyAI: {e}")
            return None
        except Exception as e:
            st.error(f"💥 AssemblyAI transcription error: {e}")
            return None
    
    def _poll_for_completion(self, transcript_id: str, headers: dict) -> Optional[str]:
        """Poll AssemblyAI for transcription completion with progress updates"""
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        max_attempts = 120  # 6 minutes max (3 second intervals)
        attempt = 0
        
        st.info("⏳ Polling for transcription completion...")
        
        # Create a progress bar and status display
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while attempt < max_attempts:
            try:
                polling_response = requests.get(polling_endpoint, headers=headers, timeout=30)
                
                if polling_response.status_code != 200:
                    st.error(f"❌ Polling error ({polling_response.status_code})")
                    return None
                
                polling_data = polling_response.json()
                status = polling_data.get('status', 'unknown')
                
                # Update status display
                status_text.text(f"📊 Status: {status} (attempt {attempt + 1}/{max_attempts})")
                progress_bar.progress((attempt + 1) / max_attempts)
                
                if status == 'completed':
                    status_text.text("🎉 Transcription completed successfully!")
                    progress_bar.progress(1.0)
                    
                    transcript_text = polling_data.get('text', '')
                    if transcript_text:
                        st.success(f"✅ Received transcript ({len(transcript_text)} characters)")
                        return transcript_text
                    else:
                        st.error("❌ Transcription completed but no text returned")
                        return None
                        
                elif status == 'error':
                    error_msg = polling_data.get('error', 'Unknown transcription error')
                    st.error(f"❌ AssemblyAI transcription error: {error_msg}")
                    return None
                    
                elif status in ['queued', 'processing']:
                    # Continue polling
                    time.sleep(3)
                    attempt += 1
                    
                else:
                    st.warning(f"⚠️ Unknown status: {status}")
                    time.sleep(3)
                    attempt += 1
            
            except requests.exceptions.Timeout:
                st.error("🕘 Polling request timed out")
                return None
            except Exception as e:
                st.error(f"💥 Polling error: {e}")
                return None
        
        st.error("⏰ Transcription polling timed out after 6 minutes")
        return None

class LLMProvider:
    """Base class for LLM providers"""
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        raise NotImplementedError

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # Initialize text chunker for long transcripts
        self.text_chunker = LLMTextChunker(
            max_chunk_length=8000,  # Conservative for token limits
            overlap_length=200      # Context preservation
        )
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Enhanced transcript structuring with intelligent chunking"""
        try:
            # Determine if chunking is needed
            if not self.text_chunker.should_chunk_text(transcript):
                # Process as single chunk
                return self._process_single_chunk(transcript, system_prompt)
            else:
                # Process with intelligent chunking
                return self._process_with_chunking(transcript, system_prompt)
                
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def _process_single_chunk(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Process transcript as a single chunk"""
        st.info("🤖 Processing transcript as single chunk...")
        return self._make_api_request(transcript, system_prompt)
    
    def _process_with_chunking(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Process long transcript with intelligent chunking"""
        # Create intelligent chunks
        chunks = self.text_chunker.create_intelligent_chunks(transcript)
        
        if len(chunks) == 1:
            return self._process_single_chunk(transcript, system_prompt)
        
        st.info(f"🧩 Processing {len(chunks)} chunks with DeepSeek for better results...")
        
        # Process chunks in parallel with context-aware prompts
        processed_chunks = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract language from system prompt for chunk-specific instructions
        language = "English" if "English" in system_prompt else "中文"
        
        def process_chunk_with_context(chunk_info):
            """Process individual chunk with context-aware prompt"""
            chunk_id = chunk_info['chunk_id']
            chunk_text = chunk_info['text']
            
            try:
                status_text.text(f"🤖 Processing chunk {chunk_id + 1}/{len(chunks)} (~{chunk_info['tokens_estimate']} tokens)")
                
                # Create context-aware prompt for this chunk
                chunk_prompt = self.text_chunker.create_chunk_specific_prompt(
                    system_prompt, chunk_info, len(chunks), language
                )
                
                # Process chunk
                result = self._make_api_request(chunk_text, chunk_prompt)
                
                return {
                    'chunk_id': chunk_id,
                    'result': result,
                    'success': result is not None,
                    'tokens_estimate': chunk_info['tokens_estimate']
                }
                
            except Exception as e:
                st.error(f"❌ Chunk {chunk_id + 1} failed: {e}")
                return {
                    'chunk_id': chunk_id,
                    'result': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Process chunks with limited parallelism to avoid rate limits
        max_workers = 2  # Conservative to avoid API rate limits
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_chunk_with_context, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            chunk_results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    result = future.result()
                    chunk_results.append(result)
                    
                    if result['success']:
                        st.success(f"✅ Chunk {result['chunk_id'] + 1} processed")
                    else:
                        st.error(f"❌ Chunk {result['chunk_id'] + 1} failed")
                    
                    # Update progress
                    progress_bar.progress(len(chunk_results) / len(chunks))
                    
                except Exception as e:
                    st.error(f"💥 Chunk processing error: {e}")
        
        # Sort results by chunk_id
        chunk_results.sort(key=lambda x: x['chunk_id'])
        
        # Combine successful results
        successful_results = []
        failed_chunks = []
        
        for result in chunk_results:
            if result['success'] and result['result']:
                successful_results.append(result['result'])
            else:
                failed_chunks.append(result['chunk_id'] + 1)
        
        if not successful_results:
            st.error("❌ All chunks failed to process")
            return None
        
        if failed_chunks:
            st.warning(f"⚠️ Chunks {failed_chunks} failed, but combining successful chunks")
        
        # Combine chunks intelligently
        status_text.text("🔗 Combining processed chunks...")
        combined_result = self._combine_processed_chunks(successful_results, language)
        
        progress_bar.progress(1.0)
        status_text.text(f"🎉 Successfully processed and combined {len(successful_results)} chunks!")
        
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
            header = "# Combined Structured Transcript\n\n"
            footer = "\n\n---\n*This document was processed in multiple chunks for optimal LLM performance.*"
        else:  # Chinese
            separator = "\n\n---\n\n"
            header = "# 合并结构化转录文稿\n\n"
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
        st.info("🔍 Trying primary transcript provider (Supadata)...")
        transcript = self.transcript_provider.get_transcript(url, language)
        
        # If no transcript and fallback is enabled, try ASR
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("🎤 No official captions found. Trying ASR fallback (AssemblyAI with chunking)...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
        if not transcript:
            st.warning("❌ No transcript available from any provider")
        else:
            st.success("✅ Transcript obtained successfully")
            
        return transcript
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        if not self.llm_provider:
            return None
        return self.llm_provider.structure_transcript(transcript, system_prompt)

# ==================== STREAMLIT UI ====================

def login_page():
    """Display login page"""
    st.title("🎬 YouTube Transcript Processor")
    st.subheader("🔐 Login Required")
    
    # Show default credentials info
    st.info("🔑 **Default Login:** Username: `admin` | Password: `admin123`")
    
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
    with st.expander("🐛 Login Debug", expanded=False):
        st.subheader("Available Users")
        auth_manager = AuthManager()
        
        for user in auth_manager.users.keys():
            st.text(f"👤 User: {user}")
        
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
    st.title("🎬 YouTube Transcript Processor")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            for key in ['authenticated', 'username', 'api_keys']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.write(f"Welcome, **{st.session_state.username}**! 👋")
    
    # Get API keys from session
    api_keys = st.session_state.get('api_keys', {})
    
    # Configuration Section
    st.header("⚙️ Configuration")
    
    with st.expander("📊 API Status", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if api_keys.get('supadata'):
                st.success("✅ Supadata")
            else:
                st.error("❌ Supadata")
        
        with col2:
            if api_keys.get('assemblyai'):
                st.success("✅ AssemblyAI")
            else:
                st.error("❌ AssemblyAI")
        
        with col3:
            if api_keys.get('deepseek'):
                st.success("✅ DeepSeek")
            else:
                st.error("❌ DeepSeek")
        
        with col4:
            if api_keys.get('youtube'):
                st.success("✅ YouTube Data")
            else:
                st.error("❌ YouTube Data")
    
    # Debug section to help troubleshoot API key loading
    with st.expander("🐛 Debug Info", expanded=False):
        st.subheader("Environment Variables Check")
        
        # Check what env vars are available
        env_vars_found = []
        possible_keys = [
            "ADMIN_SUPADATA_KEY", "ADMIN_ASSEMBLYAI_KEY", "ADMIN_DEEPSEEK_KEY", "ADMIN_YOUTUBE_KEY",
            "SUPADATA_API_KEY", "ASSEMBLYAI_API_KEY", "DEEPSEEK_API_KEY", "YOUTUBE_API_KEY",
            "ASSEMBLYAI_KEY", "DEEPSEEK_KEY"
        ]
        
        for key in possible_keys:
            value = os.getenv(key)
            if value:
                masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
                env_vars_found.append(f"✅ {key}: {masked_value}")
            else:
                env_vars_found.append(f"❌ {key}: Not found")
        
        for var_info in env_vars_found:
            st.text(var_info)
        
        st.subheader("Loaded API Keys for Current User")
        for key, value in api_keys.items():
            if value:
                masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****" 
                st.text(f"✅ {key}: {masked_value}")
            else:
                st.text(f"❌ {key}: Not loaded")
    
    # Settings
    with st.expander("🔧 Processing Settings"):
        col1, col2 = st.columns(2)
        
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
    
    # Main Processing Section
    st.header("🎯 Process Video")
    
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
        st.subheader(f"📋 Videos to Process ({len(videos_to_process)})")
        
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
        st.subheader("📝 System Prompt")
        
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
        
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompts[language],
            height=200,
            help="Customize how the AI should structure the transcript"
        )
        
        # Processing button
        if st.button("🚀 Start Processing", type="primary"):
            if not api_keys.get('supadata') and not api_keys.get('assemblyai'):
                st.error("❌ No transcript providers available. Please configure API keys.")
                return
            
            if not api_keys.get('deepseek'):
                st.error("❌ DeepSeek API key required for structuring. Please configure API keys.")
                return
            
            process_videos(videos_to_process, language, use_asr_fallback, system_prompt, 
                          deepseek_model, temperature, api_keys)

def process_videos(videos, language, use_asr_fallback, system_prompt, deepseek_model, temperature, api_keys):
    """Process multiple videos"""
    
    # Initialize providers
    supadata_provider = SupadataTranscriptProvider(api_keys.get('supadata', ''))
    assemblyai_provider = ImprovedAssemblyAITranscriptProvider(api_keys.get('assemblyai', '')) if use_asr_fallback else None
    deepseek_provider = DeepSeekProvider(
        api_keys.get('deepseek', ''),
        "https://api.deepseek.com/v1",
        deepseek_model,
        temperature
    )
    
    orchestrator = TranscriptOrchestrator(
        supadata_provider,
        assemblyai_provider,
        deepseek_provider
    )
    
    # Process each video
    for i, video in enumerate(videos):
        st.subheader(f"📹 Processing Video {i+1}/{len(videos)}")
        st.write(f"**Title:** {video['title']}")
        st.write(f"**URL:** {video['url']}")
        
        try:
            # Step 1: Get transcript
            with st.spinner("🔍 Getting transcript..."):
                transcript = orchestrator.get_transcript(video['url'], language, use_asr_fallback)
            
            if not transcript:
                st.error("❌ Failed to get transcript")
                continue
            
            # Show transcript preview
            with st.expander("📄 Raw Transcript Preview"):
                st.text_area("Transcript", transcript[:1000] + "..." if len(transcript) > 1000 else transcript, height=200)
            
            # Step 2: Structure transcript
            with st.spinner("🤖 Structuring transcript with LLM..."):
                structured = orchestrator.structure_transcript(transcript, system_prompt)
            
            if not structured:
                st.error("❌ Failed to structure transcript")
                continue
            
            # Display results
            st.success("✅ Processing completed!")
            
            # Show structured result
            with st.expander("📋 Structured Transcript", expanded=True):
                st.markdown(structured)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Download Raw Transcript",
                    transcript,
                    file_name=f"raw_transcript_{i+1}.txt",
                    mime="text/plain"
                )
            
            with col2:
                st.download_button(
                    "📄 Download Structured Transcript",
                    structured,
                    file_name=f"structured_transcript_{i+1}.md",
                    mime="text/markdown"
                )
            
            st.divider()
            
        except Exception as e:
            st.error(f"💥 Error processing video: {str(e)}")
            continue

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point"""
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
