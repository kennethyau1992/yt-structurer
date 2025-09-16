#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Transcript Processor with Executive Summary and Persistent Data
Complete application with all improvements implemented
"""

import os
import re
import json
import time
import hashlib
import pickle
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import requests
import streamlit as st

# ==================== PERSISTENCE LAYER ====================

class UserDataManager:
    """Manages user settings and history persistence"""
    
    def __init__(self, username: str):
        self.username = username
        self.data_dir = Path("user_data")
        self.data_dir.mkdir(exist_ok=True)
        self.user_file = self.data_dir / f"{username}_data.pkl"
        self.load_data()
    
    def load_data(self):
        """Load user data from file"""
        if self.user_file.exists():
            try:
                with open(self.user_file, 'rb') as f:
                    self.data = pickle.load(f)
            except:
                self.data = self.get_default_data()
        else:
            self.data = self.get_default_data()
    
    def get_default_data(self):
        """Get default user data structure"""
        return {
            'settings': {
                'language': 'English',
                'use_asr_fallback': True,
                'deepseek_model': 'deepseek-chat',
                'temperature': 0.1,
                'browser_for_cookies': 'none',
                'system_prompt': None
            },
            'history': [],
            'api_keys': {}
        }
    
    def save_data(self):
        """Save user data to file"""
        try:
            with open(self.user_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            st.warning(f"Could not save user data: {e}")
    
    def get_settings(self):
        """Get user settings"""
        return self.data.get('settings', {})
    
    def update_settings(self, settings: dict):
        """Update user settings"""
        self.data['settings'].update(settings)
        self.save_data()
    
    def add_to_history(self, entry: dict):
        """Add entry to history"""
        self.data['history'].insert(0, entry)
        self.data['history'] = self.data['history'][:50]
        self.save_data()
    
    def get_history(self):
        """Get user history"""
        return self.data.get('history', [])

# ==================== AUTHENTICATION LAYER ====================

class AuthManager:
    """Simple authentication manager for storing user credentials"""
    
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
        self._load_users_from_env()
    
    def _hash_password(self, password: str) -> str:
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users_from_env(self):
        """Load users from environment variables"""
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
        """Authenticate user credentials"""
        if username in self.users:
            return self.users[username]["password_hash"] == self._hash_password(password)
        return False
    
    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        """Get API keys for authenticated user"""
        if username in self.users:
            return self.users[username]["api_keys"]
        return {}

# ==================== YOUTUBE COOKIE MANAGER ====================

class YouTubeCookieManager:
    """Manages YouTube cookies for yt-dlp to bypass bot detection"""
    
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
                cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
                if cookies_file and os.path.exists(cookies_file):
                    base_opts['cookiefile'] = cookies_file
                else:
                    import platform
                    if platform.system() != 'Linux' and browser != 'none':
                        base_opts['cookiesfrombrowser'] = (browser,)
        
        base_opts['headers'] = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        return base_opts

# ==================== CHUNKING LAYER ====================

class AudioChunker:
    """Handles chunking of long audio files for transcription"""
    
    def __init__(self, chunk_duration_minutes: int = 12):
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.max_parallel_chunks = 3
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
            
            duration = self.get_video_duration(youtube_url, browser)
            if not duration:
                st.error("Could not determine video duration for chunking")
                return []
            
            if duration <= self.chunk_duration_seconds:
                st.info(f"Video is {duration//60}m {duration%60}s - no chunking needed")
                return []
            
            st.info(f"Video is {duration//60}m {duration%60}s - will chunk into {self.chunk_duration_seconds//60}-minute segments")
            
            num_chunks = (duration + self.chunk_duration_seconds - 1) // self.chunk_duration_seconds
            
            chunks = []
            for i in range(num_chunks):
                start_time = i * self.chunk_duration_seconds
                end_time = min((i + 1) * self.chunk_duration_seconds, duration)
                
                ydl_opts = self.cookie_manager.get_ydl_opts(use_cookies=True, browser=browser)
                ydl_opts.update({
                    'format': 'bestaudio/best',
                    'noplaylist': True,
                })
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    formats = info.get('formats', [])
                    
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
            chunk_id = chunk_info['chunk_id']
            start_time = chunk_info['start_time']
            end_time = chunk_info['end_time']
            audio_url = chunk_info['audio_url']
            
            try:
                st.info(f"Chunk {chunk_id + 1}: {start_time//60}m{start_time%60}s - {end_time//60}m{end_time%60}s")
                
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
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
            future_to_chunk = {
                executor.submit(transcribe_single_chunk, chunk): chunk 
                for chunk in chunks
            }
            
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
        
        results.sort(key=lambda x: x['chunk_id'])
        
        successful_transcripts = []
        failed_chunks = []
        
        for result in results:
            if result['success'] and result['transcript']:
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
        
        combined_transcript = "\n\n".join(successful_transcripts)
        
        st.success(f"Successfully combined {len(successful_transcripts)} chunks into final transcript")
        return combined_transcript

# ==================== LLM TEXT CHUNKER ====================

class LLMTextChunker:
    """Handles chunking of long transcripts for LLM processing"""
    
    def __init__(self, max_chunk_length: int = 8000, overlap_length: int = 200):
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
        self.min_chunk_length = 1000
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4
    
    def should_chunk_text(self, text: str) -> bool:
        """Determine if text needs chunking"""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > 6000
    
    def find_split_points(self, text: str) -> List[int]:
        """Find intelligent split points in text"""
        split_points = []
        
        for match in re.finditer(r'\n\s*\n', text):
            split_points.append(match.end())
        
        sentence_endings = r'[.!?]\s+'
        for match in re.finditer(sentence_endings, text):
            split_points.append(match.end())
        
        for match in re.finditer(r'\n', text):
            split_points.append(match.end())
        
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
            target_end = current_start + self.max_chunk_length
            
            if target_end >= len(text):
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
            
            best_split = target_end
            for split_point in reversed(split_points):
                if current_start + self.min_chunk_length <= split_point <= target_end:
                    best_split = split_point
                    break
            
            chunk_text = text[current_start:best_split]
            chunks.append({
                'chunk_id': chunk_id,
                'start_pos': current_start,
                'end_pos': best_split,
                'text': chunk_text,
                'tokens_estimate': self.estimate_tokens(chunk_text),
                'is_final_chunk': False
            })
            
            next_start = max(current_start + self.min_chunk_length, best_split - self.overlap_length)
            current_start = next_start
            chunk_id += 1
        
        st.info(f"Created {len(chunks)} intelligent chunks with overlap for context preservation")
        return chunks

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
        """Convert various Supadata transcript response shapes to plain text"""
        if resp is None:
            return ""
        
        resp_str = str(resp)
        if "BatchJob" in resp_str or "job_id" in resp_str:
            st.warning("Supadata returned a BatchJob - transcript may be processing or unavailable")
            return ""

        if isinstance(resp, str):
            if "BatchJob" in resp or "job_id" in resp:
                return ""
            return resp

        if isinstance(resp, dict):
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

        v = getattr(resp, "text", None)
        if isinstance(v, str):
            return v

        try:
            result_str = str(resp)
            if "BatchJob" not in result_str and "job_id" not in result_str:
                return result_str
            return ""
        except Exception:
            return ""

class YouTubeAudioExtractor:
    """Helper class to extract audio URLs from YouTube videos"""
    
    def __init__(self):
        self.cookie_manager = YouTubeCookieManager()
    
    def extract_audio_url(self, youtube_url: str, browser: str = 'chrome') -> Optional[str]:
        """Extract direct audio URL from YouTube video"""
        try:
            import yt_dlp
            
            st.info(f"Extracting audio stream URL using yt-dlp...")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            }
            
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            cookies_file = None
            
            if os.path.exists(render_cookies):
                st.success(f"Found Render secret cookies file!")
                cookies_file = render_cookies
            else:
                env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
                if env_cookies and os.path.exists(env_cookies):
                    st.info(f"Using cookies file from environment variable")
                    cookies_file = env_cookies
            
            if cookies_file:
                ydl_opts['cookiefile'] = cookies_file
                st.info(f"Using cookies for authentication")
            else:
                st.warning("No cookies file found - attempting without authentication")
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    
                    if info:
                        formats = info.get('formats', [])
                        
                        audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
                        
                        if audio_formats:
                            audio_formats.sort(key=lambda x: x.get('abr', 0) or 0, reverse=True)
                            best_audio = audio_formats[0]
                            st.success(f"Found audio-only stream: {best_audio.get('format_note', 'unknown quality')}")
                            return best_audio.get('url')
                        
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
                    st.info("Please update your youtube_cookies.txt file")
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
    """ASR provider with chunking support"""
    def __init__(self, api_key: str, browser: str = 'chrome'):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
        self.chunker = AudioChunker()
        self.max_duration_minutes = 60
        self.browser = browser
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            st.warning("AssemblyAI API key not provided")
            return None
        
        duration = self.chunker.get_video_duration(url, self.browser)
        
        if duration and duration > (self.max_duration_minutes * 60):
            st.info(f"Long video detected ({duration//60}m {duration%60}s) - using chunked transcription")
            return self._transcribe_with_chunking(url, language)
        else:
            st.info(f"Standard transcription for video ({duration//60 if duration else 'unknown'}m)")
            return self._transcribe_standard(url, language)
    
    def _transcribe_standard(self, url: str, language: str) -> Optional[str]:
        """Standard transcription without chunking"""
        st.info("Step 1: Extracting audio URL from YouTube video...")
        audio_url = self.audio_extractor.extract_audio_url(url, self.browser)
        
        if not audio_url:
            st.error("Cannot extract audio URL from YouTube video. ASR fallback failed.")
            return None
        
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
            
            language_map = {
                "English": "en",
                "中文": "zh",
                "en": "en",
                "zh": "zh"
            }
            
            assemblyai_lang = language_map.get(language, "en")
            
            data = {
                "audio_url": audio_url,
                "language_code": assemblyai_lang,
                "speech_model": "best"
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
            
            return self._poll_for_completion(transcript_id, headers)
            
        except Exception as e:
            st.error(f"AssemblyAI transcription error: {e}")
            return None
    
    def _poll_for_completion(self, transcript_id: str, headers: dict) -> Optional[str]:
        """Poll AssemblyAI for transcription completion"""
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        max_attempts = 120
        attempt = 0
        
        st.info("Polling for transcription completion...")
        
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
                
                status_text.text(f"Status: {status} (attempt {attempt + 1}/{max_attempts})")
                progress_bar.progress((attempt + 1) / max_attempts)
                
                if status == 'completed':
                    status_text.text("Transcription completed successfully!")
                    progress_bar.progress(1.0)
                    
                    transcript_text = polling_data.get('text', '')
                    if transcript_text:
                        st.success
