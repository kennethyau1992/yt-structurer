import os
import re
import json
import time
from typing import Optional, List, Dict, Tuple, Any
import requests
import streamlit as st

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

class AssemblyAITranscriptProvider(TranscriptProvider):
    """Fallback provider for ASR when no official captions available"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            st.warning("AssemblyAI API key not provided")
            return None
            
        try:
            st.info(f"🎤 Starting ASR transcription for: {url}")
            
            # Step 1: Get audio URL from video URL
            # AssemblyAI needs a direct audio URL, but we're passing a YouTube URL
            # This is a limitation - AssemblyAI can't directly process YouTube URLs
            # We need to extract audio first
            
            # For now, let's try the URL as-is and see what happens
            headers = {
                "authorization": self.api_key,
                "content-type": "application/json"
            }
            
            # Map our language selection to AssemblyAI language codes
            language_map = {
                "English": "en",
                "中文": "zh"
            }
            
            data = {
                "audio_url": url,  # This won't work with YouTube URLs!
                "language_code": language_map.get(language, "en")
            }
            
            st.info("📤 Submitting transcription request to AssemblyAI...")
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('error', 'Unknown error')
                st.error(f"❌ AssemblyAI submission error ({response.status_code}): {error_msg}")
                return None
                
            transcript_id = response.json()['id']
            st.info(f"✅ Transcript submitted with ID: {transcript_id}")
            
            # Step 2: Poll for completion
            polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
            max_attempts = 60  # 3 minutes max
            attempt = 0
            
            st.info("⏳ Polling for transcription completion...")
            
            while attempt < max_attempts:
                polling_response = requests.get(polling_endpoint, headers=headers, timeout=30)
                polling_response_data = polling_response.json()
                
                status = polling_response_data['status']
                st.info(f"📊 Status: {status} (attempt {attempt + 1}/{max_attempts})")
                
                if status == 'completed':
                    st.success("🎉 Transcription completed!")
                    return polling_response_data['text']
                elif status == 'error':
                    error_msg = polling_response_data.get('error', 'Unknown transcription error')
                    st.error(f"❌ AssemblyAI transcription error: {error_msg}")
                    return None
                else:
                    time.sleep(3)  # Wait before polling again
                    attempt += 1
            
            st.error("⏰ Transcription timed out")
            return None
                    
        except requests.exceptions.Timeout:
            st.error("🕐 Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error: {e}")
            return None
        except Exception as e:
            st.error(f"💥 AssemblyAI processing error: {e}")
            return None

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
    """Improved ASR provider that extracts audio URLs using yt-dlp"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.audio_extractor = YouTubeAudioExtractor()
    
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.api_key:
            st.warning("AssemblyAI API key not provided")
            return None
        
        # Step 1: Extract audio URL from YouTube video
        st.info("🎵 Step 1: Extracting audio URL from YouTube video...")
        audio_url = self.audio_extractor.extract_audio_url(url)
        
        if not audio_url:
            st.error("🚫 Cannot extract audio URL from YouTube video. ASR fallback failed.")
            return None
        
        # Step 2: Proceed with AssemblyAI transcription using the extracted audio URL
        st.info("🎤 Step 2: Starting AssemblyAI transcription...")
        return self._transcribe_audio_url(audio_url, language)
    
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
            st.error("🕐 Request timed out while submitting to AssemblyAI")
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
                st.error("🕐 Polling request timed out")
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
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
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
                    {"role": "user", "content": transcript},
                ],
                "temperature": self.temperature,
            }

            # Increased timeout for better reliability with longer transcripts
            resp = requests.post(
                endpoint, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=120  # Increased from 60 to 120 seconds
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
            raise RuntimeError("DeepSeek API request timed out after 120 seconds")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Failed to connect to DeepSeek API - check internet connection")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API network error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")

class YouTubeDataProvider:
    """Provider for YouTube Data API operations"""
    def __init__(self, api_key: str):
        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            self.available = True
        except ImportError:
            self.available = False
            self.youtube = None
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        """
        Extract video URLs from a YouTube playlist using YouTube Data API v3.
        Returns list of dicts with 'title', 'url', and 'video_id' keys.
        """
        if not self.available or not self.youtube:
            st.error("YouTube Data API client not available.")
            return []
            
        playlist_id = self.extract_playlist_id(playlist_url)
        if not playlist_id:
            st.error("Could not extract playlist ID from URL")
            return []
        
        try:
            videos = []
            next_page_token = None
            
            while True:
                # Get playlist items
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=playlist_id,
                    maxResults=50,  # Maximum allowed per request
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    title = item['snippet']['title']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    videos.append({
                        'title': title,
                        'url': video_url,
                        'video_id': video_id
                    })
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos
            
        except Exception as e:
            st.error(f"Error accessing YouTube API: {e}")
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
            st.info("🎤 No official captions found. Trying ASR fallback (AssemblyAI)...")
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

# ==================== UI LAYER ====================

class YouTubeTranscriptApp:
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
        
        # Language-specific system prompts
        self.ENGLISH_SYSTEM_PROMPT = (
            "You are a careful transcript formatter. Your job is to STRUCTURE (not rewrite) a raw English "
            "transcript into a clean outline. Do NOT invent or add content. Preserve the speaker's wording "
            "(lightly correct obvious typos, punctuation, spacing). Keep original order; remove filler noises "
            "(e.g., 'uh', 'um', repeated stutters) and timestamps. No commentary. Output in Markdown only.\n\n"
            "Required output shape:\n"
            "# <Concise Title>\n"
            "-- <Short subtitle (optional; if none present, write 'Transcript structured')>\n\n"
            "## [Section 1: <phrase taken from transcript>]\n"
            "### 1.1 <subheading from transcript>\n"
            "- Bullet: exact phrase or contiguous fragment from the transcript\n"
            "- Bullet: exact phrase or contiguous fragment from the transcript\n\n"
            "### 1.2 <subheading>\n"
            "- Bullet\n"
            "- Bullet\n\n"
            "## [Section 2: <phrase>]\n"
            "... and so on.\n\n"
            "Rules:\n"
            "- Do NOT summarize beyond light cleanup; do NOT add ideas; do NOT translate.\n"
            "- Use section/subheading names drawn from the transcript wording.\n"
            "- Keep bullets short, verbatim fragments where possible.\n"
        )

        self.CHINESE_SYSTEM_PROMPT = (
            "你是一个仔细的转录文稿格式化工具。你的任务是将原始中文转录文稿整理成清晰的大纲结构。不要发明或添加内容。"
            "保持说话者的措辞（轻微纠正明显的错别字、标点符号、间距）。保持原有顺序；删除填充词"
            "（如'呃'、'嗯'、重复的结巴）和时间戳。不要评论。仅输出Markdown格式。\n\n"
            "要求的输出形状：\n"
            "# <简洁标题>\n"
            "-- <短副标题（可选；如无，写'转录文稿已整理'）>\n\n"
            "## [第1部分：<取自转录文稿的短语>]\n"
            "### 1.1 <来自转录文稿的子标题>\n"
            "- 要点：转录文稿中的确切短语或连续片段\n"
            "- 要点：转录文稿中的确切短语或连续片段\n\n"
            "### 1.2 <子标题>\n"
            "- 要点\n"
            "- 要点\n\n"
            "## [第2部分：<短语>]\n"
            "... 依此类推。\n\n"
            "规则：\n"
            "- 除了轻微清理外，不要总结；不要添加想法；不要翻译。\n"
            "- 使用从转录文稿措辞中提取的章节/子标题名称。\n"
            "- 保持要点简短，尽可能使用逐字片段。\n"
        )

        self.SYSTEM_PROMPTS = {
            "English": self.ENGLISH_SYSTEM_PROMPT,
            "中文": self.CHINESE_SYSTEM_PROMPT
        }

        self.LANGUAGE_CODES = {
            "English": "en",
            "中文": "zh"
        }
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="YouTube Transcript Structurer", 
            page_icon="📺", 
            layout="centered"
        )
    
    def setup_session_state(self):
        if "transcript_text" not in st.session_state:
            st.session_state.transcript_text = None
        if "structured_md" not in st.session_state:
            st.session_state.structured_md = None
        if "playlist_transcripts" not in st.session_state:
            st.session_state.playlist_transcripts = None
        if "structured_playlist" not in st.session_state:
            st.session_state.structured_playlist = None
    
    def is_valid_youtube_url(self, url: str) -> bool:
        youtube_regex = re.compile(
            r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/).+",
            re.IGNORECASE,
        )
        return bool(youtube_regex.match((url or "").strip()))
    
    def is_valid_playlist_url(self, url: str) -> bool:
        playlist_regex = re.compile(
            r"^(https?://)?(www\.)?youtube\.com/playlist\?list=[\w-]+",
            re.IGNORECASE,
        )
        return bool(playlist_regex.match((url or "").strip()))
    
    def render_sidebar(self):
        with st.sidebar:
            st.subheader("API Keys & Model")
            supa_key = st.text_input("Supadata API Key", type="password", help="Required for transcript extraction")
            assemblyai_key = st.text_input("AssemblyAI API Key", type="password", help="Optional for ASR fallback")
            ds_key = st.text_input("DeepSeek API Key", type="password", help="Required for transcript structuring")
            youtube_key = st.text_input("YouTube Data API v3 Key", type="password", help="Required for playlist processing")
            
            # Add a checkbox for ASR fallback with updated help
            use_asr_fallback = st.checkbox(
                "Use ASR fallback", 
                value=False,  # Start with False until user confirms they have yt-dlp
                help="✅ Uses yt-dlp to extract audio URLs for AssemblyAI transcription"
            )
            
            # ASR Status indicator
            if use_asr_fallback and assemblyai_key:
                try:
                    import yt_dlp
                    st.success("🎉 ASR fallback ready - yt-dlp installed")
                except ImportError:
                    st.error("❌ ASR fallback enabled but yt-dlp not installed")
                    st.code("pip install yt-dlp")
            elif use_asr_fallback and not assemblyai_key:
                st.error("❌ ASR fallback enabled but no AssemblyAI key provided")
            
            with st.expander("📋 How to get API keys", expanded=False):
                st.markdown("""
                **Supadata API Key:**
                - Sign up at [Supadata](https://supadata.ai)
                
                **AssemblyAI API Key:**
                - Sign up at [AssemblyAI](https://www.assemblyai.com)
                - ✅ Now works with yt-dlp for audio extraction
                
                **DeepSeek API Key:**
                - Sign up at [DeepSeek](https://platform.deepseek.com)
                
                **YouTube Data API v3 Key:**
                1. Go to [Google Cloud Console](https://console.cloud.google.com)
                2. Create a new project or select existing
                3. Enable YouTube Data API v3
                4. Create credentials (API Key)
                5. Restrict the key to YouTube Data API v3
                """)

            st.divider()
            st.subheader("Language Settings")
            selected_language = st.selectbox(
                "Select Language",
                options=list(self.LANGUAGE_CODES.keys()),
                index=0
            )
            
            st.divider()
            st.subheader("LLM Settings")
            base_url = st.text_input("DeepSeek Base URL", value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
            model = st.text_input("DeepSeek Model", value=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
            
            return {
                "supa_key": supa_key,
                "assemblyai_key": assemblyai_key,
                "ds_key": ds_key,
                "youtube_key": youtube_key,
                "use_asr_fallback": use_asr_fallback,
                "selected_language": selected_language,
                "base_url": base_url,
                "model": model,
                "temperature": temperature
            }
    
    def process_playlist_transcripts(self, orchestrator, youtube_provider, playlist_url, language_code, use_asr_fallback):
        """
        Process all videos in a playlist and return their transcripts.
        Returns list of tuples: (video_title, video_url, transcript_text)
        """
        videos = youtube_provider.get_playlist_videos(playlist_url)
        if not videos:
            return []
        
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, video in enumerate(videos):
            status_text.text(f"Processing video {i+1}/{len(videos)}: {video['title'][:50]}...")
            
            transcript = orchestrator.get_transcript(video['url'], language_code, use_asr_fallback)
            results.append((video['title'], video['url'], transcript))
            
            progress_bar.progress((i + 1) / len(videos))
        
        status_text.text(f"Processed {len(videos)} videos!")
        return results
    
    def render_main_ui(self, sidebar_config):
        st.title("YouTube Transcript Structurer")
        st.caption("Enhanced with language selection, playlist support, and ASR fallback")

        # Remove the ASR status warning since it now works
        if sidebar_config["use_asr_fallback"]:
            try:
                import yt_dlp
                st.info("🎤 **ASR Fallback Enabled**: Will use AssemblyAI if no official captions are found.")
            except ImportError:
                st.error("❌ **ASR Fallback Error**: yt-dlp not installed. Run: `pip install yt-dlp`")

        st.markdown("### 1) Input Selection")

        # Add mode selection
        processing_mode = st.radio(
            "Choose processing mode:",
            ["Single Video", "Playlist"],
            horizontal=True
        )

        if processing_mode == "Single Video":
            url = st.text_input(
                "Paste the YouTube video URL", 
                value="", 
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fetch_clicked = st.button("Fetch Transcript", type="primary")
            with c2:
                structure_clicked = st.button("Structure Transcript")
                
        else:  # Playlist mode
            playlist_url = st.text_input(
                "Paste the YouTube playlist URL", 
                value="", 
                placeholder="https://www.youtube.com/playlist?list=..."
            )
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fetch_playlist_clicked = st.button("Fetch All Transcripts", type="primary")
            with c2:
                structure_playlist_clicked = st.button("Structure All Transcripts")

        # Initialize providers with improved ASR provider
        supa_provider = SupadataTranscriptProvider(sidebar_config["supa_key"])
        
        # Use the improved ASR provider that explains the limitation
        assemblyai_provider = ImprovedAssemblyAITranscriptProvider(sidebar_config["assemblyai_key"]) if sidebar_config["assemblyai_key"] else None
        
        deepseek_provider = DeepSeekProvider(
            sidebar_config["ds_key"], 
            sidebar_config["base_url"], 
            sidebar_config["model"], 
            sidebar_config["temperature"]
        )
        youtube_provider = YouTubeDataProvider(sidebar_config["youtube_key"])

        # Initialize orchestrator
        orchestrator = TranscriptOrchestrator(
            transcript_provider=supa_provider,
            asr_fallback_provider=assemblyai_provider if sidebar_config["use_asr_fallback"] else None,
            llm_provider=deepseek_provider
        )

        # Single video processing
        if processing_mode == "Single Video" and fetch_clicked:
            st.session_state.structured_md = None
            if not url or not self.is_valid_youtube_url(url):
                st.warning("Please enter a valid YouTube URL.")
            elif not sidebar_config["supa_key"]:
                st.warning("Please provide your Supadata API key in the sidebar.")
            else:
                language_code = self.LANGUAGE_CODES[sidebar_config["selected_language"]]
                with st.spinner(f"Fetching {sidebar_config['selected_language']} transcript..."):
                    text = orchestrator.get_transcript(url, language_code, sidebar_config["use_asr_fallback"])
                if not text:
                    st.info(f"No {sidebar_config['selected_language']} transcript is available for this video.")
                    st.session_state.transcript_text = None
                else:
                    st.success("Transcript fetched.")
                    st.session_state.transcript_text = text

        # Playlist processing
        if processing_mode == "Playlist" and fetch_playlist_clicked:
            st.session_state.playlist_transcripts = None
            st.session_state.structured_playlist = None
            
            if not playlist_url or not self.is_valid_playlist_url(playlist_url):
                st.warning("Please enter a valid YouTube playlist URL.")
            elif not sidebar_config["supa_key"]:
                st.warning("Please provide your Supadata API key in the sidebar.")
            elif not sidebar_config["youtube_key"]:
                st.warning("Please provide your YouTube Data API v3 key in the sidebar.")
            else:
                language_code = self.LANGUAGE_CODES[sidebar_config["selected_language"]]
                with st.spinner(f"Processing playlist for {sidebar_config['selected_language']} transcripts..."):
                    playlist_results = self.process_playlist_transcripts(
                        orchestrator, youtube_provider, playlist_url, language_code, sidebar_config["use_asr_fallback"]
                    )
                
                if playlist_results:
                    st.session_state.playlist_transcripts = playlist_results
                    successful_count = sum(1 for _, _, transcript in playlist_results if transcript)
                    st.success(f"Processed {len(playlist_results)} videos, {successful_count} with transcripts.")
                else:
                    st.info("No videos processed from playlist.")

        # Display single video transcript
        if processing_mode == "Single Video" and st.session_state.transcript_text:
            st.markdown("### 2) Transcript Preview")
            with st.expander("Show transcript", expanded=False):
                st.text_area("", st.session_state.transcript_text, height=240, key="single_transcript")

        # Display playlist transcripts
        if processing_mode == "Playlist" and st.session_state.playlist_transcripts:
            st.markdown("### 2) Playlist Transcripts Preview")
            
            for i, (title, video_url, transcript) in enumerate(st.session_state.playlist_transcripts):
                with st.expander(f"Video {i+1}: {title}", expanded=False):
                    st.write(f"**URL:** {video_url}")
                    if transcript:
                        st.text_area("Transcript:", transcript, height=200, key=f"playlist_transcript_{i}")
                    else:
                        st.info("No transcript available for this video")

        # Single video structuring
        if processing_mode == "Single Video" and structure_clicked:
            if not st.session_state.transcript_text:
                st.warning("Please fetch a transcript first.")
            elif not sidebar_config["ds_key"]:
                st.warning("Enter your DeepSeek API key in the sidebar to run structuring.")
            else:
                system_prompt = self.SYSTEM_PROMPTS[sidebar_config["selected_language"]]
                with st.spinner("Asking DeepSeek to structure..."):
                    md = orchestrator.structure_transcript(st.session_state.transcript_text, system_prompt)
                if md:
                    st.session_state.structured_md = md
                    st.success("Structured successfully.")
                else:
                    st.error("Failed to structure transcript.")

        # Playlist structuring
        if processing_mode == "Playlist" and structure_playlist_clicked:
            if not st.session_state.playlist_transcripts:
                st.warning("Please fetch playlist transcripts first.")
            elif not sidebar_config["ds_key"]:
                st.warning("Enter your DeepSeek API key in the sidebar to run structuring.")
            else:
                system_prompt = self.SYSTEM_PROMPTS[sidebar_config["selected_language"]]
                structured_results = []
                
                transcripts_to_process = [
                    (title, url, transcript) for title, url, transcript in st.session_state.playlist_transcripts 
                    if transcript
                ]
                
                if not transcripts_to_process:
                    st.warning("No transcripts available to structure.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each video individually with error isolation
                    success_count = 0
                    error_count = 0
                    
                    for i, (title, video_url, transcript) in enumerate(transcripts_to_process):
                        status_text.text(f"Structuring video {i+1}/{len(transcripts_to_process)}: {title[:50]}...")
                        
                        try:
                            # Individual video processing with timeout handling
                            with st.spinner(f"Processing: {title[:30]}..."):
                                structured_md = orchestrator.structure_transcript(transcript, system_prompt)
                            
                            if structured_md:
                                structured_results.append((title, video_url, structured_md, "success"))
                                success_count += 1
                                st.success(f"✅ Structured: {title[:50]}")
                            else:
                                structured_results.append((title, video_url, None, "failed"))
                                error_count += 1
                                st.error(f"❌ Failed to structure: {title[:50]} - No response from LLM")
                                
                        except Exception as e:
                            error_msg = str(e)
                            structured_results.append((title, video_url, None, f"error: {error_msg}"))
                            error_count += 1
                            st.error(f"💥 Error structuring '{title[:50]}': {error_msg}")
                            
                            # Continue processing other videos despite this error
                            continue
                        
                        # Update progress after each video (successful or failed)
                        progress_bar.progress((i + 1) / len(transcripts_to_process))
                        
                        # Brief pause to avoid overwhelming the API
                        time.sleep(0.5)
                    
                    # Final status update
                    status_text.text(f"Completed! ✅ {success_count} successful, ❌ {error_count} failed")
                    
                    # Always save results, even if some failed
                    st.session_state.structured_playlist = structured_results
                    
                    # Show summary
                    if success_count > 0:
                        st.success(f"🎉 Successfully structured {success_count} out of {len(transcripts_to_process)} transcripts!")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} videos failed to process but results for successful videos are preserved.")
                    
                    # Show detailed results breakdown
                    with st.expander("📊 Processing Results Summary", expanded=True):
                        for i, (title, url, structured_md, status) in enumerate(structured_results):
                            if status == "success":
                                st.write(f"✅ **Video {i+1}**: {title}")
                            else:
                                st.write(f"❌ **Video {i+1}**: {title} - Status: {status}")

        # Display single video structured output
        if processing_mode == "Single Video" and st.session_state.structured_md:
            st.markdown("### 3) Structured Output")
            st.markdown(st.session_state.structured_md)
            
            # Extract title from DeepSeek output for filename
            deepseek_title = self.extract_title_from_markdown(st.session_state.structured_md)
            base_filename = deepseek_title if deepseek_title else "structured_transcript"
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download .md",
                    data=st.session_state.structured_md,
                    file_name=f"{base_filename}.md",
                    mime="text/markdown",
                )
            with col2:
                st.download_button(
                    "Download .txt",
                    data=st.session_state.structured_md,
                    file_name=f"{base_filename}.txt",
                    mime="text/plain",
                )

        # Display playlist structured output
        if processing_mode == "Playlist" and st.session_state.structured_playlist:
            st.markdown("### 3) Structured Playlist Output")
            
            # Create combined markdown for all videos (including failed ones for documentation)
            combined_md = f"# Playlist Transcripts - {sidebar_config['selected_language']}\n\n"
            
            successful_videos = []
            failed_videos = []
            
            for i, result in enumerate(st.session_state.structured_playlist):
                if len(result) == 4:  # New format with status
                    title, video_url, structured_md, status = result
                else:  # Old format compatibility
                    title, video_url, structured_md = result
                    status = "success" if structured_md else "failed"
                
                if structured_md:
                    # Extract DeepSeek title for better display
                    deepseek_title = self.extract_title_from_markdown(structured_md)
                    display_title = deepseek_title if deepseek_title else title
                    successful_videos.append((i, title, video_url, structured_md, deepseek_title))
                else:
                    failed_videos.append((i, title, video_url, status))
            
            # Show successful videos
            if successful_videos:
                st.markdown(f"#### ✅ Successfully Structured Videos ({len(successful_videos)})")
                
                for i, title, video_url, structured_md, deepseek_title in successful_videos:
                    display_title = deepseek_title if deepseek_title else title
                    st.markdown(f"**Video {i+1}**: {display_title}")
                    if deepseek_title and deepseek_title != title:
                        st.caption(f"Original: {title}")
                    
                    with st.expander("Show structured content", expanded=False):
                        st.markdown(structured_md)
                    
                    # Add to combined markdown with better titles
                    combined_md += f"\n---\n\n## Video {i+1}: {display_title}\n"
                    if deepseek_title and deepseek_title != title:
                        combined_md += f"**Original Title:** {title}\n"
                    combined_md += f"**URL:** {video_url}\n"
                    combined_md += f"**Status:** ✅ Successfully structured\n\n"
                    combined_md += structured_md + "\n\n"
            
            # Show failed videos
            if failed_videos:
                st.markdown(f"#### ❌ Failed Videos ({len(failed_videos)})")
                
                for i, title, video_url, status in failed_videos:
                    with st.expander(f"❌ Video {i+1}: {title} (Failed)", expanded=False):
                        st.write(f"**URL:** {video_url}")
                        st.error(f"**Status:** {status}")
                        st.info("This video had a transcript but failed during structuring. You can try processing it individually.")
                    
                    # Add to combined markdown for documentation
                    combined_md += f"\n---\n\n## Video {i+1}: {title}\n"
                    combined_md += f"**URL:** {video_url}\n"
                    combined_md += f"**Status:** ❌ Failed - {status}\n\n"
                    combined_md += "*This video could not be structured but the transcript was available.*\n\n"
            
            # Download section - always show if there are any results
            if successful_videos or failed_videos:
                st.markdown("### 📥 Download Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download all (including failed for documentation)
                    st.download_button(
                        f"📄 Download All ({len(successful_videos + failed_videos)} videos)",
                        data=combined_md,
                        file_name="playlist_all_results.md",
                        mime="text/markdown",
                        help="Includes successful and failed videos for complete documentation"
                    )
                
                if successful_videos:
                    # Create successful-only markdown with better titles
                    success_only_md = f"# Playlist Transcripts - {sidebar_config['selected_language']} (Successful Only)\n\n"
                    for i, title, video_url, structured_md, deepseek_title in successful_videos:
                        deepseek_title = self.extract_title_from_markdown(structured_md)
                        display_title = deepseek_title if deepseek_title else title
                        success_only_md += f"\n---\n\n## Video {i+1}: {display_title}\n"
                        if deepseek_title and deepseek_title != title:
                            success_only_md += f"**Original Title:** {title}\n"
                        success_only_md += f"**URL:** {video_url}\n\n"
                        success_only_md += structured_md + "\n\n"
                    
                    with col2:
                        st.download_button(
                            f"✅ Download Successful Only ({len(successful_videos)} videos)",
                            data=success_only_md,
                            file_name="playlist_successful_structured_transcripts.md",
                            mime="text/markdown",
                            help="Only includes successfully structured videos with DeepSeek-generated titles"
                        )
                
                with col3:
                    # Individual video downloads with DeepSeek-generated titles
                    if successful_videos:
                        # Create options with both original title and DeepSeek title for clarity
                        video_options = []
                        for i, title, video_url, structured_md, deepseek_title in successful_videos:
                            display_name = f"Video {i+1}: {deepseek_title if deepseek_title else title[:30]}"
                            video_options.append((i, display_name, deepseek_title, title, structured_md, video_url))
                        
                        selected_video = st.selectbox(
                            "Download Individual Video:",
                            options=video_options,
                            format_func=lambda x: x[1]  # Display name
                        )
                        
                        if selected_video:
                            video_index, display_name, deepseek_title, original_title, structured_md, video_url = selected_video
                            
                            # Use DeepSeek title for filename, fallback to sanitized original title
                            filename_base = deepseek_title if deepseek_title else self.sanitize_filename(original_title[:30])
                            
                            individual_md = f"# {deepseek_title if deepseek_title else original_title}\n\n**URL:** {video_url}\n\n{structured_md}"
                            
                            st.download_button(
                                f"📄 Download: {deepseek_title if deepseek_title else 'Selected Video'}",
                                data=individual_md,
                                file_name=f"{filename_base}.md",
                                mime="text/markdown",
                                key=f"download_individual_{video_index}",
                                help=f"Filename: {filename_base}.md"
                            )

        st.divider()
        st.caption(
            f"Language: {sidebar_config['selected_language']} | Mode: {processing_mode} | "
            "Uses Supadata for transcripts, AssemblyAI for ASR fallback, DeepSeek for structuring, and YouTube Data API v3 for playlists."
        )

        # Installation and debugging instructions
        with st.expander("📦 Installation Requirements", expanded=False):
            st.markdown("""
            To use all features of this app, install the required packages:
            
            ```bash
            pip install streamlit supadata requests google-api-python-client yt-dlp
            ```
            
            **Package purposes:**
            - `supadata`: YouTube transcript extraction
            - `google-api-python-client`: YouTube Data API v3 for playlist processing
            - `requests`: DeepSeek and AssemblyAI API communication
            - `streamlit`: Web interface
            - `yt-dlp`: YouTube audio extraction for ASR fallback
            """)
        
        with st.expander("🔧 ASR Fallback Information", expanded=False):
            st.markdown("""
            ## ASR Fallback with yt-dlp ✅
            
            The ASR fallback now works properly using yt-dlp for audio extraction!
            
            ### How it works:
            1. **Primary**: Try to get official captions via Supadata
            2. **Fallback**: If no captions, extract audio URL using yt-dlp
            3. **Transcribe**: Send audio URL to AssemblyAI for speech recognition
            
            ### Requirements:
            ```bash
            pip install yt-dlp
            ```
            
            ### Features:
            - ✅ Extracts best quality audio streams from YouTube
            - ✅ Handles both audio-only and video+audio formats  
            - ✅ Real-time progress tracking during transcription
            - ✅ Detailed error messages and status updates
            - ✅ Automatic quality selection (highest bitrate first)
            
            ### Supported scenarios:
            - Videos without official captions/subtitles
            - Private videos (if you have access)
            - Live stream recordings
            - Multiple languages (English, Chinese)
            
            ### Note:
            ASR transcription takes longer than caption extraction (2-5 minutes depending on video length).
            """)
    
    
    def run(self):
        sidebar_config = self.render_sidebar()
        self.render_main_ui(sidebar_config)

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    app = YouTubeTranscriptApp()
    app.run()
