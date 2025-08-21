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
        try:
            # Submit transcription request
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
                "audio_url": url,
                "language_code": language_map.get(language, "en")
            }
            
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers
            )
            
            if response.status_code != 200:
                st.error(f"AssemblyAI submission error: {response.json().get('error')}")
                return None
                
            transcript_id = response.json()['id']
            
            # Poll for completion
            polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
            
            while True:
                polling_response = requests.get(polling_endpoint, headers=headers)
                polling_response_data = polling_response.json()
                
                status = polling_response_data['status']
                
                if status == 'completed':
                    return polling_response_data['text']
                elif status == 'error':
                    st.error(f"AssemblyAI transcription error: {polling_response_data.get('error')}")
                    return None
                else:
                    time.sleep(3)  # Wait before polling again
                    
        except Exception as e:
            st.error(f"AssemblyAI processing error: {e}")
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

            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")
            
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.error(f"DeepSeek error: {e}")
            return None

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
        transcript = self.transcript_provider.get_transcript(url, language)
        
        # If no transcript and fallback is enabled, try ASR
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("No official captions found. Trying ASR fallback...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
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
            
            # Add a checkbox for ASR fallback
            use_asr_fallback = st.checkbox("Use ASR fallback", value=True, 
                                          help="Use AssemblyAI if no official captions are available")
            
            with st.expander("📋 How to get API keys", expanded=False):
                st.markdown("""
                **Supadata API Key:**
                - Sign up at [Supadata](https://supadata.ai)
                
                **AssemblyAI API Key:**
                - Sign up at [AssemblyAI](https://www.assemblyai.com)
                
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
    
    def process_playlist_transcripts(self, orchestrator, youtube_provider, playlist_url, language_code):
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
            
            transcript = orchestrator.get_transcript(video['url'], language_code, True)
            results.append((video['title'], video['url'], transcript))
            
            progress_bar.progress((i + 1) / len(videos))
        
        status_text.text(f"Processed {len(videos)} videos!")
        return results
    
    def render_main_ui(self, sidebar_config):
        st.title("YouTube Transcript Structurer")
        st.caption("Enhanced with language selection, playlist support, and ASR fallback")

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

        # Initialize providers
        supa_provider = SupadataTranscriptProvider(sidebar_config["supa_key"])
        assemblyai_provider = AssemblyAITranscriptProvider(sidebar_config["assemblyai_key"]) if sidebar_config["assemblyai_key"] else None
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
                        orchestrator, youtube_provider, playlist_url, language_code
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
                    
                    for i, (title, video_url, transcript) in enumerate(transcripts_to_process):
                        status_text.text(f"Structuring video {i+1}/{len(transcripts_to_process)}: {title[:50]}...")
                        
                        try:
                            structured_md = orchestrator.structure_transcript(transcript, system_prompt)
                            structured_results.append((title, video_url, structured_md))
                        except Exception as e:
                            st.error(f"Error structuring '{title}': {e}")
                            structured_results.append((title, video_url, None))
                        
                        progress_bar.progress((i + 1) / len(transcripts_to_process))
                    
                    status_text.text("All transcripts structured!")
                    st.session_state.structured_playlist = structured_results
                    st.success(f"Structured {len([r for r in structured_results if r[2]])} transcripts successfully.")

        # Display single video structured output
        if processing_mode == "Single Video" and st.session_state.structured_md:
            st.markdown("### 3) Structured Output")
            st.markdown(st.session_state.structured_md)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download .md",
                    data=st.session_state.structured_md,
                    file_name="structured_transcript.md",
                    mime="text/markdown",
                )
            with col2:
                st.download_button(
                    "Download .txt",
                    data=st.session_state.structured_md,
                    file_name="structured_transcript.txt",
                    mime="text/plain",
                )

        # Display playlist structured output
        if processing_mode == "Playlist" and st.session_state.structured_playlist:
            st.markdown("### 3) Structured Playlist Output")
            
            # Create combined markdown for all videos
            combined_md = f"# Playlist Transcripts - {sidebar_config['selected_language']}\n\n"
            
            for i, (title, video_url, structured_md) in enumerate(st.session_state.structured_playlist):
                st.markdown(f"#### Video {i+1}: {title}")
                if structured_md:
                    with st.expander("Show structured content", expanded=False):
                        st.markdown(structured_md)
                    
                    # Add to combined markdown
                    combined_md += f"\n---\n\n## Video {i+1}: {title}\n"
                    combined_md += f"**URL:** {video_url}\n\n"
                    combined_md += structured_md + "\n\n"
                else:
                    st.info("Structuring failed for this video")
            
            if any(result[2] for result in st.session_state.structured_playlist):
                st.markdown("### Download All")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Combined .md",
                        data=combined_md,
                        file_name="playlist_structured_transcripts.md",
                        mime="text/markdown",
                    )
                with col2:
                    st.download_button(
                        "Download Combined .txt",
                        data=combined_md,
                        file_name="playlist_structured_transcripts.txt",
                        mime="text/plain",
                    )

        st.divider()
        st.caption(
            f"Language: {sidebar_config['selected_language']} | Mode: {processing_mode} | "
            "Uses Supadata for transcripts, AssemblyAI for ASR fallback, DeepSeek for structuring, and YouTube Data API v3 for playlists."
        )

        # Installation instructions
        with st.expander("📦 Installation Requirements", expanded=False):
            st.markdown("""
            To use all features of this app, install the required packages:
            
            ```bash
            pip install streamlit supadata requests google-api-python-client
            ```
            
            **Package purposes:**
            - `supadata`: YouTube transcript extraction
            - `google-api-python-client`: YouTube Data API v3 for playlist processing
            - `requests`: DeepSeek and AssemblyAI API communication
            - `streamlit`: Web interface
            """)
    
    def run(self):
        sidebar_config = self.render_sidebar()
        self.render_main_ui(sidebar_config)

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    app = YouTubeTranscriptApp()
    app.run()
