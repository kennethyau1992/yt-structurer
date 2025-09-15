# app.py
import os
import re
import json
import time
import random
from typing import Optional, List, Dict, Any, Callable

import requests
import streamlit as st

# ============================
# Environment helpers (Render + cookies.txt)
# ============================
def running_on_render() -> bool:
    return bool(os.getenv("RENDER") == "true" or os.getenv("RENDER_SERVICE_ID") or os.getenv("RENDER_INSTANCE_ID"))

def resolve_secret_cookiefile() -> Optional[str]:
    """
    Detect a cookies.txt via:
      1) $YT_COOKIES_FILE
      2) /etc/secrets/youtube_cookies.txt  (Render Secret File)
      3) /var/data/youtube_cookies.txt     (Persistent disk)
    """
    candidates = [
        os.getenv("YT_COOKIES_FILE"),
        "/etc/secrets/youtube_cookies.txt",
        "/var/data/youtube_cookies.txt",
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p) and os.path.getsize(p) > 50:
                return p
        except Exception:
            continue
    return None

def copy_cookie_to_tmp_if_readonly(path: str) -> str:
    """
    If the cookie file lives in a read-only area (like /etc/secrets),
    copy it to /tmp and return the new path. If copy fails, return original.
    """
    if not path:
        return path
    ro_roots = ("/etc/secrets", "/app")
    if any(os.path.realpath(path).startswith(root) for root in ro_roots):
        try:
            os.makedirs("/tmp", exist_ok=True)
            stamp = int(os.path.getmtime(path))
            dest = f"/tmp/yt_cookies_{stamp}.txt"
            if not os.path.exists(dest):
                with open(path, "rb") as src, open(dest, "wb") as out:
                    out.write(src.read())
            return dest
        except Exception:
            return path
    return path

# ============================
# Retry helper
# ============================
def with_retries(fn: Callable, tries: int = 2, base_delay: float = 0.6, max_delay: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i == tries - 1:
                break
            time.sleep(min(max_delay, base_delay * (1.7 ** i)) + random.uniform(0, 0.2))
    raise last

# ============================
# Error humanizer
# ============================
def humanize_yt_error(exc: Exception) -> str:
    s = str(exc)
    if "Sign in to confirm you’re not a bot" in s or "Sign in to confirm you're not a bot" in s:
        return ("YouTube is gating this video. Provide a valid cookies.txt via a Secret File "
                "(e.g., /etc/secrets/youtube_cookies.txt).")
    if "HTTP Error 429" in s or "Too Many Requests" in s:
        return "YouTube rate-limited the request. Ensure cookies are present or retry later."
    if "This video is private" in s or "Members-only" in s:
        return "Video requires authentication (private/members-only). Use authenticated cookies from your browser."
    if "Failed to load cookies" in s or "failed to load cookies" in s:
        return "Failed to load cookies. Check that the file is valid Netscape cookies.txt."
    if "could not find browser" in s or "no suitable browser" in s:
        return "Browser cookie access isn’t available on this server. Use a Secret File instead."
    if "Read-only file system" in s:
        return "Secret File path is read-only; the app copies cookies to /tmp automatically."
    return ""

# ============================
# Cookie validation (for the status panel)
# ============================
def validate_cookie_file(cookie_path: Optional[str]) -> Dict[str, Any]:
    ok = True
    msg = None
    hints = []
    eff = None
    if not cookie_path or not os.path.exists(cookie_path):
        ok = False
        msg = "No cookies.txt found."
        hints.append("Add a Secret File at /etc/secrets/youtube_cookies.txt or set YT_COOKIES_FILE.")
    else:
        try:
            size = os.path.getsize(cookie_path)
            if size < 50:
                ok = False
                msg = "cookies.txt appears empty or too small."
            else:
                with open(cookie_path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(2048)
                if ".youtube.com" not in head and "youtube" not in head.lower():
                    hints.append("cookies.txt doesn’t mention youtube; export while signed in to YouTube.")
        except Exception as e:
            ok = False
            msg = f"Could not read cookies.txt: {e}"
        eff = copy_cookie_to_tmp_if_readonly(cookie_path)
        if eff != cookie_path and eff.startswith("/tmp"):
            hints.append(f"Using writable copy: {eff}")
    return {"ok": ok, "message": msg, "hints": hints, "effective": eff or cookie_path}

# ============================
# Duration helpers (YouTube API → yt-dlp probe)
# ============================
def _parse_iso8601_duration_to_seconds(s: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h, m_, s_ = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + m_ * 60 + s_

def _extract_yt_video_id(youtube_url: str) -> str:
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    return youtube_url

def get_duration_via_youtube_api(youtube_url: str, api_key: Optional[str]) -> int:
    if not api_key:
        return 0
    vid = _extract_yt_video_id(youtube_url)
    url = "https://www.googleapis.com/youtube/v3/videos"
    r = requests.get(url, params={"id": vid, "part": "contentDetails", "key": api_key}, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return 0
    iso = items[0]["contentDetails"]["duration"]
    return _parse_iso8601_duration_to_seconds(iso)

def get_video_duration_seconds(youtube_url: str, cookie_path: Optional[str]) -> int:
    yt_api_key = os.getenv("YT_API_KEY") or os.getenv("YOUTUBE_API_KEY") or ""
    if yt_api_key:
        try:
            secs = get_duration_via_youtube_api(youtube_url, yt_api_key)
            if secs > 0:
                return secs
        except Exception:
            pass

    def _probe():
        import yt_dlp
        opts = {
            "quiet": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "http_headers": {
                "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0.0.0 Safari/537.36"),
                "Accept-Language": "en-US,en;q=0.9,zh-HK;q=0.8,zh-TW;q=0.8",
                "Referer": "https://www.youtube.com/",
            },
        }
        if cookie_path and os.path.exists(cookie_path):
            opts["cookiefile"] = copy_cookie_to_tmp_if_readonly(cookie_path)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        return int(info.get("duration") or 0)

    try:
        return with_retries(_probe, tries=1)
    except Exception:
        return 0

# ============================
# YouTube audio extractor (stream-only; cookie-aware; no browser cookies)
# ============================
class YouTubeAudioExtractor:
    @staticmethod
    def build_opts(cookie_path: Optional[str]) -> dict:
        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "nocheckcertificate": True,
            "retries": 3,
            "http_headers": {
                "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0.0.0 Safari/537.36"),
                "Accept-Language": "en-US,en;q=0.9,zh-HK;q=0.8,zh-TW;q=0.8",
                "Referer": "https://www.youtube.com/",
            },
        }
        if cookie_path and os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = copy_cookie_to_tmp_if_readonly(cookie_path)
        return ydl_opts

    @staticmethod
    def extract_audio_url(youtube_url: str, cookie_path: Optional[str]) -> Optional[str]:
        try:
            import yt_dlp
        except ImportError:
            st.error("yt-dlp not installed. Install with: pip install yt-dlp")
            return None

        st.info("🎵 Extracting audio stream URL using yt-dlp...")
        try:
            opts = YouTubeAudioExtractor.build_opts(cookie_path)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    st.error("Could not extract video information (gated or unavailable).")
                    return None

                formats = info.get("formats", [])
                audio_only = [f for f in formats if f.get("acodec") != "none" and f.get("vcodec") == "none"]
                if audio_only:
                    audio_only.sort(key=lambda x: x.get("abr", 0) or 0, reverse=True)
                    best = audio_only[0]
                    st.success(f"Found audio-only stream: {best.get('format_note', 'best')}")
                    return best.get("url")

                with_audio = [f for f in formats if f.get("acodec") != "none"]
                if with_audio:
                    with_audio.sort(key=lambda x: x.get("abr", 0) or 0, reverse=True)
                    best = with_audio[0]
                    st.success(f"Found audio stream: {best.get('format_note', 'best')}")
                    return best.get("url")

                st.error("No audio streams found in video.")
                return None
        except Exception as e:
            st.error(f"Audio extraction error: {humanize_yt_error(e) or e}")
            return None

# ============================
# Transcript providers
# ============================
class TranscriptProvider:
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class SupadataTranscriptProvider(TranscriptProvider):
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                from supadata import Supadata  # type: ignore
                self.client = Supadata(api_key=api_key)
            except Exception:
                self.client = None

    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            resp = self.client.transcript(url=url, lang=language, text=True, mode="auto")
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict) and "text" in resp:
                return resp.get("text") or ""
            if isinstance(resp, list):
                return "\n".join([str(x) for x in resp])
            return str(resp)
        except Exception:
            return None

class YouTubeTranscriptAPIProvider(TranscriptProvider):
    def __init__(self):
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401
            self.available = True
        except Exception:
            self.available = False

    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            vid = _extract_yt_video_id(url)
            langs = ["en", "zh-Hant", "yue", "zh"] if language == "English" else ["zh-Hant", "yue", "zh", "en"]
            transcript = YouTubeTranscriptApi.get_transcript(vid, languages=langs)
            if not transcript:
                return None
            return "\n".join([item.get("text", "") for item in transcript if item.get("text")])
        except Exception:
            return None

class CompositeTranscriptProvider(TranscriptProvider):
    def __init__(self, providers: List[TranscriptProvider]):
        self.providers = providers

    def get_transcript(self, url: str, language: str) -> Optional[str]:
        for p in self.providers:
            t = p.get_transcript(url, language)
            if t:
                return t
        return None

# ============================
# AssemblyAI (ASR) provider — stream URL, upload, and direct URL
# ============================
class AssemblyAIProvider:
    def __init__(self, api_key: str, base_url: str = "https://api.assemblyai.com/v2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"authorization": self.api_key, "content-type": "application/json"}

    def _poll(self, tid: str) -> Optional[str]:
        status_spot = st.empty()
        progress = st.progress(0)
        attempts = 0
        while True:
            time.sleep(3)
            attempts += 1
            r = requests.get(f"{self.base_url}/transcript/{tid}", headers={"authorization": self.api_key}, timeout=30)
            if r.status_code != 200:
                st.error(f"Polling error ({r.status_code}): {r.text[:200]}")
                return None
            data = r.json()
            status = data.get("status", "unknown")
            status_spot.text(f"📊 Status: {status} (attempt {attempts})")
            progress.progress(min(0.95, attempts / 120))
            if status == "completed":
                progress.progress(1.0)
                return data.get("text", "")
            if status == "error":
                st.error(f"AssemblyAI error: {data.get('error')}")
                return None

    def transcribe_stream_url(self, audio_url: str, language: str) -> Optional[str]:
        payload = {
            "audio_url": audio_url,
            "punctuate": True,
            "format_text": True,
            "language_code": "en" if language == "English" else "zh"
        }
        r = requests.post(f"{self.base_url}/transcript", headers=self._headers(), json=payload, timeout=30)
        if r.status_code != 200:
            try:
                err = r.json().get("error", r.text)
            except Exception:
                err = r.text
            st.error(f"AssemblyAI submission error ({r.status_code}): {err}")
            return None
        tid = r.json()["id"]
        st.success(f"✅ Transcription submitted: {tid}")
        return self._poll(tid)

    def upload_and_transcribe_file(self, file_bytes: bytes, language: str) -> Optional[str]:
        # 1) Upload bytes
        up = requests.post(f"{self.base_url}/upload", headers={"authorization": self.api_key}, data=file_bytes, timeout=300)
        if up.status_code != 200:
            st.error(f"AssemblyAI upload error ({up.status_code}): {up.text[:200]}")
            return None
        upload_url = up.json().get("upload_url")
        if not upload_url:
            st.error("AssemblyAI did not return an upload_url.")
            return None
        # 2) Transcribe by URL
        return self.transcribe_stream_url(upload_url, language)

    def transcribe_direct_audio_url(self, direct_url: str, language: str) -> Optional[str]:
        return self.transcribe_stream_url(direct_url, language)

# ============================
# LLM (DeepSeek) — structure transcript
# ============================
class DeepSeekProvider:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-reasoner", temperature: float = 0.1):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        if not self.api_key:
            return transcript  # graceful: return raw if no LLM key
        endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}],
            "temperature": self.temperature,
        }
        r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=180)
        if r.status_code != 200:
            try:
                detail = r.json().get("error", {}).get("message", r.text)
            except Exception:
                detail = r.text
            st.error(f"DeepSeek API error {r.status_code}: {detail}")
            return transcript
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            st.error("DeepSeek API returned no choices.")
            return transcript
        content = choices[0]["message"]["content"]
        return (content or transcript).strip()

# ============================
# Orchestrator (providers → ASR fallback)
# ============================
class TranscriptOrchestrator:
    def __init__(self, providers: CompositeTranscriptProvider, assemblyai: Optional[AssemblyAIProvider]):
        self.providers = providers
        self.assemblyai = assemblyai

    def get_transcript(self, yt_url: str, language: str, cookie_path: Optional[str], enable_asr_fallback: bool) -> Optional[str]:
        st.info("🔍 Trying primary transcript providers (Supadata → YouTube auto-captions).")
        t = self.providers.get_transcript(yt_url, language)
        if t:
            st.success("✅ Got transcript from providers")
            return t

        if not enable_asr_fallback:
            st.warning("❌ No official captions found and ASR fallback disabled.")
            return None
        if not self.assemblyai:
            st.error("❌ AssemblyAI key missing.")
            return None

        st.info("🎤 No official captions found. Trying ASR fallback (remote URL to AssemblyAI)...")
        audio_url = YouTubeAudioExtractor.extract_audio_url(yt_url, cookie_path)
        if not audio_url:
            # Likely gated or missing cookies
            st.error("🚫 Could not extract audio stream (possibly gated). Use the 'Provide audio directly' options below.")
            return None

        return self.assemblyai.transcribe_stream_url(audio_url, language)

# ============================
# Status panel helpers
# ============================
def safe_import(name: str):
    try:
        mod = __import__(name)
        return mod, getattr(mod, "__version__", "unknown")
    except Exception:
        return None, None

def render_status_panel(cookie_src: Optional[str], cookie_eff: Optional[str]):
    keys = {
        "Supadata": bool(os.getenv("SUPADATA_API_KEY")),
        "AssemblyAI": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "DeepSeek": bool(os.getenv("DEEPSEEK_API_KEY")),
        "YouTube Data": bool(os.getenv("YT_API_KEY") or os.getenv("YOUTUBE_API_KEY")),
    }
    yt_dlp_mod, yt_dlp_ver = safe_import("yt_dlp")
    yta_mod, yta_ver = safe_import("youtube_transcript_api")
    st_mod, st_ver = safe_import("streamlit")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**APIs**")
        for k, v in keys.items():
            st.write(f"{k}: {'✅' if v else '❌'}")
    with cols[1]:
        st.markdown("**Libraries**")
        st.write(f"yt-dlp: {'✅ '+yt_dlp_ver if yt_dlp_mod else '❌'}")
        st.write(f"youtube_transcript_api: {'✅ '+yta_ver if yta_mod else '❌'}")
        st.write(f"streamlit: {'✅ '+st_ver if st_mod else '❌'}")
    with cols[2]:
        st.markdown("**Cookies**")
        if cookie_src:
            st.write("cookies.txt: ✅ detected")
            st.caption(f"source: {cookie_src}")
            if cookie_eff and cookie_eff != cookie_src:
                st.caption(f"using:  {cookie_eff} (writable copy)")
        else:
            st.write("cookies.txt: ❌ not found")

# ============================
# Streamlit UI
# ============================
def main():
    st.set_page_config(page_title="YouTube → Transcript (Stream-only + Upload Fallback)", page_icon="🎬", layout="wide")
    st.title("🎬 YouTube → Transcript (Stream-only + Upload Fallback)")

    # Detect cookies
    cookie_src = resolve_secret_cookiefile()
    cookie_check = validate_cookie_file(cookie_src) if cookie_src else {"ok": False, "message": "No cookies.txt", "hints": [], "effective": None}
    cookie_eff = cookie_check["effective"]

    with st.expander("📊 Status", expanded=True):
        render_status_panel(cookie_src, cookie_eff)
        if not cookie_check["ok"]:
            if cookie_check["message"]:
                st.error(cookie_check["message"])
        for h in cookie_check.get("hints", []):
            st.caption("• " + h)

    # Settings
    st.subheader("Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        language = st.selectbox("Language", ["English", "中文"], index=0)
    with col2:
        enable_asr_fallback = st.checkbox("Enable ASR Fallback (AssemblyAI via remote URL)", value=True)
    with col3:
        deepseek_model = st.selectbox("DeepSeek Model", ["deepseek-reasoner", "deepseek-chat"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    default_prompts = {
        "English": "You are an expert at structuring YouTube transcripts. Use clear sections and markdown, fix punctuation, preserve meaning, add timestamps at natural transitions.",
        "中文": "你是YouTube轉錄結構化專家。請用清晰章節與Markdown，修正標點，保留原意，在自然轉場處加入時間戳。"
    }
    system_prompt = st.text_area("System Prompt", value=default_prompts[language], height=120)

    # Inputs — YouTube URL (primary path)
    st.subheader("YouTube URL")
    yt_url = st.text_input("Paste a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    # Optional manual cookies upload (only if no secret file was found)
    if not cookie_src:
        up = st.file_uploader("Upload cookies.txt (Netscape format) — optional", type=["txt"])
        if up is not None:
            tmp_path = f"/tmp/cookies_{int(time.time())}.txt"
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            cookie_eff = tmp_path
            st.success(f"cookies.txt uploaded → {tmp_path}")

    # Providers
    supa = SupadataTranscriptProvider(os.getenv("SUPADATA_API_KEY"))
    yta = YouTubeTranscriptAPIProvider()
    providers = CompositeTranscriptProvider([supa, yta])

    aai_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    aai = AssemblyAIProvider(aai_key) if aai_key else None
    ds = DeepSeekProvider(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        model=deepseek_model,
        temperature=temperature
    )

    # Process button (primary path)
    if st.button("🚀 Get Transcript from YouTube"):
        if not yt_url:
            st.error("Please provide a YouTube URL.")
            st.stop()

        # Duration (best-effort)
        secs = get_video_duration_seconds(yt_url, cookie_eff)
        if secs:
            st.info(f"🕒 Video duration: {secs//60}m {secs%60}s")

        orch = TranscriptOrchestrator(providers, aai)
        with st.spinner("🔍 Getting transcript..."):
            transcript = orch.get_transcript(yt_url, language, cookie_eff, enable_asr_fallback)

        if not transcript:
            st.error("❌ Failed to get transcript.")
            st.info("If the video is gated or cookies are missing, use the fallback below.")
            st.stop()

        with st.expander("📄 Raw Transcript", expanded=False):
            st.text_area("Transcript", transcript[:2000] + ("..." if len(transcript) > 2000 else ""), height=240)

        with st.spinner("🤖 Structuring transcript with LLM..."):
            structured = ds.structure_transcript(transcript, system_prompt)

        st.success("✅ Completed!")
        with st.expander("📋 Structured Transcript", expanded=True):
            st.markdown(structured)
        st.download_button("💾 Download Raw Transcript", transcript, file_name="raw_transcript.txt", mime="text/plain")
        st.download_button("📄 Download Structured Transcript", structured, file_name="structured_transcript.md", mime="text/markdown")

    st.divider()
    st.subheader("If YouTube is gated: provide audio directly")

    col_up, col_url = st.columns(2)
    with col_up:
        uploaded = st.file_uploader(
            "Upload audio file (.mp3, .m4a, .webm, .wav, .ogg, .flac)",
            type=["mp3", "m4a", "webm", "wav", "ogg", "flac"],
            help="We’ll upload this file to AssemblyAI and process it directly."
        )
    with col_url:
        direct_audio_url = st.text_input(
            "Or paste a direct audio URL",
            placeholder="https://... (direct link to audio file)"
        )

    if st.button("🎧 Process Provided Audio"):
        if not aai:
            st.error("AssemblyAI API key required (set ASSEMBLYAI_API_KEY).")
            st.stop()

        transcript2 = None
        if uploaded is not None:
            with st.spinner("Uploading to AssemblyAI and transcribing..."):
                file_bytes = uploaded.read()
                transcript2 = aai.upload_and_transcribe_file(file_bytes, language)

        elif direct_audio_url.strip():
            with st.spinner("Submitting direct audio URL to AssemblyAI..."):
                transcript2 = aai.transcribe_direct_audio_url(direct_audio_url.strip(), language)
        else:
            st.warning("Please upload a file or paste a direct audio URL.")
            st.stop()

        if not transcript2:
            st.error("❌ Failed to transcribe provided audio.")
            st.stop()

        with st.expander("📄 Raw Transcript (provided audio)", expanded=False):
            st.text_area("Transcript", transcript2[:2000] + ("..." if len(transcript2) > 2000 else ""), height=240)

        with st.spinner("🤖 Structuring transcript with LLM..."):
            structured2 = ds.structure_transcript(transcript2, system_prompt)

        st.success("✅ Completed from provided audio!")
        with st.expander("📋 Structured Transcript (provided audio)", expanded=True):
            st.markdown(structured2)
        st.download_button("💾 Download Raw Transcript", transcript2, file_name="raw_transcript.txt", mime="text/plain")
        st.download_button("📄 Download Structured Transcript", structured2, file_name="structured_transcript.md", mime="text/markdown")

if __name__ == "__main__":
    main()
