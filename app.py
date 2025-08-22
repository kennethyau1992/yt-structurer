
import os
import re
import json
import time
import random
import hashlib
from typing import Optional, List, Dict, Any, Callable

import requests
import streamlit as st

# ============================
# Retry
# ============================
def with_retries(fn: Callable, tries: int = 3, base_delay: float = 0.6, max_delay: float = 2.0):
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
# Error Humanizer
# ============================
def humanize_yt_error(exc: Exception) -> str:
    s = str(exc)
    if "Sign in to confirm you’re not a bot" in s or "Sign in to confirm you're not a bot" in s:
        return ("YouTube is gating this video. Enable one of: "
                "1) Use cookies from your browser (Chrome/Brave/Edge/Firefox), or "
                "2) Upload cookies.txt (Netscape format). Then retry.")
    if "HTTP Error 429" in s or "Too Many Requests" in s:
        return "YouTube rate-limited the request. Toggle cookies and retry, or wait a minute."
    if "This video is private" in s or "Members-only" in s:
        return "Video requires authentication (private/members-only). Use authenticated cookies from your browser."
    if "Failed to load cookies" in s or "failed to load cookies" in s:
        return "Failed to load cookies. Switch cookie mode or upload a valid cookies.txt, then retry."
    return ""

# ============================
# Cookie Options Builder
# ============================
def build_ytdlp_opts_from_session(quiet: bool = True) -> Dict[str, Any]:
    """
    Build yt-dlp options using the selected cookie mode.
    cookie_mode: 'none' | 'browser:chrome' | 'browser:brave' | 'browser:edge' | 'browser:firefox' | 'cookies.txt'
    """
    cookie_mode = st.session_state.get("cookie_mode", "none")
    cookiefile_path: Optional[str] = st.session_state.get("yt_cookiefile_path", None)

    opts: Dict[str, Any] = {
        "quiet": quiet,
        "noprogress": True,
        "nocheckcertificate": True,
        "source_address": "0.0.0.0",
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,zh-HK;q=0.8,zh-TW;q=0.8",
            "Referer": "https://www.youtube.com/",
        },
        "format": "bestaudio/best",
        "ratelimit": 5_000_000,
        "retries": 3,
        "ignoreerrors": True,
    }

    if cookie_mode.startswith("browser:"):
        browser = cookie_mode.split(":", 1)[1]
        opts["cookiesfrombrowser"] = (browser,)
    elif cookie_mode == "cookies.txt" and cookiefile_path:
        if os.path.exists(cookiefile_path):
            opts["cookiefile"] = cookiefile_path
    # else: none
    return opts

# ============================
# Duration via API (then probe)
# ============================
def _parse_iso8601_duration_to_seconds(s: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h, m_, s_ = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + m_ * 60 + s_

def _extract_yt_video_id(youtube_url: str) -> str:
    # strip timestamp & other params after v=
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

def get_video_duration_seconds(youtube_url: str) -> int:
    yt_api_key = st.session_state.get("api_keys", {}).get("youtube", "") or os.getenv("YT_API_KEY", "")
    if yt_api_key:
        try:
            secs = get_duration_via_youtube_api(youtube_url, yt_api_key)
            if secs > 0:
                return secs
        except Exception:
            pass
    # Fallback: lightweight probe (with current cookie mode) — but safe to ignore failures
    def _probe():
        import yt_dlp
        opts = build_ytdlp_opts_from_session()
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        return int(info.get("duration") or 0)
    try:
        return with_retries(_probe, tries=1)
    except Exception:
        return 0

# ============================
# Audio extractor (no download)
# ============================
class YouTubeAudioExtractor:
    """Extract a signed audio URL. If chosen cookie mode fails to load, fall back to 'none' once."""
    def extract_audio_url(self, youtube_url: str) -> Optional[str]:
        import yt_dlp
        st.info("🎵 Extracting audio stream URL using yt-dlp...")

        def _try_with_opts(opts):
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    raise RuntimeError("Could not extract video information")
                formats = info.get("formats", [])
                # Prefer audio-only
                audio_formats = [f for f in formats if f.get("acodec") != "none" and f.get("vcodec") == "none"]
                candidates = audio_formats or [f for f in formats if f.get("acodec") != "none"]
                if not candidates:
                    raise RuntimeError("No audio streams found in video")
                candidates.sort(key=lambda x: x.get("abr", 0) or 0, reverse=True)
                best = candidates[0]
                url = best.get("url")
                if not url:
                    raise RuntimeError("Stream URL missing")
                return url

        # First attempt: user-selected cookie mode
        opts = build_ytdlp_opts_from_session()
        try:
            url = with_retries(lambda: _try_with_opts(opts), tries=2)
            st.success("✅ Audio URL extracted")
            return url
        except Exception as e:
            # If the failure is cookie loading related, retry with cookie_mode='none'
            msg = str(e)
            human = humanize_yt_error(e)
            st.warning(human or f"First attempt failed: {msg}")
            if "Failed to load cookies" in msg or "failed to load cookies" in msg or "could not find browser" in msg:
                st.info("🔁 Retrying without cookies...")
                prev_mode = st.session_state.get("cookie_mode", "none")
                try:
                    st.session_state["cookie_mode"] = "none"
                    url = with_retries(lambda: _try_with_opts(build_ytdlp_opts_from_session()), tries=1)
                    st.session_state["cookie_mode"] = prev_mode  # restore
                    st.success("✅ Audio URL extracted (no cookies)")
                    return url
                except Exception as e2:
                    st.session_state["cookie_mode"] = prev_mode
                    # Show humanized message (likely bot-gate now)
                    human2 = humanize_yt_error(e2)
                    st.error(human2 or f"Audio extraction failed: {e2}")
                    return None
            else:
                st.error(human or f"❌ Audio extraction error: {e}")
                return None

# ============================
# AssemblyAI (remote URL)
# ============================
A2_BASE = "https://api.assemblyai.com/v2"

def _aai_key() -> str:
    key = (st.session_state.get("api_keys", {}) or {}).get("assemblyai") or os.getenv("ASSEMBLYAI_API_KEY", "")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key

def assemblyai_start_and_wait(audio_url: str, language_code: str = "en", **kwargs) -> Optional[str]:
    headers = {"authorization": _aai_key(), "content-type": "application/json"}
    payload = {"audio_url": audio_url, "punctuate": True, "format_text": True, "language_code": language_code}
    payload.update(kwargs)
    r = requests.post(f"{A2_BASE}/transcript", headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        try:
            err = r.json().get("error", r.text)
        except Exception:
            err = r.text
        st.error(f"❌ AssemblyAI submission error ({r.status_code}): {err}")
        return None
    tid = r.json()["id"]
    st.success(f"✅ Transcription submitted: {tid}")
    # Poll
    progress = st.progress(0)
    status_txt = st.empty()
    attempts = 0
    while True:
        time.sleep(3)
        attempts += 1
        pr = requests.get(f"{A2_BASE}/transcript/{tid}", headers={"authorization": _aai_key()}, timeout=30)
        if pr.status_code != 200:
            st.error(f"❌ Polling error ({pr.status_code}): {pr.text[:200]}")
            return None
        data = pr.json()
        status = data.get("status", "unknown")
        status_txt.text(f"📊 Status: {status} (attempt {attempts})")
        progress.progress(min(0.95, attempts / 120))
        if status == "completed":
            progress.progress(1.0)
            return data.get("text", "")
        if status == "error":
            st.error(f"❌ AssemblyAI error: {data.get('error')}")
            return None

# ============================
# Preflight + API Status (REDO)
# ============================
def safe_import(name: str):
    try:
        mod = __import__(name)
        return mod, getattr(mod, "__version__", "unknown")
    except Exception:
        return None, None

def render_api_status():
    # Robust key getter
    keys = (st.session_state.get("api_keys") or {}) if isinstance(st.session_state.get("api_keys"), dict) else {}
    supa_ok = bool(keys.get("supadata") or os.getenv("SUPADATA_API_KEY") or os.getenv("ADMIN_SUPADATA_KEY"))
    aai_ok = bool(keys.get("assemblyai") or os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ADMIN_ASSEMBLYAI_KEY"))
    ds_ok  = bool(keys.get("deepseek")  or os.getenv("DEEPSEEK_API_KEY")   or os.getenv("ADMIN_DEEPSEEK_KEY"))
    yt_ok  = bool(keys.get("youtube")   or os.getenv("YT_API_KEY")         or os.getenv("ADMIN_YOUTUBE_KEY") or os.getenv("YOUTUBE_API_KEY"))

    yt_dlp_mod, yt_dlp_ver = safe_import("yt_dlp")
    req_mod, req_ver = safe_import("requests")
    st_mod, st_ver = safe_import("streamlit")

    cookie_mode = st.session_state.get("cookie_mode", "none")
    cookiefile = st.session_state.get("yt_cookiefile_path", None)
    cookiefile_ok = bool(cookiefile and os.path.exists(cookiefile))

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**APIs**")
        st.write("Supadata: " + ("✅" if supa_ok else "❌"))
        st.write("AssemblyAI: " + ("✅" if aai_ok else "❌"))
        st.write("DeepSeek: " + ("✅" if ds_ok else "❌"))
        st.write("YouTube Data: " + ("✅" if yt_ok else "❌"))
    with cols[1]:
        st.markdown("**Libraries**")
        st.write(f"yt-dlp: {'✅ '+yt_dlp_ver if yt_dlp_mod else '❌'}")
        st.write(f"requests: {'✅ '+req_ver if req_mod else '❌'}")
        st.write(f"streamlit: {'✅ '+st_ver if st_mod else '❌'}")
    with cols[2]:
        st.markdown("**Cookies**")
        st.write(f"Mode: `{cookie_mode}`")
        if cookie_mode == "cookies.txt":
            st.write("cookies.txt: " + ("✅ file loaded" if cookiefile_ok else "❌ missing"))
        else:
            st.write("cookies.txt: (not used)")
    with cols[3]:
        st.markdown("**Notes**")
        if cookie_mode.startswith("browser:"):
            st.caption("Browser cookies require local browser profile access. If it fails, switch to cookies.txt.")
        elif cookie_mode == "cookies.txt" and not cookiefile_ok:
            st.caption("Upload a valid Netscape cookies.txt (yt-dlp wiki has instructions).")
        else:
            st.caption("If YouTube gates the video, enable cookies.")

# ============================
# Providers (Supadata + YouTube auto-captions)
# ============================
class TranscriptProvider:
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        raise NotImplementedError

class SupadataTranscriptProvider(TranscriptProvider):
    def __init__(self, api_key: str):
        try:
            from supadata import Supadata  # noqa
            self.client = Supadata(api_key=api_key)
            self.available = True
        except Exception:
            self.client = None
            self.available = False

    def get_transcript(self, url: str, language: str) -> Optional[str]:
        if not self.available or not self.client:
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
    def get_transcript(self, url: str, language: str) -> Optional[str]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # noqa
        except Exception:
            return None
        try:
            vid = _extract_yt_video_id(url)
            langs = ["zh-Hant", "yue", "zh", "en"] if language != "English" else ["en", "zh-Hant", "yue", "zh"]
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
# LLM (DeepSeek)
# ============================
class LLMProvider:
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        raise NotImplementedError

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}],
            "temperature": self.temperature,
        }
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=180)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {detail}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("DeepSeek API returned no choices")
        content = choices[0]["message"]["content"]
        if not content:
            raise RuntimeError("DeepSeek API returned empty content")
        return content.strip()

# ============================
# Orchestrator
# ============================
class TranscriptOrchestrator:
    def __init__(self, transcript_provider: TranscriptProvider, asr_fallback: bool, assemblyai_enabled: bool):
        self.transcript_provider = transcript_provider
        self.asr_fallback = asr_fallback
        self.assemblyai_enabled = assemblyai_enabled
        self.audio_extractor = YouTubeAudioExtractor()

    def get_transcript(self, url: str, language: str) -> Optional[str]:
        st.info("🔍 Trying primary transcript providers (Supadata → YouTube auto-captions).")
        t = self.transcript_provider.get_transcript(url, language)
        if t:
            st.success("✅ Got transcript from providers")
            return t
        if not self.asr_fallback:
            st.warning("❌ No official captions found and ASR fallback disabled")
            return None
        if not (st.session_state.get("api_keys", {}) or {}).get("assemblyai") and not os.getenv("ASSEMBLYAI_API_KEY"):
            st.error("❌ AssemblyAI key missing")
            return None
        st.info("🎤 No official captions found. Trying ASR fallback (remote URL to AssemblyAI)...")
        # Extract direct audio URL (no download)
        audio_url = self.audio_extractor.extract_audio_url(url)
        if not audio_url:
            return None
        # Submit and poll
        lang_code = "en" if language == "English" else "zh"
        result = assemblyai_start_and_wait(audio_url, language_code=lang_code)
        if result:
            return result
        # If failed (e.g., signed URL expired), re-extract once
        st.info("🔁 Retrying with a fresh signed URL...")
        audio_url = self.audio_extractor.extract_audio_url(url)
        if not audio_url:
            return None
        return assemblyai_start_and_wait(audio_url, language_code=lang_code)

# ============================
# Auth
# ============================
class AuthManager:
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        self.users["admin"] = {"password_hash": self._hash_password(admin_password),
                               "api_keys": {"supadata": "", "assemblyai": "", "deepseek": "", "youtube": ""}}
        self._load_users_from_env()

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def _load_users_from_env(self):
        admin_keys = self.users["admin"]["api_keys"]
        env_map = {
            "ADMIN_SUPADATA_KEY": "supadata",
            "ADMIN_ASSEMBLYAI_KEY": "assemblyai",
            "ADMIN_DEEPSEEK_KEY": "deepseek",
            "ADMIN_YOUTUBE_KEY": "youtube",
            "SUPADATA_API_KEY": "supadata",
            "ASSEMBLYAI_API_KEY": "assemblyai",
            "DEEPSEEK_API_KEY": "deepseek",
            "YOUTUBE_API_KEY": "youtube",
            "YT_API_KEY": "youtube",
        }
        for k, v in env_map.items():
            if os.getenv(k):
                admin_keys[v] = os.getenv(k)

    def authenticate(self, username: str, password: str) -> bool:
        return username in self.users and self.users[username]["password_hash"] == self._hash_password(password)

    def get_user_api_keys(self, username: str) -> Dict[str, str]:
        return self.users.get(username, {}).get("api_keys", {})

# ============================
# UI
# ============================
def login_page():
    st.title("🎬 YouTube Transcript — Stream Only (v2)")
    st.subheader("🔐 Login")
    st.info("🔑 **Default:** Username: `admin` | Password: `admin123`")
    with st.form("login_form"):
        u = st.text_input("Username", value="admin")
        p = st.text_input("Password", type="password")
        go = st.form_submit_button("Login")
        if go:
            if not u or not p:
                st.error("Please enter both username and password")
                return False
            auth = AuthManager()
            if auth.authenticate(u, p):
                st.session_state.authenticated = True
                st.session_state.username = u
                st.session_state.api_keys = auth.get_user_api_keys(u)
                st.rerun()
            else:
                st.error("Invalid credentials")
    return False

def main_app():
    st.title("🎬 YouTube Transcript — Stream Only (v2)")
    colA, colB = st.columns([6,1])
    with colB:
        if st.button("Logout"):
            for k in ["authenticated","username","api_keys","yt_cookiefile_path"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.write(f"Welcome, **{st.session_state.username}**! 👋")

    st.header("⚙️ Status & Settings")
    with st.expander("📊 API & Environment Status", expanded=True):
        render_api_status()

    with st.expander("🔧 Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", ["English","中文"])
            use_asr_fallback = st.checkbox("Enable ASR Fallback (AssemblyAI via remote URL)", value=True)
        with col2:
            # Cookie Mode selector
            cookie_mode = st.selectbox(
                "Cookie Mode",
                ["none","browser:chrome","browser:brave","browser:edge","browser:firefox","cookies.txt"],
                index=["none","browser:chrome","browser:brave","browser:edge","browser:firefox","cookies.txt"].index(st.session_state.get("cookie_mode","none"))
            )
            st.session_state["cookie_mode"] = cookie_mode
            cookiefile = st.file_uploader("Upload cookies.txt (optional)", type=["txt"])
            if cookiefile is not None:
                # Save to tmp
                tmp_path = os.path.join(os.path.expanduser("~"), f".cookies_{int(time.time())}.txt")
                with open(tmp_path, "wb") as f:
                    f.write(cookiefile.read())
                st.session_state["yt_cookiefile_path"] = tmp_path
                st.success("cookies.txt uploaded")

    st.header("🎯 Process Video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    default_prompts = {
        "English": "You are an expert at structuring YouTube transcripts. Use clear sections and markdown, fix punctuation, preserve meaning, add timestamps at natural transitions.",
        "中文": "你是YouTube轉錄結構化專家。請用清晰章節與Markdown，修正標點，保留原意，在自然轉場處加入時間戳。"
    }
    system_prompt = st.text_area("System Prompt", value=default_prompts[language], height=140)

    # LLM settings
    colL, colR = st.columns(2)
    with colL:
        deepseek_model = st.selectbox("DeepSeek Model", ["deepseek-chat","deepseek-reasoner"], index=1)
    with colR:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    if st.button("🚀 Start", type="primary"):
        keys = st.session_state.get("api_keys") or {}
        if not (keys.get("supadata") or os.getenv("SUPADATA_API_KEY") or os.getenv("ADMIN_SUPADATA_KEY")
                or keys.get("assemblyai") or os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ADMIN_ASSEMBLYAI_KEY")):
            st.error("❌ No transcript providers available (Supadata or AssemblyAI). Configure API keys.")
            st.stop()
        if not (keys.get("deepseek") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("ADMIN_DEEPSEEK_KEY")):
            st.error("❌ DeepSeek API key required for structuring.")
            st.stop()
        if not url:
            st.error("Please provide a YouTube URL")
            st.stop()

        # Providers
        supa_key = keys.get("supadata") or os.getenv("SUPADATA_API_KEY") or os.getenv("ADMIN_SUPADATA_KEY", "")
        aai_key = keys.get("assemblyai") or os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ADMIN_ASSEMBLYAI_KEY", "")
        ds_key  = keys.get("deepseek")  or os.getenv("DEEPSEEK_API_KEY")   or os.getenv("ADMIN_DEEPSEEK_KEY", "")

        supadata = SupadataTranscriptProvider(supa_key)
        yt_auto = YouTubeTranscriptAPIProvider()
        primary = CompositeTranscriptProvider([supadata, yt_auto])
        orch = TranscriptOrchestrator(primary, asr_fallback=use_asr_fallback, assemblyai_enabled=bool(aai_key))

        # Duration (best-effort)
        secs = get_video_duration_seconds(url)
        if secs:
            st.info(f"🕒 Video duration: {secs//60}m {secs%60}s")

        # Get transcript
        with st.spinner("🔍 Getting transcript..."):
            transcript = orch.get_transcript(url, language)
        if not transcript:
            st.error("❌ Failed to get transcript")
            st.stop()

        with st.expander("📄 Raw Transcript Preview"):
            st.text_area("Transcript", transcript[:2000]+"..." if len(transcript)>2000 else transcript, height=240)

        # LLM structure
        ds = DeepSeekProvider(ds_key, "https://api.deepseek.com/v1", deepseek_model, temperature)
        with st.spinner("🤖 Structuring transcript with LLM..."):
            try:
                structured = ds.structure_transcript(transcript, system_prompt)
            except Exception as e:
                st.error(f"DeepSeek error: {e}")
                st.stop()

        st.success("✅ Completed!")
        with st.expander("📋 Structured Transcript", expanded=True):
            st.markdown(structured)
        st.download_button("💾 Download Raw Transcript", transcript, file_name="raw_transcript.txt", mime="text/plain")
        st.download_button("📄 Download Structured Transcript", structured, file_name="structured_transcript.md", mime="text/markdown")

# ============================
# Main
# ============================
def main():
    st.set_page_config(page_title="YouTube Transcript — Stream Only (v2)", page_icon="🎬", layout="wide")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "cookie_mode" not in st.session_state:
        st.session_state.cookie_mode = "none"
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
