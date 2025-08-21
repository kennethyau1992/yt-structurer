import os
import re
import json
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st

# Try to import the Supadata client.
# If the package isn't installed, the app will show a helpful error in the UI.
try:
    from supadata import Supadata, SupadataError
    SUPADATA_AVAILABLE = True
except Exception as e:
    SUPADATA_AVAILABLE = False
    Supadata = None  # type: ignore
    class SupadataError(Exception): ...
    _supadata_import_error = e

# Try to import Google API client for YouTube Data API v3
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except Exception as e:
    YOUTUBE_API_AVAILABLE = False
    _youtube_api_import_error = e

APP_TITLE = "YouTube Transcript Structurer"
DEFAULT_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/).+",
    re.IGNORECASE,
)

PLAYLIST_REGEX = re.compile(
    r"^(https?://)?(www\.)?youtube\.com/playlist\?list=[\w-]+",
    re.IGNORECASE,
)

def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match((url or "").strip()))

def is_valid_playlist_url(url: str) -> bool:
    return bool(PLAYLIST_REGEX.match((url or "").strip()))

def extract_playlist_id(url: str) -> Optional[str]:
    """Extract playlist ID from YouTube playlist URL"""
    match = re.search(r'list=([\w-]+)', url)
    return match.group(1) if match else None

# Language-specific system prompts
ENGLISH_SYSTEM_PROMPT = (
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

CHINESE_SYSTEM_PROMPT = (
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

SYSTEM_PROMPTS = {
    "English": ENGLISH_SYSTEM_PROMPT,
    "中文": CHINESE_SYSTEM_PROMPT
}

LANGUAGE_CODES = {
    "English": "en",
    "中文": "zh"
}

def deepseek_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Minimal OpenAI-compatible /chat/completions call for DeepSeek.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected DeepSeek response: {data}")

def _normalize_transcript(resp) -> str:
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

def fetch_transcript_with_supadata(api_key: str, url: str, language: str = "en") -> Optional[str]:
    """
    Use Supadata to fetch a transcript. No ASR fallback by design.
    If no official captions or empty text, return None.
    """
    if not SUPADATA_AVAILABLE or Supadata is None:
        st.error("Supadata client not available. Install the 'supadata' package:  pip install supadata")
        if not SUPADATA_AVAILABLE:
            st.caption(f"Import error details: {getattr(globals(), '_supadata_import_error', 'unknown')}")
        return None

    client = Supadata(api_key=api_key)
    try:
        resp = client.transcript(url=url, lang=language, text=True, mode="auto")
    except SupadataError as e:
        st.error(f"Supadata error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

    text = _normalize_transcript(resp)
    text = text.strip() if isinstance(text, str) else ""
    return text or None

def get_playlist_videos(playlist_url: str) -> List[Dict[str, str]]:
    """
    Extract video URLs from a YouTube playlist.
    This is a simplified implementation - in production you might want to use YouTube Data API.
    For now, this returns a placeholder structure.
    """
    # Note: This is a simplified approach. For production use, consider:
    # 1. YouTube Data API v3
    # 2. yt-dlp library
    # 3. Other YouTube scraping libraries
    
    # Placeholder implementation - you would need to implement actual playlist parsing
    # This could be done with YouTube Data API or web scraping
    st.warning("Playlist processing is not fully implemented in this demo. You would need to:")
    st.info("""
    1. Use YouTube Data API v3 to get playlist items
    2. Or use a library like yt-dlp to extract video URLs
    3. Or implement web scraping (not recommended due to ToS)
    
    For now, please process videos individually.
    """)
    
    return []

def process_playlist_transcripts(api_key: str, playlist_url: str, language: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Process all videos in a playlist and return their transcripts.
    Returns list of tuples: (video_title, video_url, transcript_text)
    """
    videos = get_playlist_videos(playlist_url)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, video in enumerate(videos):
        status_text.text(f"Processing video {i+1}/{len(videos)}: {video['title']}")
        
        transcript = fetch_transcript_with_supadata(api_key, video['url'], language)
        results.append((video['title'], video['url'], transcript))
        
        progress_bar.progress((i + 1) / len(videos))
    
    status_text.text("All videos processed!")
    return results

# ------------- Streamlit UI -------------

st.set_page_config(page_title=APP_TITLE, page_icon="📺", layout="centered")
st.title(APP_TITLE)
st.caption("Enhanced with language selection and playlist support")

with st.sidebar:
    st.subheader("API Keys & Model")
    supa_key = st.text_input("Supadata API Key", type="password")
    ds_key = st.text_input("DeepSeek API Key", type="password")

    st.divider()
    st.subheader("Language Settings")
    selected_language = st.selectbox(
        "Select Language",
        options=list(LANGUAGE_CODES.keys()),
        index=0
    )
    
    st.divider()
    st.subheader("LLM Settings")
    base_url = st.text_input("DeepSeek Base URL", value=DEFAULT_DEEPSEEK_BASE_URL)
    model = st.text_input("DeepSeek Model", value=DEFAULT_DEEPSEEK_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

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

# Single video processing
if processing_mode == "Single Video" and fetch_clicked:
    st.session_state["structured_md"] = None
    if not url or not is_valid_youtube_url(url):
        st.warning("Please enter a valid YouTube URL.")
    elif not supa_key:
        st.warning("Please provide your Supadata API key in the sidebar.")
    else:
        language_code = LANGUAGE_CODES[selected_language]
        with st.spinner(f"Fetching {selected_language} transcript..."):
            text = fetch_transcript_with_supadata(supa_key, url, language_code)
        if not text:
            st.info(f"No {selected_language} transcript is available for this video (no official captions detected).")
            st.session_state["transcript_text"] = None
        else:
            st.success("Transcript fetched.")
            st.session_state["transcript_text"] = text

# Playlist processing
if processing_mode == "Playlist" and fetch_playlist_clicked:
    st.session_state["playlist_transcripts"] = None
    st.session_state["structured_playlist"] = None
    
    if not playlist_url or not is_valid_playlist_url(playlist_url):
        st.warning("Please enter a valid YouTube playlist URL.")
    elif not supa_key:
        st.warning("Please provide your Supadata API key in the sidebar.")
    else:
        language_code = LANGUAGE_CODES[selected_language]
        with st.spinner(f"Processing playlist for {selected_language} transcripts..."):
            playlist_results = process_playlist_transcripts(supa_key, playlist_url, language_code)
        
        if playlist_results:
            st.session_state["playlist_transcripts"] = playlist_results
            successful_count = sum(1 for _, _, transcript in playlist_results if transcript)
            st.success(f"Processed {len(playlist_results)} videos, {successful_count} with transcripts.")
        else:
            st.info("No videos processed from playlist.")

# Display single video transcript
transcript_text = st.session_state.get("transcript_text")
if processing_mode == "Single Video" and transcript_text:
    st.markdown("### 2) Transcript Preview")
    with st.expander("Show transcript", expanded=False):
        st.text_area("", transcript_text, height=240, key="single_transcript")

# Display playlist transcripts
playlist_transcripts = st.session_state.get("playlist_transcripts", [])
if processing_mode == "Playlist" and playlist_transcripts:
    st.markdown("### 2) Playlist Transcripts Preview")
    
    for i, (title, video_url, transcript) in enumerate(playlist_transcripts):
        with st.expander(f"Video {i+1}: {title}", expanded=False):
            st.write(f"**URL:** {video_url}")
            if transcript:
                st.text_area("Transcript:", transcript, height=200, key=f"playlist_transcript_{i}")
            else:
                st.info("No transcript available for this video")

# Single video structuring
if processing_mode == "Single Video" and structure_clicked:
    if not st.session_state.get("transcript_text"):
        st.warning("Please fetch a transcript first.")
    elif not ds_key:
        st.warning("Enter your DeepSeek API key in the sidebar to run structuring.")
    else:
        system_prompt = SYSTEM_PROMPTS[selected_language]
        with st.spinner("Asking DeepSeek to structure..."):
            try:
                md = deepseek_chat(
                    api_key=ds_key,
                    model=model,
                    system_prompt=system_prompt,
                    user_content=st.session_state["transcript_text"],
                    base_url=base_url,
                    temperature=temperature,
                )
            except Exception as e:
                md = None
                st.error(f"DeepSeek error: {e}")
        if md:
            st.session_state["structured_md"] = md
            st.success("Structured successfully.")

# Playlist structuring
if processing_mode == "Playlist" and structure_playlist_clicked:
    if not st.session_state.get("playlist_transcripts"):
        st.warning("Please fetch playlist transcripts first.")
    elif not ds_key:
        st.warning("Enter your DeepSeek API key in the sidebar to run structuring.")
    else:
        system_prompt = SYSTEM_PROMPTS[selected_language]
        structured_results = []
        
        transcripts_to_process = [
            (title, url, transcript) for title, url, transcript in playlist_transcripts 
            if transcript
        ]
        
        if not transcripts_to_process:
            st.warning("No transcripts available to structure.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (title, video_url, transcript) in enumerate(transcripts_to_process):
                status_text.text(f"Structuring video {i+1}/{len(transcripts_to_process)}: {title}")
                
                try:
                    structured_md = deepseek_chat(
                        api_key=ds_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_content=transcript,
                        base_url=base_url,
                        temperature=temperature,
                    )
                    structured_results.append((title, video_url, structured_md))
                except Exception as e:
                    st.error(f"Error structuring '{title}': {e}")
                    structured_results.append((title, video_url, None))
                
                progress_bar.progress((i + 1) / len(transcripts_to_process))
            
            status_text.text("All transcripts structured!")
            st.session_state["structured_playlist"] = structured_results
            st.success(f"Structured {len([r for r in structured_results if r[2]])} transcripts successfully.")

# Display single video structured output
structured_md = st.session_state.get("structured_md")
if processing_mode == "Single Video" and structured_md:
    st.markdown("### 3) Structured Output")
    st.markdown(structured_md)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download .md",
            data=structured_md,
            file_name="structured_transcript.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            "Download .txt",
            data=structured_md,
            file_name="structured_transcript.txt",
            mime="text/plain",
        )

# Display playlist structured output
structured_playlist = st.session_state.get("structured_playlist", [])
if processing_mode == "Playlist" and structured_playlist:
    st.markdown("### 3) Structured Playlist Output")
    
    # Create combined markdown for all videos
    combined_md = f"# Playlist Transcripts - {selected_language}\n\n"
    
    for i, (title, video_url, structured_md) in enumerate(structured_playlist):
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
    
    if any(result[2] for result in structured_playlist):
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
    f"Language: {selected_language} | Mode: {processing_mode} | "
    "Uses Supadata for transcripts and DeepSeek for structuring. "
    "Note: Playlist functionality requires additional implementation for video URL extraction."
)