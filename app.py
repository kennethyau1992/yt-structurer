import os
import re
import json
from typing import Optional

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

APP_TITLE = "YouTube Transcript Structurer"
DEFAULT_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/).+",
    re.IGNORECASE,
)

def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match((url or "").strip()))

# English default system prompt (ASCII-only to avoid copy/paste issues)
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

def fetch_transcript_with_supadata(api_key: str, url: str) -> Optional[str]:
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
        resp = client.transcript(url=url, lang="en", text=True, mode="auto")
    except SupadataError as e:
        st.error(f"Supadata error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

    text = _normalize_transcript(resp)
    text = text.strip() if isinstance(text, str) else ""
    return text or None

# ------------- Streamlit UI -------------

st.set_page_config(page_title=APP_TITLE, page_icon="📺", layout="centered")
st.title(APP_TITLE)
st.caption("Step 1: Paste a YouTube URL -> Step 2: Fetch transcript (Supadata) -> Step 3: Structure (DeepSeek)")

with st.sidebar:
    st.subheader("API Keys & Model")
    supa_key = st.text_input("Supadata API Key", type="password")
    ds_key = st.text_input("DeepSeek API Key", type="password")

    st.divider()
    st.subheader("LLM Settings")
    base_url = st.text_input("DeepSeek Base URL", value=DEFAULT_DEEPSEEK_BASE_URL)
    model = st.text_input("DeepSeek Model", value=DEFAULT_DEEPSEEK_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

st.markdown("### 1) YouTube URL")
url = st.text_input("Paste the YouTube address", value="", placeholder="https://www.youtube.com/watch?v=...")

c1, c2 = st.columns([1, 1])
with c1:
    fetch_clicked = st.button("Fetch Transcript", type="primary")
with c2:
    structure_clicked = st.button("Structure Transcript")

if fetch_clicked:
    st.session_state["structured_md"] = None
    if not url or not is_valid_youtube_url(url):
        st.warning("Please enter a valid YouTube URL.")
    elif not supa_key:
        st.warning("Please provide your Supadata API key in the sidebar.")
    else:
        with st.spinner("Contacting Supadata..."):
            text = fetch_transcript_with_supadata(supa_key, url)
        if not text:
            st.info("No transcript is available for this video (no official captions detected).")
            st.session_state["transcript_text"] = None
        else:
            st.success("Transcript fetched.")
            st.session_state["transcript_text"] = text

transcript_text = st.session_state.get("transcript_text")

if transcript_text:
    st.markdown("### 2) Transcript Preview")
    with st.expander("Show transcript", expanded=False):
        st.text_area("", transcript_text, height=240)

if structure_clicked:
    if not st.session_state.get("transcript_text"):
        st.warning("Please fetch a transcript first.")
    elif not ds_key:
        st.warning("Enter your DeepSeek API key in the sidebar to run structuring.")
    else:
        with st.spinner("Asking DeepSeek to structure..."):
            try:
                md = deepseek_chat(
                    api_key=ds_key,
                    model=model,
                    system_prompt=ENGLISH_SYSTEM_PROMPT,
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

structured_md = st.session_state.get("structured_md")
if structured_md:
    st.markdown("### Structured Output (Markdown)")
    st.markdown(structured_md)
    st.download_button(
        "Download .md",
        data=structured_md,
        file_name="structured_transcript.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download .txt",
        data=structured_md,
        file_name="structured_transcript.txt",
        mime="text/plain",
    )

st.divider()
st.caption(
    "Uses Supadata for transcripts and DeepSeek (OpenAI-compatible endpoint) for LLM structuring. "
    "If no official captions exist, no transcript will be returned."
)
