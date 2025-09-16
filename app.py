 """
 YouTube Transcript Processor with Executive Summary and Persistent Data
 Complete application with all improvements implemented
 """
 
 import os
 import re
-import json
 import time
+import heapq
 import hashlib
 import pickle
-import concurrent.futures
-from datetime import datetime, timedelta
+from dataclasses import dataclass, field
+from datetime import datetime
 from pathlib import Path
-from typing import Optional, List, Dict, Tuple, Any
+from textwrap import shorten
+from typing import Optional, List, Dict, Any, Tuple
+from urllib.parse import urlparse, parse_qs
+
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
diff --git a/app.py b/app.py
index d4592febfdefff26f4d291db45a83f887a04e6c3..d0630f3e96107123bf15509cfb85dffd82a260f0 100644
--- a/app.py
+++ b/app.py
@@ -251,131 +254,140 @@ class AudioChunker:
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
-        """Transcribe multiple chunks in parallel and combine results"""
+        """Transcribe multiple chunks and combine results."""
         if not chunks:
             return None
-        
-        st.info(f"Processing {len(chunks)} chunks in parallel...")
-        
-        def transcribe_single_chunk(chunk_info):
+
+        chunk_count = len(chunks)
+        st.info(f"Processing {chunk_count} chunks…")
+
+        progress_bar = st.progress(0.0)
+        status_placeholder = st.empty()
+
+        results: List[Dict[str, Any]] = []
+
+        for index, chunk_info in enumerate(chunks, start=1):
             chunk_id = chunk_info['chunk_id']
             start_time = chunk_info['start_time']
             end_time = chunk_info['end_time']
             audio_url = chunk_info['audio_url']
-            
+
+            start_min, start_sec = divmod(start_time, 60)
+            end_min, end_sec = divmod(end_time, 60)
+            status_placeholder.info(
+                f"Chunk {index}/{chunk_count}: "
+                f"{start_min:02d}:{start_sec:02d} – {end_min:02d}:{end_sec:02d}"
+            )
+
             try:
-                st.info(f"Chunk {chunk_id + 1}: {start_time//60}m{start_time%60}s - {end_time//60}m{end_time%60}s")
-                
                 transcript = assemblyai_provider._transcribe_audio_url(audio_url, language)
-                
-                return {
-                    'chunk_id': chunk_id,
-                    'start_time': start_time,
-                    'end_time': end_time,
-                    'transcript': transcript,
-                    'success': transcript is not None
-                }
-                
-            except Exception as e:
-                st.error(f"Chunk {chunk_id + 1} failed: {e}")
-                return {
-                    'chunk_id': chunk_id,
-                    'start_time': start_time,
-                    'end_time': end_time,
-                    'transcript': None,
-                    'success': False,
-                    'error': str(e)
-                }
-        
-        results = []
-        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
-            future_to_chunk = {
-                executor.submit(transcribe_single_chunk, chunk): chunk 
-                for chunk in chunks
-            }
-            
-            for future in concurrent.futures.as_completed(future_to_chunk):
-                chunk = future_to_chunk[future]
-                try:
-                    result = future.result()
-                    results.append(result)
-                    
-                    if result['success']:
-                        st.success(f"Chunk {result['chunk_id'] + 1} completed")
-                    else:
-                        st.error(f"Chunk {result['chunk_id'] + 1} failed")
-                        
-                except Exception as e:
-                    st.error(f"Chunk processing error: {e}")
-        
+                success = bool(transcript)
+
+                results.append(
+                    {
+                        'chunk_id': chunk_id,
+                        'start_time': start_time,
+                        'end_time': end_time,
+                        'transcript': transcript,
+                        'success': success,
+                    }
+                )
+
+                if success:
+                    st.success(f"Chunk {chunk_id + 1} completed")
+                else:
+                    st.warning(f"Chunk {chunk_id + 1} returned no transcript")
+
+            except Exception as exc:
+                st.error(f"Chunk {chunk_id + 1} failed: {exc}")
+                results.append(
+                    {
+                        'chunk_id': chunk_id,
+                        'start_time': start_time,
+                        'end_time': end_time,
+                        'transcript': None,
+                        'success': False,
+                        'error': str(exc),
+                    }
+                )
+
+            progress_bar.progress(index / chunk_count)
+
+        status_placeholder.empty()
+        progress_bar.empty()
+
         results.sort(key=lambda x: x['chunk_id'])
-        
-        successful_transcripts = []
-        failed_chunks = []
-        
+
+        successful_transcripts: List[str] = []
+        failed_chunks: List[int] = []
+
         for result in results:
             if result['success'] and result['transcript']:
                 start_min, start_sec = divmod(result['start_time'], 60)
                 end_min, end_sec = divmod(result['end_time'], 60)
                 timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
-                
+
                 successful_transcripts.append(f"{timestamp}\n{result['transcript']}")
             else:
                 failed_chunks.append(result['chunk_id'] + 1)
-        
+
         if not successful_transcripts:
             st.error("All chunks failed to transcribe")
             return None
-        
+
         if failed_chunks:
-            st.warning(f"Chunks {failed_chunks} failed, but continuing with successful chunks")
-        
+            st.warning(
+                "Chunks "
+                + ", ".join(str(chunk) for chunk in failed_chunks)
+                + " failed, but continuing with successful chunks"
+            )
+
         combined_transcript = "\n\n".join(successful_transcripts)
-        
+
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
diff --git a/app.py b/app.py
index d4592febfdefff26f4d291db45a83f887a04e6c3..d0630f3e96107123bf15509cfb85dffd82a260f0 100644
--- a/app.py
+++ b/app.py
@@ -728,43 +740,806 @@ class ImprovedAssemblyAITranscriptProvider(TranscriptProvider):
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
-                
+
                 if polling_response.status_code != 200:
                     st.error(f"Polling error ({polling_response.status_code})")
                     return None
-                
+
                 polling_data = polling_response.json()
                 status = polling_data.get('status', 'unknown')
-                
+
                 status_text.text(f"Status: {status} (attempt {attempt + 1}/{max_attempts})")
                 progress_bar.progress((attempt + 1) / max_attempts)
-                
+
                 if status == 'completed':
                     status_text.text("Transcription completed successfully!")
                     progress_bar.progress(1.0)
-                    
+
                     transcript_text = polling_data.get('text', '')
                     if transcript_text:
-                        st.success
+                        st.success("AssemblyAI returned a transcript")
+                        return transcript_text
+
+                    st.warning("AssemblyAI completed without returning transcript text")
+                    return ""
+
+                if status == 'error':
+                    error_msg = polling_data.get('error', 'Unknown error')
+                    st.error(f"AssemblyAI reported an error: {error_msg}")
+                    return None
+
+                attempt += 1
+                if attempt < max_attempts:
+                    time.sleep(5)
+                continue
+
+            except requests.RequestException as exc:
+                attempt += 1
+                wait_time = min(5 + attempt // 2, 15)
+                st.warning(f"Polling request failed ({exc}). Retrying in {wait_time} seconds...")
+                time.sleep(wait_time)
+            except Exception as exc:
+                st.error(f"Unexpected polling error: {exc}")
+                return None
+
+        st.error("Timed out waiting for AssemblyAI transcription to complete")
+        return None
+
+# ==================== PROCESSING AND ANALYSIS LAYER ====================
+
+
+class ExecutiveSummaryGenerator:
+    """Generates heuristic summaries and statistics for transcripts."""
+
+    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
+    WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
+    STOPWORDS = {
+        "the",
+        "and",
+        "that",
+        "with",
+        "this",
+        "from",
+        "have",
+        "your",
+        "will",
+        "they",
+        "been",
+        "their",
+        "about",
+        "there",
+        "which",
+        "into",
+        "would",
+        "could",
+        "should",
+        "while",
+        "where",
+        "these",
+        "those",
+        "when",
+        "what",
+        "here",
+        "over",
+        "also",
+        "such",
+        "than",
+        "then",
+        "because",
+        "after",
+        "before",
+        "being",
+        "doing",
+        "does",
+        "done",
+        "ourselves",
+        "yourself",
+        "myself",
+        "ours",
+        "yours",
+        "them",
+        "they're",
+        "we're",
+        "he's",
+        "she's",
+        "it's",
+    }
+
+    def __init__(self, chunker: Optional[LLMTextChunker] = None):
+        self.chunker = chunker
+
+    def _split_sentences(self, text: str) -> List[str]:
+        sentences = [s.strip() for s in self.SENTENCE_SPLIT_RE.split(text) if s.strip()]
+        if not sentences and text.strip():
+            sentences = [text.strip()]
+        return sentences
+
+    def _tokenize_words(self, text: str) -> List[str]:
+        return [match.lower() for match in self.WORD_RE.findall(text)]
+
+    def _compute_word_frequencies(self, words: List[str]) -> Dict[str, float]:
+        frequencies: Dict[str, float] = {}
+        for word in words:
+            if len(word) <= 2 or word in self.STOPWORDS:
+                continue
+            frequencies[word] = frequencies.get(word, 0.0) + 1.0
+
+        if frequencies:
+            max_freq = max(frequencies.values())
+            if max_freq:
+                for key in list(frequencies.keys()):
+                    frequencies[key] /= max_freq
+
+        return frequencies
+
+    def _score_sentences(self, sentences: List[str], frequencies: Dict[str, float]) -> List[Tuple[int, float, str]]:
+        scored: List[Tuple[int, float, str]] = []
+        for idx, sentence in enumerate(sentences):
+            words = self.WORD_RE.findall(sentence)
+            if not words:
+                continue
+            score = sum(frequencies.get(word.lower(), 0.0) for word in words)
+            normalized = score / max(len(words), 1)
+            if normalized > 0:
+                scored.append((idx, normalized, sentence.strip()))
+        return scored
+
+    def _select_sentences(self, scored: List[Tuple[int, float, str]], limit: int) -> List[str]:
+        if not scored:
+            return []
+
+        top_scored = heapq.nlargest(limit, scored, key=lambda item: item[1])
+        top_scored.sort(key=lambda item: item[0])
+        return [sentence for _, _, sentence in top_scored]
+
+    def _build_stats(self, text: str, sentences: List[str], words: List[str]) -> Dict[str, Any]:
+        unique_words = len(set(words))
+        word_count = len(words)
+        reading_time_minutes = word_count / 200 if word_count else 0
+        speaking_time_minutes = word_count / 130 if word_count else 0
+
+        return {
+            "characters": len(text),
+            "words": word_count,
+            "sentences": len(sentences),
+            "unique_words": unique_words,
+            "estimated_reading_minutes": round(reading_time_minutes, 2),
+            "estimated_speaking_minutes": round(speaking_time_minutes, 2),
+        }
+
+    def _chunk_metadata(self, text: str) -> List[Dict[str, Any]]:
+        if not self.chunker:
+            return []
+
+        chunk_data: List[Dict[str, Any]] = []
+        for chunk in self.chunker.create_intelligent_chunks(text):
+            preview_source = re.sub(r"\s+", " ", chunk.get('text', '').strip())
+            preview = shorten(preview_source, width=160, placeholder="…")
+            chunk_data.append(
+                {
+                    "chunk_id": chunk.get('chunk_id', 0),
+                    "tokens_estimate": chunk.get('tokens_estimate', 0),
+                    "is_final": chunk.get('is_final_chunk', chunk.get('is_single_chunk', False)),
+                    "preview": preview,
+                }
+            )
+        return chunk_data
+
+    def analyze(self, text: str) -> Dict[str, Any]:
+        cleaned = text.strip()
+        if not cleaned:
+            return {
+                "summary": "",
+                "key_points": [],
+                "keywords": [],
+                "stats": {
+                    "characters": 0,
+                    "words": 0,
+                    "sentences": 0,
+                    "unique_words": 0,
+                    "estimated_reading_minutes": 0,
+                    "estimated_speaking_minutes": 0,
+                },
+                "chunks": [],
+            }
+
+        sentences = self._split_sentences(cleaned)
+        words = self._tokenize_words(cleaned)
+        frequencies = self._compute_word_frequencies(words)
+        scored_sentences = self._score_sentences(sentences, frequencies)
+
+        summary_sentences = self._select_sentences(scored_sentences, limit=5)
+        key_point_sentences = self._select_sentences(scored_sentences, limit=8)
+
+        keywords = [word for word, _ in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:10]]
+
+        stats = self._build_stats(cleaned, sentences, words)
+        chunk_data = self._chunk_metadata(cleaned)
+
+        return {
+            "summary": " ".join(summary_sentences),
+            "key_points": key_point_sentences,
+            "keywords": keywords,
+            "stats": stats,
+            "chunks": chunk_data,
+        }
+
+
+@dataclass
+class TranscriptProcessingResult:
+    """Represents the outcome of a transcript retrieval attempt."""
+
+    transcript: str
+    provider: str
+    used_fallback: bool
+    fetch_duration: float
+    attempted_providers: List[str] = field(default_factory=list)
+    errors: List[str] = field(default_factory=list)
+    warnings: List[str] = field(default_factory=list)
+    cached: bool = False
+
+
+class TranscriptProcessingManager:
+    """Coordinates transcript retrieval and summary analysis."""
+
+    def __init__(self):
+        self.chunker = LLMTextChunker()
+        self.summary_generator = ExecutiveSummaryGenerator(self.chunker)
+        self._cache: Dict[str, TranscriptProcessingResult] = {}
+
+    @staticmethod
+    def _cache_key(url: str, language: str) -> str:
+        cache_input = f"{url}|{language}".encode("utf-8", errors="ignore")
+        return hashlib.sha256(cache_input).hexdigest()
+
+    def fetch_transcript(
+        self,
+        youtube_url: str,
+        language: str,
+        api_keys: Dict[str, str],
+        use_asr_fallback: bool,
+        browser: str,
+    ) -> TranscriptProcessingResult:
+        """Fetch transcript using available providers with optional fallback."""
+
+        cache_key = self._cache_key(youtube_url, language)
+        if cache_key in self._cache:
+            cached = self._cache[cache_key]
+            return TranscriptProcessingResult(
+                transcript=cached.transcript,
+                provider=cached.provider,
+                used_fallback=cached.used_fallback,
+                fetch_duration=0.0,
+                attempted_providers=list(cached.attempted_providers),
+                errors=list(cached.errors),
+                warnings=list(cached.warnings),
+                cached=True,
+            )
+
+        start_time = time.time()
+        transcript_text = ""
+        provider_used = ""
+        attempted: List[str] = []
+        errors: List[str] = []
+        warnings: List[str] = []
+
+        supadata_key = (api_keys or {}).get("supadata", "").strip()
+        assembly_key = (api_keys or {}).get("assemblyai", "").strip()
+
+        if supadata_key:
+            attempted.append("Supadata")
+            provider = SupadataTranscriptProvider(supadata_key)
+            transcript_text = provider.get_transcript(youtube_url, language) or ""
+            if transcript_text:
+                provider_used = "Supadata"
+            else:
+                warnings.append("Supadata transcript unavailable for this video")
+        else:
+            warnings.append("Supadata API key not provided")
+
+        used_fallback = False
+
+        if not transcript_text:
+            if use_asr_fallback:
+                if assembly_key:
+                    attempted.append("AssemblyAI")
+                    provider = ImprovedAssemblyAITranscriptProvider(assembly_key, browser)
+                    transcript_text = provider.get_transcript(youtube_url, language) or ""
+                    if transcript_text:
+                        provider_used = "AssemblyAI"
+                        used_fallback = True
+                    else:
+                        errors.append("AssemblyAI fallback did not return a transcript")
+                else:
+                    warnings.append("AssemblyAI API key not provided for fallback transcription")
+            else:
+                warnings.append("ASR fallback disabled and no transcript available from Supadata")
+
+        fetch_duration = time.time() - start_time
+
+        if not transcript_text:
+            errors.append("No transcript could be retrieved for the supplied video")
+
+        result = TranscriptProcessingResult(
+            transcript=transcript_text,
+            provider=provider_used or "Unavailable",
+            used_fallback=used_fallback,
+            fetch_duration=fetch_duration,
+            attempted_providers=attempted,
+            errors=errors,
+            warnings=warnings,
+        )
+
+        if transcript_text:
+            self._cache[cache_key] = result
+
+        return result
+
+
+# ==================== STREAMLIT UTILITIES ====================
+
+
+def is_streamlit_running() -> bool:
+    """Detect whether the script is executed within a Streamlit runtime."""
+
+    try:
+        from streamlit.runtime.scriptrunner import get_script_run_ctx
+
+        return get_script_run_ctx() is not None
+    except Exception:
+        return False
+
+
+def extract_video_id(youtube_url: str) -> Optional[str]:
+    """Extract the YouTube video identifier from the provided URL."""
+
+    try:
+        parsed = urlparse(youtube_url)
+    except Exception:
+        return None
+
+    hostname = parsed.netloc.lower()
+
+    if hostname.endswith("youtu.be"):
+        return parsed.path.lstrip("/") or None
+
+    if "youtube.com" in hostname:
+        if parsed.path.startswith("/shorts/"):
+            return parsed.path.split("/shorts/")[-1] or None
+        params = parse_qs(parsed.query)
+        if "v" in params:
+            return params["v"][0]
+
+    return None
+
+
+def validate_youtube_url(youtube_url: str) -> bool:
+    """Basic validation to ensure the URL appears to be a YouTube link."""
+
+    if not youtube_url:
+        return False
+
+    try:
+        parsed = urlparse(youtube_url)
+    except Exception:
+        return False
+
+    hostname = parsed.netloc.lower()
+    return any(domain in hostname for domain in ("youtube.com", "youtu.be"))
+
+
+def ensure_session_defaults() -> None:
+    """Initialize Streamlit session state with expected keys."""
+
+    defaults = {
+        "authenticated": False,
+        "username": None,
+        "api_keys": {},
+        "user_data_manager": None,
+        "processing_result": None,
+        "analysis": None,
+    }
+
+    for key, value in defaults.items():
+        if key not in st.session_state:
+            st.session_state[key] = value
+
+
+def sync_settings_from_storage(user_data_manager: UserDataManager) -> None:
+    """Ensure stored settings populate the current Streamlit session."""
+
+    settings = user_data_manager.get_settings()
+    defaults = {
+        "language": "English",
+        "use_asr_fallback": True,
+        "browser_for_cookies": "none",
+        "deepseek_model": "deepseek-chat",
+        "temperature": 0.1,
+        "system_prompt": None,
+    }
+
+    for key, fallback in defaults.items():
+        session_key = f"settings_{key}"
+        if session_key not in st.session_state:
+            st.session_state[session_key] = settings.get(key, fallback)
+
+
+def update_user_settings(user_data_manager: UserDataManager) -> None:
+    updated_settings = {
+        "language": st.session_state.get("settings_language", "English"),
+        "use_asr_fallback": st.session_state.get("settings_use_asr_fallback", True),
+        "browser_for_cookies": st.session_state.get("settings_browser_for_cookies", "none"),
+        "deepseek_model": st.session_state.get("settings_deepseek_model", "deepseek-chat"),
+        "temperature": st.session_state.get("settings_temperature", 0.1),
+        "system_prompt": st.session_state.get("settings_system_prompt"),
+    }
+    user_data_manager.update_settings(updated_settings)
+
+
+def render_sidebar(user_data_manager: UserDataManager) -> None:
+    """Render sidebar controls for settings and API key management."""
+
+    sync_settings_from_storage(user_data_manager)
+
+    st.sidebar.header("Processing Settings")
+    st.sidebar.selectbox(
+        "Transcript language",
+        options=["English", "中文"],
+        key="settings_language",
+    )
+    st.sidebar.checkbox(
+        "Enable AssemblyAI fallback", key="settings_use_asr_fallback"
+    )
+    st.sidebar.selectbox(
+        "Browser for YouTube cookies",
+        options=["none", "chrome", "firefox", "edge"],
+        key="settings_browser_for_cookies",
+    )
+
+    with st.sidebar.expander("Advanced LLM settings", expanded=False):
+        st.text_input(
+            "Preferred DeepSeek model",
+            key="settings_deepseek_model",
+        )
+        st.slider(
+            "LLM temperature",
+            min_value=0.0,
+            max_value=1.0,
+            value=float(st.session_state.get("settings_temperature", 0.1)),
+            key="settings_temperature",
+        )
+        st.text_area(
+            "Custom system prompt",
+            key="settings_system_prompt",
+            height=120,
+        )
+
+    if st.sidebar.button("Save settings", key="save_settings_button"):
+        update_user_settings(user_data_manager)
+        st.sidebar.success("Settings saved")
+
+    stored_keys = user_data_manager.data.get("api_keys", {})
+    current_keys = {**stored_keys, **st.session_state.get("api_keys", {})}
+
+    with st.sidebar.expander("API keys", expanded=False):
+        supadata_key = st.text_input(
+            "Supadata API key",
+            value=current_keys.get("supadata", ""),
+            type="password",
+        )
+        assembly_key = st.text_input(
+            "AssemblyAI API key",
+            value=current_keys.get("assemblyai", ""),
+            type="password",
+        )
+        deepseek_key = st.text_input(
+            "DeepSeek API key",
+            value=current_keys.get("deepseek", ""),
+            type="password",
+        )
+        youtube_key = st.text_input(
+            "YouTube API key",
+            value=current_keys.get("youtube", ""),
+            type="password",
+        )
+
+        if st.button("Save API keys", key="save_api_keys_button"):
+            user_data_manager.data.setdefault("api_keys", {})
+            user_data_manager.data["api_keys"].update(
+                {
+                    "supadata": supadata_key.strip(),
+                    "assemblyai": assembly_key.strip(),
+                    "deepseek": deepseek_key.strip(),
+                    "youtube": youtube_key.strip(),
+                }
+            )
+            user_data_manager.save_data()
+            st.session_state["api_keys"] = user_data_manager.data.get("api_keys", {})
+            st.sidebar.success("API keys updated")
+
+
+def handle_authentication(auth_manager: AuthManager) -> bool:
+    """Render authentication controls and update session state."""
+
+    ensure_session_defaults()
+
+    if not st.session_state["authenticated"]:
+        st.sidebar.header("Sign in")
+        with st.sidebar.form("login_form"):
+            username = st.text_input("Username", key="login_username")
+            password = st.text_input(
+                "Password", type="password", key="login_password"
+            )
+            submitted = st.form_submit_button("Log in")
+
+        if submitted:
+            if auth_manager.authenticate(username, password):
+                st.session_state["authenticated"] = True
+                st.session_state["username"] = username
+
+                user_data_manager = UserDataManager(username)
+                st.session_state["user_data_manager"] = user_data_manager
+
+                stored_keys = auth_manager.get_user_api_keys(username)
+                combined_keys = {
+                    **stored_keys,
+                    **user_data_manager.data.get("api_keys", {}),
+                }
+                st.session_state["api_keys"] = combined_keys
+
+                st.sidebar.success("Login successful")
+                st.experimental_rerun()
+            else:
+                st.sidebar.error("Invalid username or password")
+
+        return False
+
+    st.sidebar.caption(f"Signed in as {st.session_state['username']}")
+    if st.sidebar.button("Log out", key="logout_button"):
+        st.session_state["authenticated"] = False
+        st.session_state["username"] = None
+        st.session_state["api_keys"] = {}
+        st.session_state["user_data_manager"] = None
+        st.session_state["processing_result"] = None
+        st.session_state["analysis"] = None
+        st.experimental_rerun()
+
+    return True
+
+
+def add_history_entry(
+    user_data_manager: UserDataManager,
+    youtube_url: str,
+    result: TranscriptProcessingResult,
+    analysis: Optional[Dict[str, Any]],
+    language: str,
+) -> None:
+    """Persist the latest processing result to the user's history."""
+
+    video_id = extract_video_id(youtube_url)
+    summary_preview = ""
+    if analysis and analysis.get("summary"):
+        summary_preview = shorten(analysis["summary"], width=160, placeholder="…")
+
+    history_entry = {
+        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
+        "url": youtube_url,
+        "video_id": video_id,
+        "provider": result.provider,
+        "language": language,
+        "summary": summary_preview,
+        "keywords": (analysis or {}).get("keywords", [])[:5],
+    }
+
+    user_data_manager.add_to_history(history_entry)
+
+
+def render_history(user_data_manager: UserDataManager) -> None:
+    """Display the user's recent processing history."""
+
+    st.subheader("Recent activity")
+    history = user_data_manager.get_history()
+
+    if not history:
+        st.info("No history recorded yet. Process a video to see entries here.")
+        return
+
+    for entry in history[:10]:
+        label = entry.get("timestamp", "Unknown time")
+        title = entry.get("summary") or entry.get("url") or "Previous result"
+        with st.expander(f"{label} — {title}"):
+            st.markdown(f"**Video URL:** {entry.get('url', 'Unknown')}")
+            if entry.get("video_id"):
+                st.markdown(f"**Video ID:** `{entry['video_id']}`")
+            st.markdown(f"**Provider:** {entry.get('provider', 'Unknown')}")
+            st.markdown(f"**Language:** {entry.get('language', 'Unknown')}")
+            keywords = entry.get("keywords", [])
+            if keywords:
+                st.markdown("**Keywords:** " + ", ".join(keywords))
+
+
+def render_processing_interface(
+    user_data_manager: UserDataManager, processor: TranscriptProcessingManager
+) -> None:
+    """Render the main interface for processing YouTube transcripts."""
+
+    st.header("YouTube Transcript Processor")
+    st.write(
+        "Provide a YouTube video URL to retrieve its transcript and generate an executive summary."
+    )
+
+    youtube_url = st.text_input(
+        "YouTube video URL",
+        key="youtube_input",
+        placeholder="https://www.youtube.com/watch?v=...",
+    )
+
+    language = st.session_state.get("settings_language", "English")
+    use_asr_fallback = st.session_state.get("settings_use_asr_fallback", True)
+    browser = st.session_state.get("settings_browser_for_cookies", "none")
+
+    if st.button("Process video", key="process_video_button"):
+        if not validate_youtube_url(youtube_url):
+            st.error("Please provide a valid YouTube URL")
+        else:
+            with st.spinner("Fetching transcript..."):
+                result = processor.fetch_transcript(
+                    youtube_url=youtube_url,
+                    language=language,
+                    api_keys=st.session_state.get("api_keys", {}),
+                    use_asr_fallback=use_asr_fallback,
+                    browser=browser,
+                )
+
+            st.session_state["processing_result"] = result
+
+            if result.transcript:
+                analysis = processor.summary_generator.analyze(result.transcript)
+                st.session_state["analysis"] = analysis
+                add_history_entry(user_data_manager, youtube_url, result, analysis, language)
+            else:
+                st.session_state["analysis"] = None
+
+    result: Optional[TranscriptProcessingResult] = st.session_state.get("processing_result")
+    analysis: Optional[Dict[str, Any]] = st.session_state.get("analysis")
+
+    if not result:
+        return
+
+    st.subheader("Processing details")
+    columns = st.columns(4)
+    columns[0].metric("Provider", result.provider or "None")
+    columns[1].metric("Used fallback", "Yes" if result.used_fallback else "No")
+    columns[2].metric(
+        "Fetch duration",
+        f"{result.fetch_duration:.1f}s" if not result.cached else "0.0s (cached)",
+    )
+    columns[3].metric(
+        "Transcript length",
+        f"{len(result.transcript.split()):,} words" if result.transcript else "0 words",
+    )
+
+    if result.cached:
+        st.info("Using cached transcript from earlier in this session")
+
+    for warning in result.warnings:
+        if warning:
+            st.warning(warning)
+
+    for error in result.errors:
+        if error:
+            st.error(error)
+
+    if not result.transcript:
+        return
+
+    if analysis:
+        st.subheader("Executive summary")
+        summary_text = analysis.get("summary", "")
+        if summary_text:
+            st.markdown(summary_text)
+        else:
+            st.info("Summary could not be generated for this transcript")
+
+        key_points = analysis.get("key_points", [])
+        keywords = analysis.get("keywords", [])
+        stats = analysis.get("stats", {})
+
+        col_a, col_b = st.columns(2)
+        with col_a:
+            st.markdown("**Key points**")
+            if key_points:
+                for point in key_points:
+                    st.markdown(f"- {point}")
+            else:
+                st.write("No key points extracted")
+
+        with col_b:
+            st.markdown("**Keywords**")
+            if keywords:
+                st.write(", ".join(keywords))
+            else:
+                st.write("No keywords identified")
+
+        metric_cols = st.columns(4)
+        metric_cols[0].metric("Characters", f"{stats.get('characters', 0):,}")
+        metric_cols[1].metric("Words", f"{stats.get('words', 0):,}")
+        metric_cols[2].metric("Sentences", f"{stats.get('sentences', 0):,}")
+        metric_cols[3].metric(
+            "Reading time",
+            f"{stats.get('estimated_reading_minutes', 0):.2f} min",
+        )
+
+        chunk_data = analysis.get("chunks", [])
+        if chunk_data:
+            st.subheader("Suggested transcript chunks")
+            for chunk in chunk_data[:10]:
+                st.markdown(
+                    f"**Chunk {chunk.get('chunk_id', 0) + 1}** — approx. {chunk.get('tokens_estimate', 0):,} tokens"
+                )
+                st.code(chunk.get("preview", ""))
+
+    with st.expander("Full transcript", expanded=False):
+        st.text(result.transcript)
+        st.download_button(
+            "Download transcript",
+            data=result.transcript,
+            file_name="transcript.txt",
+            mime="text/plain",
+        )
+
+
+def main() -> None:
+    """Entry point for the Streamlit application."""
+
+    if not is_streamlit_running():
+        print("This application is intended to be run with 'streamlit run app.py'.")
+        return
+
+    st.set_page_config(page_title="YouTube Transcript Processor", layout="wide")
+
+    auth_manager = AuthManager()
+    authenticated = handle_authentication(auth_manager)
+
+    if not authenticated:
+        st.title("YouTube Transcript Processor")
+        st.write("Sign in using the sidebar to begin processing videos.")
+        return
+
+    user_data_manager = st.session_state.get("user_data_manager")
+    if not isinstance(user_data_manager, UserDataManager):
+        username = st.session_state.get("username") or "guest"
+        user_data_manager = UserDataManager(username)
+        st.session_state["user_data_manager"] = user_data_manager
+
+    render_sidebar(user_data_manager)
+
+    processor = TranscriptProcessingManager()
+    render_processing_interface(user_data_manager, processor)
+    render_history(user_data_manager)
+
+
+if __name__ == "__main__":
+    main()
