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
                    time.sleep(3)
                    attempt += 1
                    
                else:
                    st.warning(f"Unknown status: {status}")
                    time.sleep(3)
                    attempt += 1
            
            except Exception as e:
                st.error(f"Polling error: {e}")
                return None
        
        st.error("Transcription polling timed out after 6 minutes")
        return None

# ==================== IMPROVED LLM PROVIDER WITH SUMMARY ====================

class LLMProvider:
    """Base class for LLM providers"""
    def generate_summary(self, transcript: str, language: str) -> Optional[str]:
        raise NotImplementedError
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        raise NotImplementedError

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.text_chunker = LLMTextChunker(
            max_chunk_length=8000,
            overlap_length=200
        )
    
    def generate_summary(self, transcript: str, language: str) -> Optional[str]:
        """Generate executive summary with key points - FAST mode"""
        try:
            if language == "English":
                summary_prompt = """You are an expert at analyzing transcripts and creating concise executive summaries.

Create a brief executive summary with the following structure:

## Executive Summary

### Key Points
- List 3-5 main points from the transcript
- Each point should be concise and impactful
- Focus on the most important information

### Main Ideas
- Identify 2-3 core themes or concepts
- Brief explanation of each (1-2 sentences)

### Quick Overview
- 2-3 paragraph summary of the entire content
- Highlight the most critical information
- Keep it under 300 words

Format using markdown. Be concise and focus on value."""
            else:
                summary_prompt = """你是分析转录文本并创建简洁执行摘要的专家。

创建以下结构的简要执行摘要：

## 执行摘要

### 关键要点
- 列出转录文本中的3-5个主要观点
- 每个要点应简洁有力
- 专注于最重要的信息

### 主要思想
- 识别2-3个核心主题或概念
- 每个简要说明（1-2句）

### 快速概览
- 2-3段整个内容的摘要
- 突出最关键的信息
- 保持在300字以内

使用markdown格式。保持简洁并专注于价值。"""
            
            return self._make_api_request(
                transcript[:12000],
                summary_prompt,
                temperature=0.3,
                model="deepseek-chat"
            )
            
        except Exception as e:
            st.error(f"Summary generation error: {e}")
            return None
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Enhanced transcript structuring with intelligent chunking"""
        try:
            if not self.text_chunker.should_chunk_text(transcript):
                return self._process_single_chunk(transcript, system_prompt)
            else:
                return self._process_with_chunking(transcript, system_prompt)
                
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")
    
    def _process_single_chunk(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Process transcript as a single chunk"""
        return self._make_api_request(transcript, system_prompt)
    
    def _process_with_chunking(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Process long transcript with intelligent chunking"""
        chunks = self.text_chunker.create_intelligent_chunks(transcript)
        
        if len(chunks) == 1:
            return self._process_single_chunk(transcript, system_prompt)
        
        st.info(f"Processing {len(chunks)} chunks for detailed structuring...")
        
        language = "English" if "English" in system_prompt else "中文"
        
        processed_chunks = []
        for i, chunk_info in enumerate(chunks):
            chunk_text = chunk_info['text']
            
            chunk_prompt = self._create_chunk_prompt(system_prompt, chunk_info, len(chunks), language)
            
            with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                result = self._make_api_request(chunk_text, chunk_prompt)
                
                if result:
                    processed_chunks.append(result)
                else:
                    st.warning(f"Chunk {i+1} failed, skipping...")
        
        if not processed_chunks:
            st.error("All chunks failed to process")
            return None
        
        return self._combine_processed_chunks(processed_chunks, language)
    
    def _create_chunk_prompt(self, base_prompt: str, chunk_info: Dict, total_chunks: int, language: str) -> str:
        """Create context-aware prompt for chunk"""
        chunk_id = chunk_info['chunk_id']
        is_final = chunk_info.get('is_final_chunk', False)
        
        if chunk_info.get('is_single_chunk', False):
            return base_prompt
        
        if language == "English":
            chunk_context = f"""
CHUNK CONTEXT: Processing chunk {chunk_id + 1} of {total_chunks}
- Process ONLY this chunk's content
- {"This is the FINAL chunk" if is_final else "More chunks follow"}
"""
        else:
            chunk_context = f"""
分块上下文：处理第 {chunk_id + 1} 块，共 {total_chunks} 块
- 仅处理此块内容
- {"这是最后一块" if is_final else "后续还有更多块"}
"""
        
        return chunk_context + base_prompt
    
    def _combine_processed_chunks(self, chunks: List[str], language: str) -> str:
        """Combine processed chunks"""
        if len(chunks) == 1:
            return chunks[0]
        
        separator = "\n\n---\n\n"
        return separator.join(chunks)
    
    def _make_api_request(self, text: str, system_prompt: str, temperature: Optional[float] = None, model: Optional[str] = None) -> Optional[str]:
        """Make API request to DeepSeek"""
        try:
            endpoint = self.base_url + "/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": model or self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": temperature or self.temperature,
            }

            resp = requests.post(
                endpoint, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=180
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
        except Exception as e:
            raise RuntimeError(f"DeepSeek processing error: {str(e)}")

class YouTubeDataProvider:
    """Provider for YouTube Data API operations"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.available = bool(api_key)
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, str]]:
        """Extract video URLs from a YouTube playlist"""
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
                params = {
                    'part': 'snippet',
                    'playlistId': playlist_id,
                    'maxResults': 50,
                    'key': self.api_key
                }
                
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                response = requests.get(
                    f"{self.base_url}/playlistItems",
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    st.error(f"YouTube API error ({response.status_code})")
                    return []
                
                playlist_response = response.json()
                
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
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos
            
        except Exception as e:
            st.error(f"Error processing playlist: {e}")
            return []
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from URL"""
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
        st.info("Trying primary transcript provider (Supadata)...")
        transcript = self.transcript_provider.get_transcript(url, language)
        
        def is_valid_transcript(text):
            """Check if transcript is valid and substantial enough"""
            if not text:
                return False
            
            if "BatchJob" in str(text) or "job_id" in str(text):
                st.warning("Received job object instead of transcript")
                return False
            
            cleaned = text.strip()
            if len(cleaned) < 100:
                st.warning(f"Transcript too short ({len(cleaned)} characters)")
                return False
            
            word_count = len(cleaned.split())
            if word_count < 20:
                st.warning(f"Transcript has only {word_count} words")
                return False
            
            return True
        
        if not is_valid_transcript(transcript):
            transcript = None
            st.warning("Primary provider returned invalid transcript")
        
        if not transcript and use_fallback and self.asr_fallback_provider:
            st.info("Trying ASR fallback (AssemblyAI)...")
            transcript = self.asr_fallback_provider.get_transcript(url, language)
            
            if not is_valid_transcript(transcript):
                st.error("ASR also failed to produce valid transcript")
                transcript = None
        
        if transcript:
            st.success(f"Transcript obtained ({len(transcript)} chars, {len(transcript.split())} words)")
        else:
            st.warning("No transcript available")
            
        return transcript
    
    def generate_summary(self, transcript: str, language: str) -> Optional[str]:
        """Generate executive summary"""
        if not self.llm_provider:
            return None
        return self.llm_provider.generate_summary(transcript, language)
    
    def structure_transcript(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Structure full transcript"""
        if not self.llm_provider:
            return None
        return self.llm_provider.structure_transcript(transcript, system_prompt)

# ==================== STREAMLIT UI ====================

def login_page():
    """Display login page"""
    st.title("YouTube Transcript Processor")
    st.subheader("Login Required")
    
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
    
    return False

def get_default_prompt(language: str) -> str:
    """Get default system prompt"""
    if language == "English":
        return """You are an expert at analyzing and structuring YouTube video transcripts. Your task is to convert raw transcript text into a well-organized, readable document.

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

Format the output as a clean, professional document that would be easy to read and reference."""
    else:
        return """你是一个专业的YouTube视频转录文本分析和结构化专家。你的任务是将原始转录文本转换为组织良好、易于阅读的文档。

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

def process_videos_improved(videos, language, use_asr_fallback, system_prompt, deepseek_model, temperature, api_keys, browser, user_data):
    """Improved video processing with summary first approach"""
    
    # Initialize providers
    supadata_provider = SupadataTranscriptProvider(api_keys.get('supadata', ''))
    assemblyai_provider = ImprovedAssemblyAITranscriptProvider(
        api_keys.get('assemblyai', ''), 
        browser=browser
    ) if use_asr_fallback else None
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
        st.subheader(f"Processing Video {i+1}/{len(videos)}")
        st.write(f"**Title:** {video['title']}")
        st.write(f"**URL:** {video['url']}")
        
        try:
            # Step 1: Get transcript
            with st.spinner("Getting transcript..."):
                transcript = orchestrator.get_transcript(video['url'], language, use_asr_fallback)
            
            if not transcript:
                st.error("Failed to get transcript")
                continue
            
            # Create containers for async display
            summary_container = st.container()
            detailed_container = st.container()
            
            # Step 2: Generate Executive Summary FIRST (fast)
            summary = None
            with summary_container:
                with st.spinner("Generating executive summary..."):
                    summary = orchestrator.generate_summary(transcript, language)
                
                if summary:
                    st.success("Executive Summary Ready!")
                    
                    with st.expander("Executive Summary", expanded=True):
                        st.markdown(summary)
                        
                        st.download_button(
                            "Download Summary",
                            summary,
                            file_name=f"summary_{video['title'][:30]}.md",
                            mime="text/markdown",
                            key=f"summary_download_{i}"
                        )
                else:
                    st.warning("Could not generate summary")
            
            # Step 3: Process detailed transcript
            structured = None
            with detailed_container:
                st.info("Processing detailed transcript (this may take a moment)...")
                
                progress_text = st.empty()
                progress_text.text("Structuring transcript with AI...")
                
                with st.spinner("Creating detailed structured document..."):
                    structured = orchestrator.structure_transcript(transcript, system_prompt)
                
                progress_text.empty()
                
                if structured:
                    st.success("Detailed Transcript Ready!")
                    
                    with st.expander("Detailed Structured Transcript"):
                        st.markdown(structured[:2000] + "..." if len(structured) > 2000 else structured)
                        
                        if len(structured) > 2000:
                            if st.checkbox("Show full transcript", key=f"show_full_{i}"):
                                st.markdown(structured)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Raw Transcript",
                            transcript,
                            file_name=f"raw_{video['title'][:30]}.txt",
                            mime="text/plain",
                            key=f"raw_download_{i}"
                        )
                    with col2:
                        st.download_button(
                            "Download Structured Transcript",
                            structured,
                            file_name=f"structured_{video['title'][:30]}.md",
                            mime="text/markdown",
                            key=f"structured_download_{i}"
                        )
                else:
                    st.error("Failed to structure transcript")
            
            # Save to history
            history_entry = {
                'url': video['url'],
                'title': video['title'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'transcript': transcript,
                'summary': summary,
                'structured': structured
            }
            user_data.add_to_history(history_entry)
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            continue
    
    st.success("All videos processed successfully!")

def main_app():
    """Main application interface"""
    st.title("YouTube Transcript Processor")
    
    # Initialize user data manager
    if 'user_data_manager' not in st.session_state:
        st.session_state.user_data_manager = UserDataManager(st.session_state.username)
    
    user_data = st.session_state.user_data_manager
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            user_data.save_data()
            for key in ['authenticated', 'username', 'api_keys', 'user_data_manager']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.write(f"Welcome, **{st.session_state.username}**!")
    
    # Get API keys from session
    api_keys = st.session_state.get('api_keys', {})
    
    # Load saved settings
    saved_settings = user_data.get_settings()
    
    # Tabs for main interface
    tab1, tab2, tab3 = st.tabs(["Process Video", "History", "Settings"])
    
    with tab1:
        st.header("Process Video")
        
        # Quick Mode Toggle
        col1, col2 = st.columns([3, 1])
        with col2:
            quick_mode = st.checkbox(
                "Quick Mode",
                value=True,
                help="Use faster model (deepseek-chat) for quicker results"
            )
        
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
                            st.success(f"Loaded {len(videos_to_process)} videos")
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
            
            # Process button
            if st.button("Start Processing", type="primary"):
                language = saved_settings.get('language', 'English')
                use_asr_fallback = saved_settings.get('use_asr_fallback', True)
                deepseek_model = "deepseek-chat" if quick_mode else saved_settings.get('deepseek_model', 'deepseek-chat')
                temperature = saved_settings.get('temperature', 0.1)
                browser_for_cookies = saved_settings.get('browser_for_cookies', 'none')
                system_prompt = saved_settings.get('system_prompt') or get_default_prompt(language)
                
                if not api_keys.get('supadata') and not api_keys.get('assemblyai'):
                    st.error("No transcript providers available. Configure API keys in Settings.")
                    return
                
                if not api_keys.get('deepseek'):
                    st.error("DeepSeek API key required. Configure in Settings.")
                    return
                
                process_videos_improved(
                    videos_to_process, language, use_asr_fallback, 
                    system_prompt, deepseek_model, temperature, 
                    api_keys, browser_for_cookies, user_data
                )
    
    with tab2:
        st.header("Processing History")
        
        history = user_data.get_history()
        if history:
            for entry in history[:10]:
                with st.expander(f"{entry['title']} - {entry.get('timestamp', 'Unknown time')}"):
                    st.write(f"**URL:** {entry['url']}")
                    
                    if entry.get('summary'):
                        st.markdown("### Executive Summary")
                        st.markdown(entry['summary'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if entry.get('transcript'):
                            st.download_button(
                                "Raw Transcript",
                                entry['transcript'],
                                file_name=f"transcript_{entry['title'][:30]}.txt",
                                mime="text/plain"
                            )
                    with col2:
                        if entry.get('summary'):
                            st.download_button(
                                "Summary",
                                entry['summary'],
                                file_name=f"summary_{entry['title'][:30]}.md",
                                mime="text/markdown"
                            )
                    with col3:
                        if entry.get('structured'):
                            st.download_button(
                                "Structured",
                                entry['structured'],
                                file_name=f"structured_{entry['title'][:30]}.md",
                                mime="text/markdown"
                            )
        else:
            st.info("No processing history yet")
    
    with tab3:
        st.header("Settings")
        
        with st.form("settings_form"):
            st.subheader("Processing Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox(
                    "Language",
                    ["English", "中文"],
                    index=0 if saved_settings.get('language', 'English') == 'English' else 1,
                    help="Select transcription language"
                )
                
                use_asr_fallback = st.checkbox(
                    "Enable ASR Fallback",
                    value=saved_settings.get('use_asr_fallback', True),
                    help="Use AssemblyAI when captions unavailable"
                )
                
                browser_for_cookies = st.selectbox(
                    "Browser for Cookies",
                    ["none", "chrome", "firefox", "edge", "safari"],
                    index=["none", "chrome", "firefox", "edge", "safari"].index(
                        saved_settings.get('browser_for_cookies', 'none')
                    ),
                    help="Browser for YouTube authentication"
                )
            
            with col2:
                deepseek_model = st.selectbox(
                    "DeepSeek Model",
                    ["deepseek-chat", "deepseek-reasoner"],
                    index=0 if saved_settings.get('deepseek_model', 'deepseek-chat') == 'deepseek-chat' else 1,
                    help="Model for transcript processing"
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=saved_settings.get('temperature', 0.1),
                    step=0.1,
                    help="Controls AI creativity"
                )
            
            st.subheader("System Prompt")
            use_custom_prompt = st.checkbox("Use Custom Prompt", value=saved_settings.get('system_prompt') is not None)
            
            if use_custom_prompt:
                system_prompt = st.text_area(
                    "Custom System Prompt",
                    value=saved_settings.get('system_prompt', get_default_prompt(language)),
                    height=200,
                    help="Customize AI instructions"
                )
            else:
                system_prompt = None
            
            if st.form_submit_button("Save Settings"):
                new_settings = {
                    'language': language,
                    'use_asr_fallback': use_asr_fallback,
                    'deepseek_model': deepseek_model,
                    'temperature': temperature,
                    'browser_for_cookies': browser_for_cookies,
                    'system_prompt': system_prompt if use_custom_prompt else None
                }
                user_data.update_settings(new_settings)
                st.success("Settings saved!")
        
        # API Status
        st.subheader("API Status")
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
                st.success("✅ YouTube")
            else:
                st.error("❌ YouTube")
        
        # Cookie Status
        with st.expander("YouTube Authentication"):
            render_cookies = '/etc/secrets/youtube_cookies.txt'
            if os.path.exists(render_cookies):
                st.success(f"✅ Render cookies found")
            else:
                st.warning("⚠️ No Render cookies file")
                st.info("Add youtube_cookies.txt to Render secrets")

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
    main()#!/usr/bin/env python3
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
