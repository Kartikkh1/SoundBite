import os
import tempfile
import uuid
import asyncio
import re
from accelerate import Accelerator
from transformers import pipeline
import gradio as gr
import yt_dlp
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Custom instruction for the LLM
prompt_template = """Write a detailed summary of the following audio transcript. 
Focus on the main arguments and provide the result in bullet points:
"{text}"
SUMMARY:"""

summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


class VideoTooLongError(Exception):
    """Custom exception for videos exceeding the maximum allowed length."""
    pass

def is_valid_url(url):
    # General URL regex from https://stackoverflow.com/questions/55712161/url-validation-regex-in-python
    url_regex = re.compile(
        r'^(?:http)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+'
    )

    if not re.match(url_regex, url):
        return False
    if not re.match(youtube_regex, url):
        return False
    return True

def length_filter(info_dict, *, incomplete):
    duration = info_dict.get('duration')
    if duration and duration > 1200: # 1200 seconds = 20 minutes
        raise VideoTooLongError('Video is too long (max 20 minutes)')
    return None

async def extract_audio_from_url(url):
    temp_dir = tempfile.gettempdir()
    # Generate a unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    audio_base_path = os.path.join(temp_dir, f"audio_{unique_id}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{audio_base_path}.%(ext)s', 
        'noplaylist': True,
        'max_filesize': 30 * 1024 * 1024, # 30 MB limit
        'match_filter': length_filter,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    
    try:
        # Run yt-dlp in a thread to prevent blocking the async loop
        def download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return f"{audio_base_path}.wav"
        
        return True, await asyncio.to_thread(download)
    except VideoTooLongError as e:
        return False, f"### ❌ Error: Video is too long, maximum allowed length is 20 minutes."
    except yt_dlp.DownloadError as e:
        error_msg = str(e)
        if "File is larger than max-filesize" in error_msg:
            return False, "### ❌ Error: User attempted to download a file exceeding 30MB."
        else:
            print(f"yt-dlp Download Error: {e}")
            return False, "### ❌ Error: Failed to download audio. Please check the URL and try again."
    except yt_dlp.PostProcessorError as e:
        print(f"yt-dlp Post-processing Error: {e}")
        return False, "### ❌ Error: Failed to process audio after download. The file might be corrupted or in an unsupported format."
    except Exception as e:
        print(f"General Error downloading audio: {e}")
        return False, "### ❌ Error: Something went wrong during download, please try again later."


async def process_audio(audio_filename):
    device = Accelerator().device
    
    # --- Step 1: Transcription ---
    yield "### ⏳ Status: Transcribing..."
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        chunk_length_s=30,
        return_timestamps=True
    )

    # Run heavy inference in a thread
    result = await asyncio.to_thread(pipe, audio_filename)
    transcription = result["text"]

    # --- Step 2: LangChain Chunking ---
    yield "### ⏳ Status: Summarizing..."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(transcription)
    docs = [Document(page_content=t) for t in texts]

    # --- Step 3: Summarization ---
    yield "### ⏳ Status: Generating AI summary..."
    if len(docs) == 1:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
        res = await chain.ainvoke(docs)
    else:
        chain = load_summarize_chain(
            llm, 
            chain_type="map_reduce", 
            map_prompt=summary_prompt, 
            combine_prompt=summary_prompt
        )
        res = await chain.ainvoke(docs, config=RunnableConfig(max_concurrency=5))

    yield f"## ✅ Summary\n\n{res['output_text']}"

async def process_url_and_audio(audio_filename, url_input):
    # Disable submit button and clear button at the start
    yield "### ⏳ Status: Starting...", gr.update(), gr.update(interactive=False), gr.update(interactive=False)
    
    target_audio = audio_filename
    
    if url_input:
        if not is_valid_url(url_input):
            yield "### ❌ Error: Only YouTube links are supported.", gr.update(), gr.update(interactive=True), gr.update(interactive=True)
            return
        yield "### ⏳ Status: Downloading audio from URL...", gr.update(), gr.update(), gr.update()
        success, result = await extract_audio_from_url(url_input)
        if not success:
            yield result, gr.update(), gr.update(interactive=True), gr.update(interactive=True)
            return
        target_audio = result

    if not target_audio:
        yield "### ❌ Error\nPlease upload a file or enter a valid URL.", gr.update(), gr.update(interactive=True), gr.update(interactive=True)
        return


    try:
        yield "### ⏳ Status: Starting Transcribing...", gr.update(), gr.update(), gr.update()
        async for status_update in process_audio(target_audio):
            if status_update.startswith("## ✅ Summary"):
                yield "### ✅ Complete!", gr.update(value=status_update), gr.update(interactive=True), gr.update(interactive=True)
            else:
                yield status_update, gr.update(), gr.update(), gr.update()
        return 
    except Exception as e:
        yield f"### ❌ Error\nAn error occurred during processing: {str(e)}", gr.update(), gr.update(interactive=True), gr.update(interactive=True)
        return
    finally:
        if url_input and target_audio and os.path.exists(target_audio):
            os.remove(target_audio)

def main():
    with gr.Blocks(title="AI Audio Summarizer") as demo:
        gr.Markdown("# 🎙️ Audio Summary Tool")
        gr.Markdown("Get a bulleted summary from any audio file or YouTube link.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload Audio", type="filepath", sources=["upload", "microphone"])
                url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                submit_btn = gr.Button("Generate Summary", variant="primary", interactive=False)

            with gr.Column(scale=1):
                status_output = gr.Markdown("### ⏳ Status: Awaiting input...", elem_id="status_output")
                summary_output = gr.Markdown(label="The summary will appear here...", elem_id="summary_output")
                clear_btn = gr.ClearButton([audio_input, url_input, summary_output, status_output], elem_id="clear_btn")

        # Logic to enable button only when input exists
        def set_btn_interactive(audio, url):
            if audio is not None or (url and url.strip() != ""):
                return gr.update(interactive=True)
            return gr.update(interactive=False)

        audio_input.change(set_btn_interactive, [audio_input, url_input], submit_btn)
        url_input.change(set_btn_interactive, [audio_input, url_input], submit_btn)

        submit_btn.click(
            fn=process_url_and_audio,
            inputs=[audio_input, url_input],
            outputs=[status_output, summary_output, clear_btn, submit_btn],
            show_progress="hidden"
        )
    demo.queue().launch(debug=True, theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()