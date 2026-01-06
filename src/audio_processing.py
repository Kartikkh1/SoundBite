import os
import tempfile
import uuid
import asyncio
import re
from accelerate import Accelerator
import torch
from transformers import pipeline
import yt_dlp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableConfig
from src.llm_utils import llm, summary_prompt


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
        'no_warnings': True,
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
    
    # --- Step 1: Transcription ---
    yield "### ⏳ Status: Transcribing..."
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu", # Change to "cuda" if using GPU,
        chunk_length_s=30,
        return_timestamps=True,
        model_kwargs={"dtype": torch.float32}, # Slim images prefer float32 on CPU
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
