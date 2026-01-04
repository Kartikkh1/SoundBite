from accelerate import Accelerator
from transformers import pipeline
import gradio as gr
import yt_dlp
from dotenv import load_dotenv
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Define your custom instruction
prompt_template = """Write a detailed summary of the following audio transcript. 
Focus on the main arguments and provide the result in bullet points:
"{text}"
SUMMARY:"""

# Create the Prompt object
summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


async def extract_audio_from_url(url):
    # Use a directory for temp files so we don't have naming conflicts
    temp_dir = tempfile.gettempdir()
    # We use a placeholder for the extension because yt-dlp handles it
    audio_base_path = os.path.join(temp_dir, "input_audio")
    
    ydl_opts = {
        # 'bestaudio' is often webm or m4a; FFmpeg will convert this to wav
        'format': 'bestaudio/best',
        'outtmpl': f'{audio_base_path}.%(ext)s', 
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # Point to the directory containing ffmpeg AND ffprobe
        'ffmpeg_location': '/opt/homebrew/bin', 
        'quiet': False, # Set to False to see the actual logs if it fails
        'verbose': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # extract_info is often more reliable for catching errors early
            info = ydl.extract_info(url, download=True)
            # The post-processor changes the extension to .wav
            final_filename = f"{audio_base_path}.wav"
            
        return final_filename
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Test the function
# result = await extract_audio_from_url("https://www.youtube.com/watch?v=i0Oduk7Lc60")

async def process_url_and_audio(audio_filename, url_input):
    if url_input:
        audio_filename = await extract_audio_from_url(url_input)
        if not audio_filename:
            return "Error: Could not extract audio from URL."

    if not audio_filename:
        return "Error: No audio provided. Please upload a file or enter a URL."
    
    try:
        summary = await process_audio(audio_filename)
        return summary
    finally:
        if url_input and audio_filename and os.path.exists(audio_filename):
            os.remove(audio_filename)


async def process_audio(audio_filename):
    device = Accelerator().device
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        chunk_length_s=30,
        return_timestamps=True
    )

    result = pipe(audio_filename)
    transcription = result["text"]

    # --- Step 2: LangChain Chunking ---
    # We use the recursive splitter to keep sentences intact
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = text_splitter.split_text(transcription)
    docs = [Document(page_content=t) for t in texts]

    # --- Step 3: Adaptive Summarization ---
    if len(docs) == 1:
        # Use 'stuff' for short text (Faster & Cheaper)
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
        result = await chain.ainvoke(docs)
    else:
        # Use 'map_reduce' for long text (Parallel & Scalable)
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=summary_prompt, combine_prompt=summary_prompt)
        # Create a proper config object instead of a dict
        config = RunnableConfig(max_concurrency=5)
        result = await chain.ainvoke(
            docs,
            config=config
        )

    summary = result["output_text"]
    # Return summary for UI and full_text for the hidden State (Q&A)
    return summary

def main():
    with gr.Blocks(title="AI Audio Summarizer") as demo:
        gr.Markdown("# 🎙️ Audio Summary Tool")
        gr.Markdown("Upload an audio file to get an instant AI-generated summary.")

        # Create a horizontal row
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                url_input = gr.Textbox(
                    label="Or paste a YouTube/Video URL",
                    placeholder="E.g, https://www.youtube.com/watch?v=123"
                )
                submit_btn = gr.Button("Generate Summary", variant="primary", interactive=False)

            # Right Column: Outputs
            with gr.Column(scale=1):
                summary_output = gr.Markdown(
                    label="The summary will appear here...",
                    
                )
                clear_btn = gr.ClearButton([audio_input, url_input, summary_output])

        # Connect the button to the function
        submit_btn.click(
            fn=process_url_and_audio,
            inputs=[audio_input, url_input],
            outputs=summary_output,
            show_progress="full"
        )

        def set_btn_interactive(audio, url):
            return gr.update(interactive=True if audio is not None or url else False)

        audio_input.change(
            fn=set_btn_interactive,
            inputs=[audio_input, url_input],
            outputs=submit_btn
        )
        url_input.change(
            fn=set_btn_interactive,
            inputs=[audio_input, url_input],
            outputs=submit_btn
        )

    demo.launch()


if __name__ == "__main__":
    main()
