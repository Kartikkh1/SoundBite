from accelerate import Accelerator
from transformers import pipeline
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableConfig

load_dotenv(override=True)
llm = ChatOpenAI(temperature=0, model="gpt-4o")

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
        chain = load_summarize_chain(llm, chain_type="stuff")
        result = await chain.ainvoke(docs)
    else:
        # Use 'map_reduce' for long text (Parallel & Scalable)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        # Create a proper config object instead of a dict
        config = RunnableConfig(max_concurrency=5)
        result = await chain.ainvoke(
            docs,
            config=config
        )

    summary = result["output_text"]
    print(summary)

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
                submit_btn = gr.Button("Generate Summary", variant="primary", interactive=False)

            # Right Column: Outputs
            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=10,
                    placeholder="The summary will appear here..."
                )
                clear_btn = gr.ClearButton([audio_input, summary_output])

        # Connect the button to the function
        submit_btn.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=summary_output,
            show_progress="full"
        )

        # Use a lambda to toggle interactivity in one line
        audio_input.change(
            fn=lambda x: gr.update(interactive=True if x is not None else False),
            inputs=audio_input,
            outputs=submit_btn
        )

    demo.launch()


if __name__ == "__main__":
    main()
