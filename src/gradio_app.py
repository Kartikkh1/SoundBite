import os
import gradio as gr
from src.audio_processing import VideoTooLongError, is_valid_url, extract_audio_from_url, process_audio


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

def run_app():
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

