# 🎙️ AI Audio Summarizer

This project provides an AI-powered tool to generate bulleted summaries from audio files and YouTube video links. It leverages state-of-the-art speech-to-text transcription (Whisper model) and large language models (GPT-4o-mini via LangChain) to distill key information from audio content.

## ✨ Features

*   **Audio Upload:** Summarize local audio files.
*   **YouTube URL Processing:** Extract audio from YouTube videos and generate summaries.
*   **AI Transcription:** Uses OpenAI's Whisper model for accurate speech-to-text.
*   **LLM Summarization:** Employs `gpt-4o-mini` via LangChain for concise, bullet-point summaries.
*   **Gradio Web Interface:** Easy-to-use web UI for interaction.
*   **Dockerized Deployment:** Portable and consistent environment setup using Docker Compose.
*   **Persistent Model Caching:** Efficiently caches AI models to avoid repeated downloads in Docker.

## 🚀 Getting Started

These instructions will get your project up and running locally using Docker Compose.

### Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kartikkh1/SoundBite
    cd audio-summary
    ```
2.  **Environment Variables:**
    Create a `.env` file in the root of your project (same directory as `compose.yaml`) and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    **Important:** Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

3.  **Build and Run with Docker Compose:**
    Navigate to your project's root directory in your terminal and run the following command to build the Docker image and start the application:
    ```bash
    docker compose up --build
    ```
    *   The `--build` flag ensures that Docker Compose rebuilds the image, incorporating any changes and downloading necessary dependencies (including `ffmpeg` and Python packages).
    *   During the **first run**, the Whisper AI model will be downloaded and cached. This might take some time depending on your internet connection. Subsequent runs will use the cached model, speeding up startup.

## 🌐 Usage

Once the Docker container is running, open your web browser and navigate to:

```
http://127.0.0.1:7860
```

You can then:
*   **Upload an audio file:** Use the "Upload Audio" component to select a local audio file.
*   **Enter a YouTube URL:** Paste a valid YouTube link into the "YouTube URL" textbox.
*   Click the "Generate Summary" button. The application will then transcribe the audio and provide a bulleted summary.

You will see status updates on the interface (e.g., "Loading AI Model...", "Transcribing...", "Summarizing...") as the process unfolds.

## 🛑 Stopping the Application

To stop the Docker containers, press `Ctrl+C` in the terminal where `docker compose up` is running.

To stop and remove the containers (but keep the image and local `hf_cache` data), use:
```bash
docker compose down
```
