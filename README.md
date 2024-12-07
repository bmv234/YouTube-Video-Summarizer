# YouTube Video Summarizer

This Python application summarizes YouTube videos by first checking for existing transcriptions, and if none are available, downloads the video and transcribes it using Faster Whisper. The summary can be generated using either OpenAI's GPT-4 or a local LLM via Ollama.

## Features

- Utilizes existing YouTube transcriptions when available
- Downloads YouTube videos using yt-dlp (only when necessary)
- Transcribes audio using Faster Whisper (with GPU support)
- Generates summaries using either:
  - OpenAI GPT-4 (cloud-based)
  - Local LLMs via Ollama (runs locally)
- Automatic file organization:
  - Saves transcriptions in `transcriptions/` folder
  - Saves summaries in `summaries/` folder
  - Temporary downloads in `downloads/` folder
- Smart GPU/CPU handling:
  - Automatically detects GPU availability
  - Can be forced to use CPU via environment variable
- Automatic cleanup of temporary files

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)
- CUDA-compatible GPU (optional, for faster transcription)
- OpenAI API key (if using OpenAI's GPT-4)
- Ollama (if using local LLMs)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd youtube-video-summarizer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (if using local LLMs):
   Visit [Ollama's website](https://ollama.ai) for installation instructions.

4. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Configure your settings in the `.env` file:
```bash
# Required only if using OpenAI (GPT-4)
OPENAI_API_KEY=your_api_key_here

# LLM choice: 'openai' or 'ollama'
LLM_PROVIDER=openai

# Ollama model name (required if using ollama)
# Examples: llama2, mistral, codellama, etc.
OLLAMA_MODEL=llama2

# Force CPU usage for transcription (optional)
# Set to "true" to force CPU usage, "false" or omit to auto-detect
FORCE_CPU=false
```

## Usage

Run the script using Python:
```bash
python main.py
```

The application will:
1. Prompt you for a YouTube video URL
2. Check if the video has existing transcriptions/captions
3. If transcriptions exist:
   - Use the existing transcriptions
4. If no transcriptions exist:
   - Download the video's audio
   - Transcribe the audio using Faster Whisper (GPU if available)
5. Generate a summary using your configured LLM
6. Save outputs:
   - Transcription saved to `transcriptions/[video_title]_[timestamp].txt`
   - Summary saved to `summaries/[video_title]_summary_[timestamp].txt`
7. Display the summary in the console
8. Clean up temporary downloaded files

## Output Organization

The project organizes its outputs in the following directory structure:

```
youtube-video-summarizer/
├── transcriptions/     # Stores video transcriptions
├── summaries/         # Stores video summaries
└── downloads/         # Temporary folder for downloaded files
```

Each file is saved with a timestamp and the video's title for easy reference.

## Performance

The application prioritizes efficiency by:
1. First checking for existing YouTube transcriptions
2. Only downloading and transcribing videos when necessary
3. Using GPU acceleration when available (unless forced to use CPU)
4. Automatically cleaning up temporary files
5. Organizing outputs in dedicated folders

## Note

- Make sure you have sufficient OpenAI API credits if using GPT-4
- If using Ollama, ensure you have adequate system resources for running the local LLM
- The quality and speed of summarization may vary depending on your chosen LLM provider and model
- Transcription speed depends on your hardware and whether GPU is available/used
