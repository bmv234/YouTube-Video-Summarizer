# YouTube Video Summarizer

A Python application that downloads YouTube videos, transcribes them using Faster Whisper (with GPU acceleration), and generates summaries using either OpenAI's models or Ollama.

## Features

- Downloads YouTube videos using yt-dlp
- Attempts to use existing YouTube transcriptions when available
- Transcribes videos using Faster Whisper with GPU acceleration (falls back to CPU if needed)
- Generates summaries using either OpenAI models (GPT-4o by default) or Ollama
- Organizes transcriptions and summaries with timestamps
- Cleans up downloaded files automatically

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional, for faster transcription)
- CUDA 12.1 and cuDNN 9.x installed (for GPU acceleration)
- FFmpeg (for audio extraction)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-video-summarizer.git
cd youtube-video-summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
# Required only if using OpenAI
OPENAI_API_KEY=your_api_key_here

# LLM choice: 'openai' or 'ollama'
LLM_PROVIDER=openai

# OpenAI model choice (optional, defaults to gpt-4o)
# Examples: gpt-40, gpt-4, gpt-3.5-turbo, gpt-4-1106-preview
OPENAI_MODEL=gpt-4o

# Ollama model name (required if using ollama)
# Examples: llama2, mistral, codellama, etc.
OLLAMA_MODEL=llama2

# Force CPU usage for transcription (optional)
# Set to "true" to force CPU usage, "false" or omit to auto-detect
FORCE_CPU=false
```

## Usage

1. Run the script:
```bash
python main.py
```

2. Enter a YouTube URL when prompted

3. The script will:
   - Try to find an existing YouTube transcription
   - If none exists, download the video and transcribe it
   - Generate a summary using the configured LLM
   - Save both transcription and summary to timestamped files
   - Display the summary in the terminal

## Output Files

- Transcriptions are saved in the `transcriptions/` directory
- Summaries are saved in the `summaries/` directory as Markdown files
- Downloaded files are automatically cleaned up after processing

## LLM Providers

### OpenAI
- Requires an API key
- Supports multiple models (GPT-4o by default)
- Provides high-quality summaries
- Costs money per API call

### Ollama
- Free and runs locally
- Requires Ollama to be installed and running
- Quality depends on the chosen model
- Supports various models (llama2, mistral, etc.)

## GPU Acceleration

The transcription process will automatically use GPU acceleration if:
1. A CUDA-capable GPU is available
2. CUDA 12.1 and cuDNN 9.x are properly installed
3. The `FORCE_CPU` environment variable is not set to "true"

If GPU acceleration fails or is unavailable, the script will automatically fall back to CPU transcription.

## License

MIT License
