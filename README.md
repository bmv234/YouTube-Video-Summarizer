# YouTube Video Summarizer

A Python application that downloads YouTube videos, transcribes them using Faster Whisper (with GPU acceleration), and generates summaries using either OpenAI's models or Ollama.

## Features

- Downloads YouTube videos using yt-dlp
- Attempts to use existing YouTube transcriptions when available
- Transcribes videos using Faster Whisper with GPU acceleration (falls back to CPU if needed)
- Generates summaries using either OpenAI models (GPT-4o by default) or Ollama
- Supports two summary modes:
  - Short: Concise 2-3 paragraph summary with key takeaways
  - Detailed: Comprehensive analysis with sections and subsections
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

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
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
# Examples: gpt-4o, gpt-4, gpt-3.5-turbo, gpt-4-1106-preview
OPENAI_MODEL=gpt-4o

# Ollama model name (required if using ollama)
# Examples: llama2, mistral, codellama, etc.
OLLAMA_MODEL=llama2

# Force CPU usage for transcription (optional)
# Set to "true" to force CPU usage, "false" or omit to auto-detect
FORCE_CPU=false
```

## Usage

### Linux/macOS:
```bash
./run.sh
# Or
python main.py
```

### Windows:
Right-click `run.ps1` and select "Run with PowerShell"  
Or open PowerShell and run:
```powershell
.\run.ps1
```

The script will:
1. Create and activate a virtual environment if needed
2. Install required packages if not already installed
3. Run the summarizer

When running, you will be prompted to:
1. Enter a YouTube URL
2. Choose a summary type:
   - Option 1: Short summary (2-3 paragraphs with key takeaways)
   - Option 2: Detailed summary (Comprehensive analysis with sections)

The script will then:
- Try to find an existing YouTube transcription
- If none exists, download the video and transcribe it
- Generate a summary using the configured LLM and chosen summary type
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

## Troubleshooting

### Windows:
If you encounter issues running the PowerShell script:
1. Make sure Python 3.12+ is installed and in your PATH
2. Ensure FFmpeg is installed and in your PATH
3. If using GPU acceleration, verify CUDA 12.1 and cuDNN 9.x are installed
4. Check that all required environment variables are set in your .env file
5. Try running PowerShell as Administrator if you encounter permission issues

### Linux/macOS:
1. Make sure the run.sh script is executable: `chmod +x run.sh`
2. Verify Python and FFmpeg are installed and in your PATH
3. Check CUDA installation if using GPU acceleration
4. Ensure your .env file contains the necessary configuration

## License

MIT License
