import os
from dotenv import load_dotenv
import yt_dlp
from faster_whisper import WhisperModel
import openai
import ollama
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import torch
from datetime import datetime
import ctypes
from ctypes.util import find_library

# Load environment variables
load_dotenv()

# Configure LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
if LLM_PROVIDER == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Transcription settings
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

# Create necessary directories
os.makedirs('downloads', exist_ok=True)
os.makedirs('summaries', exist_ok=True)
os.makedirs('transcriptions', exist_ok=True)

def check_cudnn():
    """Check if cuDNN is available."""
    try:
        cudnn_paths = [
            'libcudnn.so.8',
            'libcudnn.so.7',
            'libcudnn.so'
        ]
        for path in cudnn_paths:
            if find_library(path):
                return True
        return False
    except Exception:
        return False

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    raise ValueError("Invalid YouTube URL")

def get_youtube_transcript(video_id):
    """Get transcript directly from YouTube if available."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript_list])
    except Exception:
        return None

def download_youtube_video(url):
    """Download YouTube video and return the path to the audio file and video title."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloads/%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = os.path.join('downloads', f"{info['title']}.mp3")
        return audio_path, info['title']

def is_gpu_available():
    """Check if CUDA GPU is available and working with cuDNN."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check CUDA
        torch.cuda.init()
        # Create a small tensor on GPU using the recommended method
        test_tensor = torch.tensor([1.0], device='cuda')
        # Check cuDNN
        if not check_cudnn():
            print("CUDA is available but cuDNN is not found. Falling back to CPU.")
            return False
        # If we got here, both CUDA and cuDNN are working
        return True
    except Exception as e:
        print(f"GPU check failed: {str(e)}")
        return False

def transcribe_audio(audio_path):
    """Transcribe audio using Faster Whisper."""
    # Check if we should use GPU
    use_gpu = not FORCE_CPU and is_gpu_available()
    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "int8"
    
    print(f"Transcribing using: {'GPU' if use_gpu else 'CPU'}")
    
    # Load the Whisper model
    model = WhisperModel("base", device=device, compute_type=compute_type)
    
    # Transcribe the audio
    segments, _ = model.transcribe(audio_path)
    
    # Combine all segments into one text
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def get_summary_openai(transcript):
    """Get summary using OpenAI's GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise video summaries. "
                                        "Please summarize the following transcript in a clear and organized way, "
                                        "highlighting the main points and key takeaways. "
                                        "Format your response in markdown with appropriate headers and bullet points."},
            {"role": "user", "content": transcript}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message['content']

def get_summary_ollama(transcript):
    """Get summary using Ollama."""
    prompt = f"""Please provide a concise summary of the following video transcript, 
    highlighting the main points and key takeaways. Format your response in markdown with 
    appropriate headers and bullet points:

    {transcript}"""
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "num_predict": 500,  # Similar to max_tokens but for Ollama
        }
    )
    
    return response['response']

def get_summary(transcript, video_title, video_url):
    """Get summary based on configured LLM provider and format as markdown."""
    if LLM_PROVIDER == "openai":
        summary = get_summary_openai(transcript)
    elif LLM_PROVIDER == "ollama":
        summary = get_summary_ollama(transcript)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    
    # Format the complete markdown content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"""# {video_title}

## Video Information
- **URL:** {video_url}
- **Summarized:** {timestamp}
- **LLM Provider:** {LLM_PROVIDER.upper()}
{f"- **Model:** {OLLAMA_MODEL}" if LLM_PROVIDER == "ollama" else "- **Model:** GPT-4"}

## Summary
{summary}
"""
    return markdown_content

def save_text_file(content, folder, filename, extension="txt"):
    """Save content to a text file with timestamp."""
    # Sanitize the filename
    safe_filename = sanitize_filename(filename)
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{safe_filename}_{timestamp}.{extension}"
    # Create full path
    filepath = os.path.join(folder, full_filename)
    
    # Save the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath

def cleanup_downloads():
    """Clean up downloaded files."""
    if os.path.exists('downloads'):
        for file in os.listdir('downloads'):
            os.remove(os.path.join('downloads', file))

def main():
    try:
        print(f"Using LLM Provider: {LLM_PROVIDER}")
        if LLM_PROVIDER == "ollama":
            print(f"Ollama Model: {OLLAMA_MODEL}")
        print(f"Force CPU: {FORCE_CPU}")
        
        # Get YouTube URL from user
        url = input("Please enter the YouTube video URL: ")
        
        # First try to get transcript directly from YouTube
        video_id = get_video_id(url)
        print("Checking for existing YouTube transcription...")
        transcript = get_youtube_transcript(video_id)
        
        # Get video info for filename
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            video_info = ydl.extract_info(url, download=False)
            video_title = video_info['title']
        
        if transcript:
            print("Found existing YouTube transcription!")
        else:
            print("No existing transcription found. Downloading and transcribing video...")
            audio_path, video_title = download_youtube_video(url)
            transcript = transcribe_audio(audio_path)
        
        # Save transcription
        trans_path = save_text_file(transcript, 'transcriptions', video_title)
        print(f"Transcription saved to: {trans_path}")
        
        print("Generating summary...")
        summary = get_summary(transcript, video_title, url)
        
        # Save summary as markdown
        summary_path = save_text_file(summary, 'summaries', f"{video_title}_summary", "md")
        print(f"Summary saved to: {summary_path}")
        
        print("\n=== Video Summary ===")
        print(summary)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up downloaded files
        cleanup_downloads()

if __name__ == "__main__":
    main()
