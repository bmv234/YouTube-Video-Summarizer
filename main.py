import os
from dotenv import load_dotenv
import yt_dlp
from faster_whisper import WhisperModel
from openai import OpenAI
import ollama
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to GPT-4o if not specified
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

if LLM_PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create necessary directories
os.makedirs('downloads', exist_ok=True)
os.makedirs('summaries', exist_ok=True)
os.makedirs('transcriptions', exist_ok=True)

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
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

def transcribe_audio(audio_path):
    """Transcribe audio using Faster Whisper."""
    try:
        # First try with GPU
        if not os.getenv("FORCE_CPU", "false").lower() == "true":
            try:
                print("Attempting GPU transcription...")
                model = WhisperModel("base", device="cuda", compute_type="float16", download_root="models")
                segments, _ = model.transcribe(audio_path)
                return " ".join([segment.text for segment in segments])
            except Exception as e:
                print(f"GPU transcription failed: {str(e)}")
                print("Falling back to CPU...")
        
        # Use CPU if GPU failed or was forced off
        print("Using CPU for transcription...")
        model = WhisperModel("base", device="cpu", compute_type="int8", download_root="models")
        segments, _ = model.transcribe(audio_path)
        return " ".join([segment.text for segment in segments])
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise

def get_summary_openai(transcript, summary_type):
    """Get summary using OpenAI's models."""
    system_message = {
        "short": "You are a helpful assistant that provides concise video summaries. "
                "Create a brief summary that captures the main points in 2-3 paragraphs. "
                "Format your response in markdown with a brief bullet list of key takeaways.",
        "detailed": "You are a helpful assistant that provides comprehensive video summaries. "
                   "Create a detailed summary that thoroughly analyzes the content, including main points, "
                   "supporting details, and important context. Structure your response in markdown with "
                   "clear sections, subsections, and bullet points for better readability."
    }

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_message[summary_type]},
            {"role": "user", "content": transcript}
        ],
        max_tokens=16384
    )
    
    return response.choices[0].message.content

def get_summary_ollama(transcript, summary_type):
    """Get summary using Ollama."""
    prompts = {
        "short": f"""Please provide a brief summary of the following video transcript in 2-3 paragraphs. 
        Include a bullet list of key takeaways. Format your response in markdown:

        {transcript}""",
        "detailed": f"""Please provide a comprehensive summary of the following video transcript. 
        Include main points, supporting details, and important context. Structure your response in 
        markdown with clear sections, subsections, and bullet points for better readability:

        {transcript}"""
    }
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompts[summary_type],
        options={
            "num_predict": -1,
        }
    )
    
    return response['response']

def get_summary(transcript, video_title, video_url, summary_type):
    """Get summary based on configured LLM provider and format as markdown."""
    if LLM_PROVIDER == "openai":
        summary = get_summary_openai(transcript, summary_type)
    elif LLM_PROVIDER == "ollama":
        summary = get_summary_ollama(transcript, summary_type)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    
    # Format the complete markdown content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"""# {video_title}

## Video Information
- **URL:** {video_url}
- **Summarized:** {timestamp}
- **LLM Provider:** {LLM_PROVIDER.upper()}
{f"- **Model:** {OLLAMA_MODEL}" if LLM_PROVIDER == "ollama" else f"- **Model:** {OPENAI_MODEL}"}
- **Summary Type:** {"Short" if summary_type == "short" else "Detailed"}

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

def get_summary_type():
    """Get the user's preferred summary type."""
    while True:
        print("\nPlease choose a summary type:")
        print("1. Short (2-3 paragraphs with key takeaways)")
        print("2. Detailed (Comprehensive analysis with sections)")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return "short"
        elif choice == "2":
            return "detailed"
        else:
            print("Invalid choice. Please enter 1 or 2.")

def main():
    try:
        print(f"Using LLM Provider: {LLM_PROVIDER}")
        if LLM_PROVIDER == "openai":
            print(f"OpenAI Model: {OPENAI_MODEL}")
        else:
            print(f"Ollama Model: {OLLAMA_MODEL}")
        print(f"Force CPU: {os.getenv('FORCE_CPU', 'false')}")
        
        # Get YouTube URL from user
        url = input("Please enter the YouTube video URL: ")
        
        # Get summary type preference
        summary_type = get_summary_type()
        
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
        summary = get_summary(transcript, video_title, url, summary_type)
        
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
