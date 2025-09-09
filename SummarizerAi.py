import os
import re
import torch
import whisper
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "small"

model = whisper.load_model(DEFAULT_MODEL, device=DEVICE)

def fetch_transcript(video_id, language="en"):
    """Try to get transcript from YouTube first"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language]).fetch()
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return None

def download_audio(url, folder="audio_cache"):
    """Download YouTube audio to a temp file"""
    os.makedirs(folder, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def transcribe_audio(file_path, model_override=None):
    """Transcribe audio with Whisper"""
    m = model if model_override is None else whisper.load_model(model_override, device=DEVICE)
    result = m.transcribe(file_path)
    return result["text"]

def summarize_youtube_video(url, model_name=DEFAULT_MODEL):
    """Full pipeline: transcript -> Whisper fallback"""
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)

    transcript = fetch_transcript(video_id)
    if transcript:
        print("Transcript found on YouTube")
    else:
        print("No transcript found. Downloading audio + transcribing...")
        audio_file = download_audio(url)
        transcript = transcribe_audio(audio_file, model_override=model_name)

    return transcript
