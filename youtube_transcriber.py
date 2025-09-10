import os
import re
import torch
import whisper
from youtube_transcript_api import YouTubeTranscriptApi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "small"
_model_cache = {}

def get_whisper_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name, device=DEVICE)
    return _model_cache[model_name]

# -------------------------
# Transcript fetching
# -------------------------
def fetch_transcript(video_id, language="en"):
    """
    Fetch transcript from YouTubeTranscriptApi. Returns text or None.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language]).fetch()
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        print(f"[Transcript API] {e}")
        return None

# -------------------------
# Audio fallback (optional)
# -------------------------
def download_audio(url, folder="audio_cache"):
    """
    Fallback to download audio only if transcript is missing.
    """
    from yt_dlp import YoutubeDL

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
    model_name = model_override or DEFAULT_MODEL
    m = get_whisper_model(model_name)
    result = m.transcribe(file_path)
    return result["text"]

# -------------------------
# Main summarization function
# -------------------------
def summarize_youtube_video(url, model_name=DEFAULT_MODEL):
    """
    Returns transcript text for a YouTube video.
    Uses transcript API first, then falls back to audio transcription.
    """
    # Extract video ID
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
