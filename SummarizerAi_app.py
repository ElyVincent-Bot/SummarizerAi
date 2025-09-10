import re
import os
import streamlit as st
import torch
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# Load OpenAI key
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Whisper setup
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "small"
_model_cache = {}

def get_whisper_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name, device=DEVICE)
    return _model_cache[model_name]

# -------------------------
# YouTube transcript
# -------------------------
def fetch_transcript(video_id, language="en"):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language]).fetch()
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        print(f"[Transcript API] {e}")
        return None

# -------------------------
# Audio fallback with Whisper
# -------------------------
def download_audio(url, folder="audio_cache"):
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
