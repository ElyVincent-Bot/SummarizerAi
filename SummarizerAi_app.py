import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import whisper
from openai import OpenAI
import os
from dotenv import load_dotenv

# -------------------------
# Load OpenAI key from environment / Streamlit secrets
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
# YouTube transcript fetching
# -------------------------
def fetch_transcript(video_id, language="en"):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language]).fetch()
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        print(f"[Transcript API] {e}")
        return None

def summarize_youtube_video(url, model_name=DEFAULT_MODEL):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)

    transcript = fetch_transcript(video_id)
    if transcript:
        print("Transcript found on YouTube.")
    else:
        print("No transcript found for this video.")
        transcript = ""

    return transcript

# -------------------------
# Streamlit App
# -------------------------
st.title("Summarizer AI - YouTube Video Summarizer")

# Whisper model selection (optional, for future audio transcription)
model_option = st.selectbox(
    "Select Whisper model (smaller=faster, larger=more accurate)",
    ["tiny", "base", "small", "medium", "large"],
    index=2
)

# Input URLs (one per line)
youtube_urls = st.text_area(
    "Paste YouTube URLs here (one per line):",
    placeholder="https://www.youtube.com/watch?v=example1\nhttps://www.youtube.com/watch?v=example2"
)

if st.button("Summarize"):
    if not youtube_urls.strip():
        st.warning("Please enter at least one YouTube URL.")
    else:
        urls = [url.strip() for url in youtube_urls.split("\n") if url.strip()]
        for url in urls:
            st.info(f"Processing: {url}")
            try:
                transcript_text = summarize_youtube_video(url, model_name=model_option)

                if not transcript_text:
                    st.warning(f"No transcript available for {url}. Skipping.")
                    continue

                # Summarize using OpenAI
                prompt = f"Summarize this text concisely:\n\n{transcript_text}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.choices[0].message.content
                st.success("Summary ready!")
                st.write(summary)

            except Exception as e:
                st.error(f"Error processing {url}: {e}")