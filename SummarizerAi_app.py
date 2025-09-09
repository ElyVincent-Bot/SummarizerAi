import requests
import streamlit as st
import os
from dotenv import load_dotenv
from SummarizerAi import summarize_youtube_video
from openai import OpenAI

BACKEND_URL = "https://summarizerai.railway.internal"

def summarize_video(video_url: str):
    try:
        response = requests.post(
            f"{BACKEND_URL}/summarize",
            json={"url": video_url}
        )
        response.raise_for_status()
        return response.json().get("summary", "No summary returned.")
    except Exception as e:
        st.error(f"Error contacting backend: {e}")
        return None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.title("SummarizerAi - YouTube Video Summarizer")

# Whisper model selection
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
    if youtube_urls.strip() == "":
        st.warning("Please enter at least one YouTube URL.")
    else:
        urls = [url.strip() for url in youtube_urls.split("\n") if url.strip()]
        for url in urls:
            st.info(f"Processing: {url}")
            try:
                transcript = summarize_youtube_video(url, model_name=model_option)

                # Summarize using OpenAI
                prompt = f"Summarize this text concisely:\n\n{transcript}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.choices[0].message.content
                st.success("Summary ready!")
                st.write(summary)
            except Exception as e:
                st.error(f"Error processing {url}: {e}")
