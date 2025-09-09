from fastapi import APIRouter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class SummarizeRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"

@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
        )
        summary = response.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-streamlit-app.streamlit.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

class VideoRequest(BaseModel):
    url: str

@router.post("/summarize")
async def summarize(request: VideoRequest):
    summary = summarize_youtube_video(request.url)
    return {"summary": summary}
