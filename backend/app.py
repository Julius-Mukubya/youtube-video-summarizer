import os
import re
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# BART context window is ~1 024 tokens.  At ~4 chars/token, 3 500 chars gives
# comfortable headroom after tokenizer overhead.
CHUNK_SIZE = 3_500

# After per-chunk summaries are combined, run a second-pass summary if the
# combined text still exceeds this threshold.
SECOND_PASS_THRESHOLD = CHUNK_SIZE


# ── Model loading ──────────────────────────────────────────────────────────────

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the summarization model once at startup; release on shutdown."""
    print("Loading summarization model…")
    ml_models["summarizer"] = pipeline("summarization", model=MODEL_NAME)
    print("Model ready.")
    yield
    ml_models.clear()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube Video Summarizer API",
    description="Fetch a YouTube transcript and summarize it using distilbart-cnn-12-6.",
    version="1.0.0",
    lifespan=lifespan,
)

_raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8501")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str:
    """Return the 11-character video ID from a YouTube URL, or raise ValueError."""
    match = _YOUTUBE_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract a video ID from: {url!r}")
    return match.group(1)


def fetch_transcript(video_id: str) -> str:
    """
    Fetch the transcript for *video_id* and return it as a single string.

    Uses the v1 YouTubeTranscriptApi — instantiate, then call .fetch() with a
    language priority list.  Prefers manually created captions; the library
    handles that preference internally when using .list() + find_transcript().
    Raises HTTPException with appropriate status codes on failure.
    """
    try:
        ytt = YouTubeTranscriptApi()
        # list() returns a TranscriptList; find_transcript() prefers manual
        # captions over auto-generated ones automatically.
        transcript_list = ytt.list(video_id)
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        fetched = transcript.fetch()
        # Each snippet exposes a .text attribute in v1.
        return " ".join(snippet.text for snippet in fetched)
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=422,
            detail="Transcripts are disabled for this video.",
        )
    except VideoUnavailable:
        raise HTTPException(
            status_code=422,
            detail="The video is unavailable or private.",
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=422,
            detail="No English transcript found for this video.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch transcript: {exc}",
        )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split *text* into chunks of at most *chunk_size* characters, breaking on
    sentence boundaries ('. ') where possible to avoid cutting mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        # Try to break at the last sentence boundary within the window.
        boundary = text.rfind(". ", 0, chunk_size)
        cut = (boundary + 2) if boundary != -1 else chunk_size
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


def summarize_text(text: str, min_length: int, max_length: int) -> str:
    """Run the model on a single chunk and return the summary string."""
    result = ml_models["summarizer"](
        text,
        min_length=min_length,
        max_length=max_length,
        do_sample=False,
    )
    return result[0]["summary_text"]


# ── Schemas ────────────────────────────────────────────────────────────────────

class TranscriptInfoResponse(BaseModel):
    video_id: str
    transcript_char_count: int
    chunk_count: int
    thumbnail_url: str


class SummarizeVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    min_length: int = Field(30, ge=10, le=100, description="Min summary length per chunk (tokens)")
    max_length: int = Field(130, ge=50, le=300, description="Max summary length per chunk (tokens)")

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        if not _YOUTUBE_RE.search(v):
            raise ValueError("Must be a valid YouTube URL (youtube.com/watch?v=... or youtu.be/...)")
        return v


class ChunkSummary(BaseModel):
    chunk_index: int
    chunk_char_count: int
    summary: str


class SummarizeVideoResponse(BaseModel):
    video_id: str
    transcript_char_count: int
    chunk_count: int
    chunk_summaries: List[ChunkSummary]
    final_summary: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """Check that the API and model are running."""
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/transcript-info", response_model=TranscriptInfoResponse, tags=["Summarization"])
def transcript_info(url: str):
    """
    Fetch the transcript for a YouTube video and return metadata only —
    no summarization is performed.  Use this to show the user how many chunks
    their video will be split into before they commit to the full summarize call.
    """
    video_id = extract_video_id(url)
    transcript = fetch_transcript(video_id)
    chunks = chunk_text(transcript)
    return TranscriptInfoResponse(
        video_id=video_id,
        transcript_char_count=len(transcript),
        chunk_count=len(chunks),
        thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    )


@app.post("/summarize-video", response_model=SummarizeVideoResponse, tags=["Summarization"])
def summarize_video(request: SummarizeVideoRequest):
    """
    Fetch a YouTube transcript and return a summary.

    The transcript is split into chunks that fit within the model's context
    window.  Each chunk is summarized individually, then the chunk summaries
    are combined and passed through a second-pass summary if they are still
    too long to fit in a single model call.

    - **url**: YouTube video URL
    - **min_length**: minimum token length for each chunk summary (default 30)
    - **max_length**: maximum token length for each chunk summary (default 130)
    """
    if request.min_length >= request.max_length:
        raise HTTPException(
            status_code=422,
            detail="min_length must be less than max_length.",
        )

    video_id = extract_video_id(request.url)
    transcript = fetch_transcript(video_id)

    if len(transcript.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Transcript is too short to summarize (fewer than 50 characters).",
        )

    chunks = chunk_text(transcript)

    chunk_summaries: List[ChunkSummary] = []
    for i, chunk in enumerate(chunks):
        summary = summarize_text(chunk, request.min_length, request.max_length)
        chunk_summaries.append(
            ChunkSummary(chunk_index=i, chunk_char_count=len(chunk), summary=summary)
        )

    # Combine chunk summaries for the final summary.
    combined = " ".join(cs.summary for cs in chunk_summaries)

    # Second-pass summary if the combined text is still too long.
    if len(combined) > SECOND_PASS_THRESHOLD:
        second_pass_chunks = chunk_text(combined)
        second_pass_summaries = [
            summarize_text(c, request.min_length, request.max_length)
            for c in second_pass_chunks
        ]
        final_summary = " ".join(second_pass_summaries)
    else:
        final_summary = (
            summarize_text(combined, request.min_length, request.max_length)
            if len(chunks) > 1
            else chunk_summaries[0].summary
        )

    return SummarizeVideoResponse(
        video_id=video_id,
        transcript_char_count=len(transcript),
        chunk_count=len(chunks),
        chunk_summaries=chunk_summaries,
        final_summary=final_summary,
    )
