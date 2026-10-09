import os
import re

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎬",
    layout="centered",
)

# ── API config ─────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Matches youtube.com/watch?v=ID and youtu.be/ID
_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    """Return the 11-char video ID from a YouTube URL, or None if not matched."""
    m = _YOUTUBE_RE.search(url)
    return m.group(1) if m else None


# ── API helpers ────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_transcript_info(url: str) -> dict | None:
    """
    Call GET /transcript-info — fast, no model inference.
    Returns the parsed dict or None on any error (silent; used for preview only).
    """
    try:
        r = requests.get(f"{API_URL}/transcript-info", params={"url": url}, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def summarize_video(url: str, min_length: int, max_length: int) -> dict:
    """
    Call POST /summarize-video.
    Raises ValueError for 422 validation errors, RuntimeError for other failures.
    """
    response = requests.post(
        f"{API_URL}/summarize-video",
        json={"url": url, "min_length": min_length, "max_length": max_length},
        timeout=600,
    )
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 422:
        detail = response.json().get("detail", "Invalid input.")
        if isinstance(detail, list):
            raise ValueError("; ".join(err.get("msg", str(err)) for err in detail))
        raise ValueError(detail)
    else:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🎬 YouTube Video Summarizer")
st.caption("Paste a YouTube URL to get an AI-generated summary of the video transcript.")

# Health indicator
if check_api_health():
    st.success("API is online", icon="✅")
else:
    st.error(
        f"Cannot reach the API at {API_URL}. "
        "Make sure the FastAPI backend is running with `uvicorn backend.app:app --reload`.",
        icon="🔴",
    )

st.divider()

# ── URL input ──────────────────────────────────────────────────────────────────

# Pre-populate from example button if one was clicked
default_url = st.session_state.pop("example_url", "")

url_input = st.text_input(
    "YouTube URL",
    value=default_url,
    placeholder="https://www.youtube.com/watch?v=...",
)

# ── Thumbnail preview ──────────────────────────────────────────────────────────
# Shown as soon as the URL contains a valid video ID — no API call needed,
# YouTube thumbnails are publicly accessible at a fixed URL pattern.

video_id = extract_video_id(url_input)

if video_id:
    thumb_col, info_col = st.columns([1, 2])
    with thumb_col:
        st.image(
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            use_container_width=True,
        )
    with info_col:
        st.markdown(f"**Video ID:** `{video_id}`")
        st.markdown(
            f"[Open on YouTube ↗](https://www.youtube.com/watch?v={video_id})"
        )
        # Fetch transcript metadata so we can show chunk count before summarizing
        info = get_transcript_info(url_input)
        if info:
            st.markdown(
                f"**Transcript:** {info['transcript_char_count']:,} chars · "
                f"**Chunks:** {info['chunk_count']}"
            )
            if info["chunk_count"] > 1:
                st.caption(
                    f"This video will be summarized in {info['chunk_count']} chunks. "
                    "Longer videos take more time."
                )

    st.divider()

# ── Controls ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    min_length = st.slider(
        "Min summary length (tokens)", min_value=10, max_value=100, value=30, step=5
    )
with col2:
    max_length = st.slider(
        "Max summary length (tokens)", min_value=50, max_value=300, value=130, step=10
    )

# ── Summarize button ───────────────────────────────────────────────────────────

if st.button("Summarize", type="primary", use_container_width=True):
    if not url_input.strip():
        st.warning("Please enter a YouTube URL.")
    elif not video_id:
        st.warning("That doesn't look like a valid YouTube URL.")
    elif min_length >= max_length:
        st.warning("Min length must be less than max length.")
    else:
        # st.status gives a collapsible step-by-step progress block
        with st.status("Working…", expanded=True) as status:
            st.write("📡 Fetching transcript…")
            try:
                result = summarize_video(url_input.strip(), min_length, max_length)

                chunk_count = result["chunk_count"]
                st.write(
                    f"✂️ Transcript split into {chunk_count} chunk{'s' if chunk_count != 1 else ''}."
                )
                st.write(
                    f"🤖 Summarizing {chunk_count} chunk{'s' if chunk_count != 1 else ''} "
                    f"({'+ second-pass' if chunk_count > 1 else 'single pass'})…"
                )
                status.update(label="Done!", state="complete", expanded=False)

                # ── Final summary ──────────────────────────────────────────────
                st.subheader("📝 Summary")
                st.write(result["final_summary"])

                # ── Stats ──────────────────────────────────────────────────────
                st.caption(
                    f"Video ID: `{result['video_id']}` · "
                    f"Transcript: {result['transcript_char_count']:,} chars · "
                    f"Chunks: {result['chunk_count']}"
                )

                # ── Per-chunk details ──────────────────────────────────────────
                if result["chunk_count"] > 1:
                    with st.expander(
                        f"View per-chunk summaries ({result['chunk_count']} chunks)"
                    ):
                        for cs in result["chunk_summaries"]:
                            st.markdown(
                                f"**Chunk {cs['chunk_index'] + 1}** "
                                f"({cs['chunk_char_count']:,} chars)"
                            )
                            st.write(cs["summary"])
                            st.divider()

            except ValueError as e:
                status.update(label="Validation error", state="error", expanded=False)
                st.error(str(e))
            except RuntimeError as e:
                status.update(label="API error", state="error", expanded=False)
                st.error(str(e))
            except requests.exceptions.Timeout:
                status.update(label="Timed out", state="error", expanded=False)
                st.error(
                    "Request timed out. The video may be very long or the model is still "
                    "loading — try again in a moment."
                )
            except requests.exceptions.ConnectionError:
                status.update(label="Connection lost", state="error", expanded=False)
                st.error(
                    "Lost connection to the API. Make sure the FastAPI backend is still running."
                )

# ── Example videos ─────────────────────────────────────────────────────────────

EXAMPLES = [
    ("Python in 100 Seconds", "https://www.youtube.com/watch?v=x7X9w_GIm1s"),
    ("What is Machine Learning?", "https://www.youtube.com/watch?v=ukzFI9rgwfU"),
    ("How the Internet Works", "https://www.youtube.com/watch?v=x3c1ih2NJEg"),
]

with st.expander("Try an example video"):
    for label, url in EXAMPLES:
        if st.button(label, key=url):
            st.session_state["example_url"] = url
            st.rerun()
