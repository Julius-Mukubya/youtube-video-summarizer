# YouTube Video Summarizer

Fetches the transcript of any YouTube video and summarizes it using
[sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) —
the same model used by the [text-summarizer](../text-summarizer) project.

```
youtube-video-summarizer/
├── backend/
│   ├── app.py              # FastAPI backend — transcript fetch, chunking, summarization
│   ├── requirements.txt    # Backend dependencies
│   ├── .env                # Local config (gitignored)
│   └── .env.example        # Config template (committed)
├── frontend/
│   ├── streamlit_app.py    # Streamlit UI
│   ├── requirements.txt    # Frontend dependencies
│   ├── .env                # Local config (gitignored)
│   └── .env.example        # Config template (committed)
├── .gitignore
└── README.md
```

---

## How it works

1. The frontend sends the YouTube URL to `POST /summarize-video` on the backend.
2. The backend extracts the video ID and fetches the transcript via
   `youtube-transcript-api` (no YouTube API key required — uses public captions).
3. The transcript is split into ~3 500-character chunks that fit within BART's
   ~1 024-token context window.
4. Each chunk is summarized individually.
5. The chunk summaries are combined. If the combined text is still too long,
   a second-pass summary is run over it to produce the final result.

---

## Backend (FastAPI)

### Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
source venv/bin/activate         # Mac/Linux

pip install -r backend/requirements.txt
```

> **PyTorch install note** — `backend/requirements.txt` includes
> `--index-url https://download.pytorch.org/whl/cpu` so pip downloads the
> CPU-only PyTorch wheel (~250 MB) instead of the full CUDA build (~2 GB).
> Remove that line and adjust the version spec if you need GPU support.

> **First-run download** — On the first startup, `transformers` downloads
> the model weights (~1.2 GB) to `~/.cache/huggingface/hub/`. Subsequent
> startups load from the local cache and are much faster.

### Configuration

Copy the example env file and edit as needed:

```bash
cp backend/.env.example backend/.env
```

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:8501` | Comma-separated list of allowed CORS origins |

### Running

```bash
uvicorn backend.app:app --reload
```

API runs at `http://127.0.0.1:8000`.
Interactive docs at `http://127.0.0.1:8000/docs`.

### Endpoints

#### `GET /health`

```json
{ "status": "ok", "model": "sshleifer/distilbart-cnn-12-6" }
```

#### `POST /summarize-video`

**Request body**

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "min_length": 30,
  "max_length": 130
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | string | Yes | — | YouTube video URL |
| `min_length` | integer | No | 30 | Min summary length per chunk in tokens (10–100) |
| `max_length` | integer | No | 130 | Max summary length per chunk in tokens (50–300) |

**Response**

```json
{
  "video_id": "dQw4w9WgXcQ",
  "transcript_char_count": 12400,
  "chunk_count": 4,
  "chunk_summaries": [
    { "chunk_index": 0, "chunk_char_count": 3482, "summary": "..." },
    ...
  ],
  "final_summary": "..."
}
```

**Example with curl**

```bash
curl -X POST http://127.0.0.1:8000/summarize-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=x7X9w_GIm1s"}'
```

---

## Frontend (Streamlit)

### Setup

```bash
pip install -r frontend/requirements.txt
```

### Configuration

```bash
cp frontend/.env.example frontend/.env
```

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend |

### Running

Make sure the backend is running first, then:

```bash
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`.

### Features

- API health indicator
- YouTube URL input with min/max summary length sliders
- Final summary with transcript stats (char count, chunk count)
- Expandable per-chunk summaries for longer videos
- Example video buttons to try immediately

---

## Stack

- [FastAPI](https://fastapi.tiangolo.com/) — REST API backend
- [Streamlit](https://streamlit.io/) — web frontend
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — transcript fetching (no API key needed)
- [Transformers](https://huggingface.co/docs/transformers) (v4) — summarization model
- [PyTorch](https://pytorch.org/) (CPU) — model inference
