# Workhair AI (Luna)

Streamlit chatbot with RAG for Workhair salon.

## Requirements
- Docker + Docker Compose
- OpenRouter API key

## Setup
1) Create `.env` from `.env.example`
2) Set `OPENROUTER_API_KEY`
3) Start the app

```bash
docker compose up -d --build
```

Open http://localhost:8501

## Environment Variables
- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (optional)
- `OPENROUTER_TIMEOUT` (optional)
- `OPENROUTER_MAX_RETRIES` (optional)
- `RAG_TOP_K` (optional)
- `RAG_CACHE_DIR` (optional)
- `LOG_LEVEL` (optional)

## Healthcheck
Dockerfile includes a healthcheck at `/_stcore/health`.
