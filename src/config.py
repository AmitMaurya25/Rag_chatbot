"""
Centralized configuration for the RAG chatbot.

All environment-dependent values (API keys, model names, paths, tuning
knobs) are read here once, so the rest of the app never touches
os.environ directly. This makes it easy to swap models, change chunking
strategy, or point at a different vector store location without hunting
through the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root (parent of the src/ folder) regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root (no-op if the file doesn't exist).
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- API ---
    # Get a free key from https://console.groq.com/keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # --- Models ---
    # Chat model served via Groq's API (generous free tier, fast, stable).
    # gpt-oss-20b is Groq's current recommended free-tier default as of
    # their June 2026 deprecation of the older Llama 3.x free models.
    # Full current list: https://console.groq.com/docs/models
    chat_model: str = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    # Local embedding model (runs on your CPU, no API calls, no rate limit).
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Retrieval ---
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "4"))

    # --- Paths ---
    vectorstore_dir: Path = PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "data/vectorstore")
    uploads_dir: Path = PROJECT_ROOT / os.getenv("UPLOADS_DIR", "data/uploads")
    log_dir: Path = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

    # --- Misc ---
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))


settings = Settings()

# Ensure runtime directories exist.
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)


def validate_settings() -> list[str]:
    """Returns a list of human-readable problems with the current config.
    Empty list means the app is safe to run."""
    problems = []
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        problems.append(
            "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
            "with a free key from https://console.groq.com/keys"
        )
    return problems
