"""
FAISS vector store management.

Mirrors the "Embedding Generation and Storing in Vector Store" step from
the original notebook:
    embeddings = OpenAIEmbeddings(...)
    vector_store = FAISS.from_documents(chunks, embeddings)

...but swapped to a free local Hugging Face embedding model, and extended
with disk persistence
so the index survives app restarts, plus incremental add_documents for
uploading multiple files across a session.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

_INDEX_NAME = "index"

# Cached at module level so the (fairly small) embedding model is only
# loaded into memory once per process, not once per call.
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local, free, no-API-limit embedding model. Downloads once on first
    run (a few hundred MB) then runs entirely on your own CPU — no network
    calls, so it can never hit a provider rate limit."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


def _index_files_exist() -> bool:
    faiss_file = settings.vectorstore_dir / f"{_INDEX_NAME}.faiss"
    pkl_file = settings.vectorstore_dir / f"{_INDEX_NAME}.pkl"
    return faiss_file.exists() and pkl_file.exists()


def load_vectorstore() -> FAISS | None:
    """Load a previously persisted FAISS index from disk, if one exists."""
    if not _index_files_exist():
        return None
    try:
        vs = FAISS.load_local(
            str(settings.vectorstore_dir),
            get_embeddings(),
            index_name=_INDEX_NAME,
            # Safe here: the index was written by this same app, not
            # an untrusted third party.
            allow_dangerous_deserialization=True,
        )
        log.info("Loaded existing FAISS index from %s", settings.vectorstore_dir)
        return vs
    except Exception:
        log.exception("Failed to load existing FAISS index; starting fresh.")
        return None


def save_vectorstore(vs: FAISS) -> None:
    vs.save_local(str(settings.vectorstore_dir), index_name=_INDEX_NAME)
    log.info("Persisted FAISS index to %s", settings.vectorstore_dir)


def build_or_extend_vectorstore(chunks: list[Document], existing: FAISS | None) -> FAISS:
    """Create a new FAISS index from chunks, or add chunks to an existing one."""
    if not chunks:
        raise ValueError("No chunks to index.")

    if existing is None:
        vs = FAISS.from_documents(chunks, get_embeddings())
        log.info("Created new FAISS index with %d chunks", len(chunks))
    else:
        existing.add_documents(chunks)
        vs = existing
        log.info("Added %d chunks to existing FAISS index", len(chunks))

    save_vectorstore(vs)
    return vs


def delete_vectorstore() -> None:
    """Wipe the persisted index (used by a 'reset knowledge base' action)."""
    for ext in (".faiss", ".pkl"):
        f = settings.vectorstore_dir / f"{_INDEX_NAME}{ext}"
        if f.exists():
            f.unlink()
    log.info("Deleted FAISS index files.")


def get_retriever(vs: FAISS, k: int | None = None):
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.retriever_top_k},
    )
