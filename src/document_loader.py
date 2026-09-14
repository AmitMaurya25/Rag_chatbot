"""
Document ingestion: load PDFs / DOCX / TXT files uploaded by the user and
split them into chunks ready for embedding.

This generalizes the "Indexing (Text Splitting)" step from the original
notebook (RecursiveCharacterTextSplitter) to work on real documents
instead of a single YouTube transcript string.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


def load_file(file_path: Path) -> list[Document]:
    """Load a single file on disk into a list of LangChain Documents."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise UnsupportedFileType(
            f"'{suffix}' is not supported. Allowed types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_path.name
    log.info("Loaded %d page(s)/section(s) from %s", len(docs), file_path.name)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents using the same RecursiveCharacterTextSplitter strategy
    as the original notebook, with size/overlap pulled from config."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    log.info("Split into %d chunk(s)", len(chunks))
    return chunks


def ingest_file(file_path: Path) -> list[Document]:
    """Load + split a single uploaded file in one call."""
    docs = load_file(file_path)
    return split_documents(docs)


def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """Persist an uploaded file (from Streamlit's file_uploader) to disk
    under the configured uploads directory and return its path."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"'{suffix}' is not supported. Allowed types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    dest = settings.uploads_dir / filename
    # Avoid overwriting a file with the same name.
    counter = 1
    while dest.exists():
        dest = settings.uploads_dir / f"{Path(filename).stem}_{counter}{suffix}"
        counter += 1

    dest.write_bytes(file_bytes)
    log.info("Saved uploaded file to %s", dest)
    return dest
