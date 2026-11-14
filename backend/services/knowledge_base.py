"""Local PDF extraction, chunking, Chroma persistence, and retrieval."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any

import chromadb
import fitz
from chromadb.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_configured_pdf_dir = Path(os.getenv("KIT_PDF_DIR", str(PROJECT_ROOT.parent / "college docs"))).expanduser()
PDF_DIR = _configured_pdf_dir if _configured_pdf_dir.is_absolute() else PROJECT_ROOT / _configured_pdf_dir
_configured_chroma_dir = Path(os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))).expanduser()
CHROMA_DIR = _configured_chroma_dir if _configured_chroma_dir.is_absolute() else PROJECT_ROOT / _configured_chroma_dir
COLLECTION_NAME = "kit_college_documents"
CHUNK_WORDS = 420
CHUNK_OVERLAP = 70
_lock = threading.RLock()
_collection = None


def collection():
    """Return persistent collection; Chroma's local default embedder creates vectors."""
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
                _collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
    return _collection


def _chunks_for_pdf(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]], str]:
    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text", sort=True)).strip()
            words = text.split()
            if not words:
                continue
            step = CHUNK_WORDS - CHUNK_OVERLAP
            for start in range(0, len(words), step):
                chunk_words = words[start:start + CHUNK_WORDS]
                if not chunk_words:
                    continue
                chunk = " ".join(chunk_words)
                chunk_id = hashlib.sha256(f"{path.name}:{content_hash}:{page_number}:{start}".encode("utf-8")).hexdigest()
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": path.name,
                    "title": path.stem.replace("_", " "),
                    "page": page_number,
                    "chunk_start_word": start,
                    "content_hash": content_hash,
                })
    return ids, documents, metadatas, content_hash


def index_pdf_folder(folder: Path = PDF_DIR) -> dict[str, Any]:
    """Extract selectable PDF text, chunk with overlap, and upsert vectors."""
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")

    pending: list[tuple[Path, list[str], list[str], list[dict[str, Any]]]] = []
    skipped: list[str] = []
    for path in pdfs:
        ids, documents, metadatas, _ = _chunks_for_pdf(path)
        if not documents:
            skipped.append(path.name)
            continue
        pending.append((path, ids, documents, metadatas))

    target = collection()
    total_chunks = 0
    with _lock:
        for path, ids, documents, metadatas in pending:
            # Remove stale tail chunks if source PDF became shorter, then write current chunks.
            target.delete(where={"source": path.name})
            for start in range(0, len(ids), 64):
                end = start + 64
                target.upsert(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
            total_chunks += len(ids)

    return {
        "files_found": len(pdfs),
        "files_indexed": len(pending),
        "chunks_indexed": total_chunks,
        "skipped_no_text": skipped,
        "collection_count": target.count(),
        "vector_store": str(CHROMA_DIR),
    }


def search_documents(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find nearest text chunks with source/page citation metadata."""
    if not query.strip():
        return []
    target = collection()
    count = target.count()
    if not count:
        return []
    with _lock:
        result = target.query(
            query_texts=[query.strip()],
            n_results=min(max(1, limit), count),
            include=["documents", "metadatas", "distances"],
        )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"text": text, "metadata": metadata or {}, "distance": distances[index] if index < len(distances) else None}
        for index, (text, metadata) in enumerate(zip(documents, metadatas))
    ]
