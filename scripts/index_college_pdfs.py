"""Index local KIT PDFs into the persistent Chroma vector collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.knowledge_base import PDF_DIR, index_pdf_folder


def main() -> None:
    result = index_pdf_folder(PDF_DIR)
    print(f"PDF folder: {PDF_DIR}")
    print(f"Files indexed: {result['files_indexed']}/{result['files_found']}")
    print(f"Vector chunks: {result['chunks_indexed']}")
    print(f"Chroma collection total: {result['collection_count']}")
    if result["skipped_no_text"]:
        print("Skipped PDFs with no selectable text: " + ", ".join(result["skipped_no_text"]))
    print(f"Persistent DB: {result['vector_store']}")


if __name__ == "__main__":
    main()
