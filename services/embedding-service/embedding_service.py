from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INGESTION_SERVICE_DIR = PROJECT_ROOT / "services" / "ingestion-service"
for candidate in (PROJECT_ROOT, INGESTION_SERVICE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from parser import clean_ocr_text, extract_text_from_file


EMBEDDER_PATH = Path(__file__).resolve().parent.parent / "embedding-service" / "embedder.py"

_spec = importlib.util.spec_from_file_location("embedder_impl", EMBEDDER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load embedding helper from {_spec}")

_embedder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_embedder_module)


def build_index(file_path: str, source_type: str, docs: list[str] | None = None):
    """Build a persisted FAISS index using the repository's embedded helper module."""
    if docs is None:
        docs = extract_text_from_file(file_path)

    cleaned_docs = [clean_ocr_text(doc) for doc in docs if clean_ocr_text(doc)]
    cleaned_docs = list(dict.fromkeys(cleaned_docs))

    if not cleaned_docs:
        return None, None

    return _embedder_module.create_vectorstore(
        file_path=file_path,
        docs=cleaned_docs,
        source_type=source_type,
    )
