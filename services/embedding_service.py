"""Compatibility shim for the legacy services package import path.

This lets the retrieval microservice keep importing
`from services.embedding_service import build_index` while the real
implementation continues to live in the embedding-service folder.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_EMBEDDING_SERVICE_PATH = Path(__file__).resolve().parent / "embedding-service" / "embedding_service.py"

_spec = importlib.util.spec_from_file_location("legacy_embedding_service", _EMBEDDING_SERVICE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load embedding helper from {_EMBEDDING_SERVICE_PATH}")

_embedding_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_embedding_module)

build_index = _embedding_module.build_index

__all__ = ["build_index"]
