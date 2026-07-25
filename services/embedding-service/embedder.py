import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document as LC_Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORDB_ROOT = PROJECT_ROOT / "vectordb"


def build_vectorstore_path(file_path: str, source_name: str | None = None) -> Path:
    """Build a stable, file-specific folder under the local vectordb directory for persistent FAISS storage.

    If `source_name` is provided, it is used (sanitized) to build a stable store name
    so re-uploads with the same original filename reuse the same vectorstore.
    Otherwise fall back to using the resolved file path digest (legacy behavior).
    """
    if source_name:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_name).stem)
        digest = hashlib.md5(safe_stem.encode("utf-8")).hexdigest()[:8]
    else:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_path).stem)
        digest = hashlib.md5(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()[:8]

    return VECTORDB_ROOT / f"{safe_stem}_{digest}"


def _find_existing_vectorstore_by_source(source_name: str) -> Path | None:
    """Return an existing vectorstore Path matching a sanitized `source_name`, if present."""
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_name).stem)
    pattern = f"{safe_stem}_*"
    if not VECTORDB_ROOT.exists():
        return None

    for candidate in VECTORDB_ROOT.glob(pattern):
        if (candidate / "index.faiss").exists() and (candidate / "index.pkl").exists():
            return candidate

    return None


def _get_embeddings(local_files_only: bool = False) -> HuggingFaceEmbeddings:
    model_kwargs = {"local_files_only": True} if local_files_only else {}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
    )


def create_vectorstore(file_path: str, docs: list[str], source_type: str, source_name: str | None = None) -> tuple[FAISS, Path]:
    """Turn cleaned text chunks into a persisted FAISS store with metadata."""
    # If a source_name is provided, prefer an existing store that matches its sanitized name.
    if source_name:
        existing = _find_existing_vectorstore_by_source(source_name)
        if existing is not None:
            embeddings = _get_embeddings(local_files_only=True)
            vectorstore = FAISS.load_local(
                str(existing),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return vectorstore, existing

    vectorstore_path = build_vectorstore_path(file_path, source_name=source_name)

    if (vectorstore_path / "index.faiss").exists() and (vectorstore_path / "index.pkl").exists():
        embeddings = _get_embeddings(local_files_only=True)
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, vectorstore_path

    embeddings = _get_embeddings()

    vector_documents = []
    for index, chunk in enumerate(docs):
        vector_documents.append(
            LC_Document(
                page_content=chunk,
                metadata={
                    "source_file": Path(file_path).name,
                    "source_type": source_type,
                    "chunk_index": index,
                    "chunk_length": len(chunk),
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    vectorstore = FAISS.from_documents(vector_documents, embeddings)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_path))
    return vectorstore, vectorstore_path
