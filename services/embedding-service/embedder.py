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


def build_vectorstore_path(file_path: str) -> Path:
    """Build a stable, file-specific folder under the local vectordb directory for persistent FAISS storage."""
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_path).stem)
    digest = hashlib.md5(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()[:8]
    return VECTORDB_ROOT / f"{safe_stem}_{digest}"


def _get_embeddings(local_files_only: bool = False) -> HuggingFaceEmbeddings:
    model_kwargs = {"local_files_only": True} if local_files_only else {}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
    )


def create_vectorstore(file_path: str, docs: list[str], source_type: str) -> tuple[FAISS, Path]:
    """Turn cleaned text chunks into a persisted FAISS store with metadata."""
    vectorstore_path = build_vectorstore_path(file_path)

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
