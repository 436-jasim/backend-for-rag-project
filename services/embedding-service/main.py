from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.embedding_service import build_index

app = FastAPI(title="Embedding Service")


class EmbedRequest(BaseModel):
    chunks: list[str]
    source: str
    session_id: str | None = None
    source_type: str = "uploaded"


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "Embedding Service is running"}


@app.post("/embed")
async def embed(payload: EmbedRequest):
    file_path = payload.source or payload.session_id or "uploaded-document"
    vectorstore, vectorstore_path = build_index(
        file_path=file_path,
        source_type=payload.source_type,
        docs=payload.chunks,
    )

    if vectorstore is None or vectorstore_path is None:
        raise HTTPException(status_code=500, detail="Failed to build vector embeddings.")

    return {
        "status": "success",
        "message": f"Successfully indexed {payload.source}",
        "vectorstore_path": str(vectorstore_path),
    }
