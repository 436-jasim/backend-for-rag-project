import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile
import httpx
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INGESTION_SERVICE_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, INGESTION_SERVICE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from parser import clean_ocr_text, extract_text_from_file

app = FastAPI(title="Ingestion Service")

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8004")


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "Ingestion Service is running"}


@app.post("/upload")
async def process_upload(file: UploadFile = File(...), session_id: str | None = Form(None)):
    suffix = Path(file.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        chunks = extract_text_from_file(str(tmp_path))
        cleaned_chunks = [clean_ocr_text(chunk) for chunk in chunks if clean_ocr_text(chunk)]
        if not cleaned_chunks:
            raise HTTPException(status_code=400, detail="Could not extract readable text.")

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"chunks": cleaned_chunks, "source": file.filename, "session_id": session_id}
            )
            if res.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to index vectors.")

        return {"status": "success", "message": f"Successfully indexed {file.filename}"}

    finally:
        if tmp_path.exists():
            tmp_path.unlink()