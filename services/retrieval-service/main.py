print("=" * 60)
print("RUNNING RETRIEVAL SERVICE")
print(__file__)
print("=" * 60)
import asyncio
import sys
import tempfile
from pathlib import Path

from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVICE_DIR = Path(__file__).resolve().parent

for candidate in (PROJECT_ROOT, SERVICE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import rag
from shared.database import attach_uploaded_file_context, chats_collection, get_session_uploaded_context, clear_active_global_context
from shared.security import verify_token

app = FastAPI(title="Laptop RAG API")

@app.on_event("startup")
async def startup_restore_global_context():
    """Attempt to restore previously uploaded context from MongoDB.
    If nothing is stored there, fall back to the bundled default dataset
    so the service is immediately usable for laptop queries."""
    restored = await rag.restore_global_context_from_db()
    if not restored and DEFAULT_DATASET_PATH.exists():
        print(f"No persisted context found — loading default dataset: {DEFAULT_DATASET_PATH}")
        await asyncio.to_thread(rag.initialize_rag_system, str(DEFAULT_DATASET_PATH))
    elif not restored:
        print("Warning: No persisted context and default dataset not found. Upload a file to use RAG.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED PATH: Points to /data/dataset.csv relative to project root
# (goes up two directory levels from services/retrieval-service/main.py)
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset.csv"


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "Laptop RAG API is running"}


@app.post("/retrieve")
async def retrieve_answer(payload: dict):
    message = payload.get("message") or payload.get("input")
    session_id = payload.get("session_id")
    user_id = payload.get("user_id")

    if not message or not session_id:
        raise HTTPException(status_code=422, detail="Both 'message' and 'session_id' are required.")

    uploaded_context = None
    if user_id and session_id:
        uploaded_context = await get_session_uploaded_context(user_id=user_id, session_id=session_id)
        uploaded_file_context = uploaded_context.get("uploaded_file_context") if isinstance(uploaded_context, dict) else None
        if uploaded_file_context:
            message = (
                "Use the uploaded document context below to answer the question. "
                "If the question is about the uploaded file, answer from that context first.\n\n"
                f"Uploaded document context:\n{uploaded_file_context}\n\n"
                f"Question:\n{message}"
            )

    if rag.conversational_router_chain is None:
        restored = await rag.restore_global_context_from_db()
        if not restored:
            await asyncio.to_thread(rag.initialize_rag_system, str(DEFAULT_DATASET_PATH))

    answer = await asyncio.to_thread(rag.rag_answer, message, session_id)
    return {"answer": answer}


@app.post("/img_upload")
async def upload_image(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    request: Request = None,
):
    """Receives image/PDF uploads from React, extracts OCR text, embeds it, and makes it available as document context for follow-up queries."""

    suffix = Path(file.filename).suffix.lower()
    image_types = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"]

    if suffix not in image_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Send an image file (.png, .jpg, .jpeg, .bmp, .tiff) or a .pdf for /img_upload.",
        )

    tmp_path = None
    user_id = None
    try:
        auth_header = (request.headers.get("authorization") or "") if request else ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            user_id = verify_token(token)

        if not user_id and session_id:
            session_doc = await chats_collection.find_one({"_id": ObjectId(session_id)})
            if session_doc:
                user_id = session_doc.get("user_id")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        ocr_query = await asyncio.to_thread(rag.extract_text_as_query, str(tmp_path))
        if not ocr_query:
            return {
                "status": "error",
                "message": "Unable to read text from the uploaded image. Please try a clearer image.",
            }

        if session_id and user_id:
            await attach_uploaded_file_context(
                user_id=user_id,
                session_id=session_id,
                file_name=file.filename,
                file_context=ocr_query,
            )
            return {
                "status": "success",
                "message": "OCR text extracted and attached to this chat session.",
                "answer": ocr_query[:800],
                "source_type": "session_ocr_context",
                "ask_user_confirmation": False,
            }

        # Global OCR uploads remain a reusable app-wide context file.
        await asyncio.to_thread(rag.initialize_rag_system, str(tmp_path))

        if rag.conversational_router_chain is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build a usable OCR-based RAG index from {file.filename}.",
            )

        await rag.persist_global_context()

        config = {"configurable": {"session_id": session_id or "image-upload"}}
        response = await asyncio.to_thread(
            rag.conversational_router_chain.invoke,
            {"input": ocr_query},
            config,
        )
        answer = response.get("answer") if isinstance(response, dict) else str(response)

        return {
            "status": "success",
            "message": "OCR text extracted successfully. Do you want to use this extracted data as a query?",
            "answer": answer,
            "source_type": "ocr_context",
            "ask_user_confirmation": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    request: Request = None,
):
    """Receives document uploads from React and stores either chat-scoped context or app-wide global context."""

    suffix = Path(file.filename).suffix.lower()
    text_types = [".csv", ".txt", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    if suffix not in text_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Send .csv, .txt, .docx, .pdf, or an image file for document uploads.",
        )

    tmp_path = None
    user_id = None
    try:
        auth_header = (request.headers.get("authorization") or "") if request else ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            user_id = verify_token(token)

        if not user_id and session_id:
            session_doc = await chats_collection.find_one({"_id": ObjectId(session_id)})
            if session_doc:
                user_id = session_doc.get("user_id")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        uploaded_context = await asyncio.to_thread(rag.extract_text_as_query, str(tmp_path))

        if session_id and user_id:
            await attach_uploaded_file_context(
                user_id=user_id,
                session_id=session_id,
                file_name=file.filename,
                file_context=uploaded_context or "",
            )
            return {"status": "success", "message": f"Successfully attached {file.filename} to this chat session."}

        # Global context upload: rebuild the reusable app-wide dataset and persist the active file to MongoDB.
        await asyncio.to_thread(rag.initialize_rag_system, str(tmp_path))
        if rag.conversational_router_chain is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build a usable RAG index from {file.filename}. Please upload a supported file with readable content.",
            )

        await rag.persist_global_context()

        return {"status": "success", "message": f"Successfully indexed {file.filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


@app.post("/reset-to-default")
async def reset_to_default():
    if not DEFAULT_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Default dataset not found at {DEFAULT_DATASET_PATH}")

    try:
        await clear_active_global_context()
        await asyncio.to_thread(rag.initialize_rag_system, str(DEFAULT_DATASET_PATH))
        return {"status": "success", "message": "Reverted to default dataset and dropped the active global context file."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
