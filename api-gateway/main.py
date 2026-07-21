from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
import importlib.util
from pathlib import Path

# Import the authentication router
from auth_routes import router as auth_router

from shared.database import (
    create_chat_session, get_user_chat_sessions,
    delete_chat_session, save_message, chats_collection,
    get_recent_chat_history
)
from shared.security import verify_token

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_SERVICE_PATH = PROJECT_ROOT / "services" / "retrieval-service" / "main.py"


def _load_service_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load service module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieval_service = _load_service_module("retrieval_service_main", RETRIEVAL_SERVICE_PATH)

app = FastAPI(title="RAG API Gateway")

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://frontend-for-rag-project.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the Authentication Router (/auth/login, /auth/signup, /auth/me)
app.include_router(auth_router)


@app.on_event("startup")
async def startup_restore_retrieval_context():
    await retrieval_service.startup_restore_global_context()

@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "RAG API Gateway is running"}

# Forward File Upload to Retrieval Service (builds embeddings + updates RAG chain + attaches session context)
@app.post("/upload")
async def proxy_upload(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    request: Request = None,
):
    return await retrieval_service.upload_file(file=file, session_id=session_id, request=request)


# Forward Image/PDF Upload to Retrieval Service (OCR + RAG chain update)
@app.post("/img_upload")
async def proxy_img_upload(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    request: Request = None,
):
    return await retrieval_service.upload_image(file=file, session_id=session_id, request=request)


# Forward Chat Request to Retrieval Service & Persist Chat
@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = body.get("message") or body.get("input")
    session_id = body.get("session_id")
    user_id = body.get("user_id")

    if not user_id:
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            user_id = verify_token(token)

    if not message or not session_id:
        raise HTTPException(status_code=422, detail="Both 'message' and 'session_id' are required.")

    data = await retrieval_service.retrieve_answer(
        {"message": message, "session_id": session_id, "user_id": user_id or "unknown"}
    )
    answer = data.get("answer")

    # Persist chat history in DB
    await save_message(user_id=user_id or "unknown", session_id=session_id, role="user", content=message)
    await save_message(user_id=user_id or "unknown", session_id=session_id, role="assistant", content=answer)

    return {"answer": answer}

@app.post("/reset-to-default")
async def proxy_reset_to_default():
    return await retrieval_service.reset_to_default()


# Chat Session Management Endpoints
@app.post("/chat-session")
async def create_session(payload: dict | None = None):
    payload = payload or {}
    session_id = await create_chat_session(
        user_id=payload.get("user_id", "unknown"),
        title=payload.get("title", "New Search")
    )
    return {"session_id": session_id}

@app.get("/chat-sessions")
async def list_sessions(user_id: str | None = None, request: Request = None):
    if not user_id:
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            user_id = verify_token(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to fetch chat sessions.")

    return {"sessions": await get_user_chat_sessions(user_id)}


@app.get("/messages")
async def get_messages(session_id: str, request: Request):
    user_id = None
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        user_id = verify_token(token)

    if not user_id and session_id:
        session_doc = await chats_collection.find_one({"_id": ObjectId(session_id)})
        if session_doc:
            user_id = session_doc.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to fetch chat messages.")

    history = await get_recent_chat_history(user_id=user_id, session_id=session_id, limit=100)

    formatted_messages = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            formatted_messages.append({"message": content, "answer": None})
        elif role == "assistant":
            if formatted_messages and formatted_messages[-1].get("answer") is None:
                formatted_messages[-1]["answer"] = content
            else:
                formatted_messages.append({"message": None, "answer": content})

    return {"messages": formatted_messages}


@app.post("/chat-session/delete")
async def delete_chat_session_route(payload: dict | None = None):
    payload = payload or {}
    user_id = payload.get("user_id")
    session_id = payload.get("session_id")

    if not user_id or not session_id:
        raise HTTPException(status_code=422, detail="Both 'user_id' and 'session_id' are required.")

    deleted = await delete_chat_session(user_id=user_id, session_id=session_id)
    return {"deleted": deleted}
