import os
import asyncio
from datetime import datetime, timezone
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

# Async MongoDB Client
client = AsyncIOMotorClient(MONGO_URI)

# Database
db = client["laptop_rag"]

# Collections
users_collection = db["users"]
chats_collection = db["chat_history"]  # Holds session metadata
messages_collection = db["messages"]   # Holds actual turns (user/assistant)
global_contexts_collection = db["global_contexts"]  # One reusable app-wide context file


# =====================================================================
# INDEX CREATION (Ensures fast lookups per user and session)
# =====================================================================
async def init_indexes():
    """Create indexes to keep user data strictly segregated and fast to query."""
    # Fast lookup of all chat sessions for a user
    await chats_collection.create_index([("user_id", 1), ("updated_at", -1)])

    # Fast retrieval of history for a specific session
    await messages_collection.create_index([("user_id", 1), ("session_id", 1), ("created_at", 1)])

    # One reusable app-wide global context per deployment
    await global_contexts_collection.create_index([("is_active", 1), ("updated_at", -1)])


# =====================================================================
# RAG CHAT HELPER FUNCTIONS
# =====================================================================

async def create_chat_session(user_id: str, title: str = "New Laptop Search") -> str:
    """Creates a new segregated chat session for a user with its own message history."""
    session_doc = {
        "user_id": user_id,
        "title": title,
        "messages": [],
        "message_count": 0,
        "last_message_preview": None,
        "uploaded_file_name": None,
        "uploaded_file_context": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    result = await chats_collection.insert_one(session_doc)
    return str(result.inserted_id)


async def save_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    embedding: list[float] = None
):
    """Appends a message to a user-specific session and persists it independently."""
    session_filter = {"_id": session_id, "user_id": user_id}
    try:
        session_filter["_id"] = ObjectId(session_id)
    except Exception:
        session_filter["_id"] = session_id

    now = datetime.now(timezone.utc)
    message_doc = {
        "user_id": user_id,
        "session_id": str(session_filter["_id"]),
        "role": role,
        "content": content,
        "created_at": now,
    }
    if embedding is not None:
        message_doc["embedding"] = embedding

    await messages_collection.insert_one(message_doc)

    update_result = await chats_collection.update_one(
        session_filter,
        {
            "$push": {"messages": message_doc},
            "$inc": {"message_count": 1},
            "$set": {
                "updated_at": now,
                "last_message_preview": content[:200],
            }
        }
    )

    if update_result.matched_count == 0:
        fallback_doc = {
            "_id": session_filter["_id"],
            "user_id": user_id,
            "title": "New Laptop Search",
            "messages": [message_doc],
            "message_count": 1,
            "last_message_preview": content[:200],
            "created_at": now,
            "updated_at": now
        }
        await chats_collection.insert_one(fallback_doc)


async def get_recent_chat_history(user_id: str, session_id: str, limit: int = 6) -> list[dict]:
    """
    Fetches the last N messages for a user's session from the dedicated messages collection.
    """
    cursor = messages_collection.find(
        {
            "user_id": user_id,
            "session_id": session_id,
        },
        {"embedding": 0}
    ).sort("created_at", -1).limit(limit)

    messages = await cursor.to_list(length=limit)
    messages.reverse()
    return messages


async def attach_uploaded_file_context(user_id: str, session_id: str, file_name: str, file_context: str) -> bool:
    """Stores the most recent uploaded file context on the user's specific session doc."""
    session_filter = {"_id": session_id, "user_id": user_id}
    try:
        session_filter["_id"] = ObjectId(session_id)
    except Exception:
        session_filter["_id"] = session_id

    now = datetime.now(timezone.utc)
    update_result = await chats_collection.update_one(
        session_filter,
        {
            "$set": {
                "uploaded_file_name": file_name,
                "uploaded_file_context": file_context,
                "updated_at": now,
            }
        }
    )
    return update_result.matched_count > 0


async def get_session_uploaded_context(user_id: str, session_id: str) -> dict:
    """Fetches the stored uploaded file metadata for a specific session."""
    session_filter = {"_id": session_id, "user_id": user_id}
    try:
        session_filter["_id"] = ObjectId(session_id)
    except Exception:
        session_filter["_id"] = session_id

    session_doc = await chats_collection.find_one(session_filter, {"uploaded_file_name": 1, "uploaded_file_context": 1})
    if not session_doc:
        return {"uploaded_file_name": None, "uploaded_file_context": None}

    return {
        "uploaded_file_name": session_doc.get("uploaded_file_name"),
        "uploaded_file_context": session_doc.get("uploaded_file_context"),
    }


async def get_user_chat_sessions(user_id: str) -> list[dict]:
    """Lists all chat sessions for a user in a sidebar-friendly format."""
    if not user_id:
        return []

    candidate_ids = {user_id}

    # Be resilient to older records that were created using a legacy user_id.
    try:
        user_doc = await users_collection.find_one(
            {"$or": [{"_id": ObjectId(user_id)}, {"user_id": user_id}, {"username": user_id}, {"email": user_id}]},
            {"_id": 1, "user_id": 1}
        )
        if user_doc:
            candidate_ids.add(str(user_doc.get("_id")))
            candidate_ids.add(str(user_doc.get("user_id")))
    except Exception:
        # If the supplied user_id is not a valid ObjectId string, we just continue with the raw value.
        pass

    cursor = chats_collection.find(
        {"user_id": {"$in": list(candidate_ids)}}
    ).sort("updated_at", -1)

    sessions = await cursor.to_list(length=100)
    serializable_sessions = []

    for session in sessions:
        session_id = str(session.pop("_id"))
        normalized_messages = []
        for message in session.get("messages", []):
            normalized_message = dict(message)
            if "_id" in normalized_message:
                normalized_message["_id"] = str(normalized_message["_id"])
            if "created_at" in normalized_message and hasattr(normalized_message["created_at"], "isoformat"):
                normalized_message["created_at"] = normalized_message["created_at"].isoformat()
            normalized_messages.append(normalized_message)

        session["session_id"] = session_id
        session["_id"] = session_id
        session["messages"] = normalized_messages
        session["created_at"] = session.get("created_at").isoformat() if hasattr(session.get("created_at"), "isoformat") else session.get("created_at")
        session["updated_at"] = session.get("updated_at").isoformat() if hasattr(session.get("updated_at"), "isoformat") else session.get("updated_at")
        session["message_count"] = session.get("message_count", len(normalized_messages))
        session["last_message_preview"] = session.get("last_message_preview") or (
            normalized_messages[-1].get("content", "")[:200] if normalized_messages else None
        )
        serializable_sessions.append(session)

    return serializable_sessions


async def delete_chat_session(user_id: str, session_id: str) -> bool:
    """Deletes a single chat session and its associated message records for that user."""
    session_filter = {"user_id": user_id}
    try:
        session_filter["_id"] = ObjectId(session_id)
    except Exception:
        session_filter["_id"] = session_id

    deletion = await chats_collection.delete_one(session_filter)
    if deletion.deleted_count:
        await messages_collection.delete_many({"user_id": user_id, "session_id": session_id})
        return True
    return False


async def save_global_context(file_name: str, context_chunks: list[str], context_text: str, vectorstore_path: str) -> dict:
    """Stores one reusable app-wide context file in MongoDB for future chat sessions."""
    now = datetime.now(timezone.utc)

    await global_contexts_collection.update_many(
        {"is_active": True},
        {"$set": {"is_active": False, "updated_at": now}}
    )

    document = {
        "file_name": file_name,
        "context_chunks": context_chunks,
        "context_text": context_text,
        "vectorstore_path": vectorstore_path,
        "source_type": "global",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }

    result = await global_contexts_collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    return document


async def clear_active_global_context() -> bool:
    """Removes the current app-wide uploaded context from MongoDB and falls back to the default dataset."""
    now = datetime.now(timezone.utc)
    result = await global_contexts_collection.update_many(
        {"is_active": True},
        {"$set": {"is_active": False, "updated_at": now}}
    )
    return result.modified_count > 0


async def get_active_global_context() -> dict | None:
    """Fetches the currently active reusable global chat context."""
    context = await global_contexts_collection.find_one(
        {"is_active": True},
        sort=[("updated_at", -1)]
    )
    if not context:
        return None

    context["_id"] = str(context.get("_id"))
    return context


# =====================================================================
# CONNECTION TEST
# =====================================================================
async def test_connection():
    try:
        await client.admin.command("ping")
        await init_indexes()
        print("✅ MongoDB Connected Successfully & Indexes Verified")
    except Exception as e:
        print("❌ MongoDB Connection Failed")
        print(e)

if __name__ == "__main__":
    asyncio.run(test_connection())