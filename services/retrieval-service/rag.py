from __future__ import annotations

import importlib.util
from pathlib import Path

_SERVICE_RAG_CHAIN_PATH = Path(__file__).resolve().parent / "rag_chain.py"

_spec = importlib.util.spec_from_file_location("retrieval_service_rag_chain", _SERVICE_RAG_CHAIN_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load retrieval service rag chain from {_SERVICE_RAG_CHAIN_PATH}")

_rag_chain_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rag_chain_module)

rag_chain = _rag_chain_module

# Compatibility shim so the service imports keep working.
conversational_router_chain = rag_chain.conversational_router_chain
memory_store = rag_chain.memory_store
current_dataset_name = rag_chain.current_dataset_name
current_dataset_type = rag_chain.current_dataset_type
default_rag_chain = rag_chain.default_rag_chain
uploaded_rag_chain = rag_chain.uploaded_rag_chain


def initialize_rag_system(file_path: str):
    rag_chain.initialize_rag_system(file_path)
    globals()["conversational_router_chain"] = rag_chain.conversational_router_chain
    globals()["memory_store"] = rag_chain.memory_store
    globals()["current_dataset_name"] = rag_chain.current_dataset_name
    globals()["current_dataset_type"] = rag_chain.current_dataset_type
    globals()["default_rag_chain"] = rag_chain.default_rag_chain
    globals()["uploaded_rag_chain"] = rag_chain.uploaded_rag_chain


def rag_answer(question: str, session_id: str) -> str:
    return rag_chain.rag_answer(question, session_id)


async def persist_global_context():
    return await rag_chain.persist_global_context()


def extract_text_as_query(file_path: str) -> str:
    from parser import extract_text_as_query as parser_extract_text_as_query
    return parser_extract_text_as_query(file_path)


async def restore_global_context_from_db():
    return await rag_chain.restore_global_context_from_db()
