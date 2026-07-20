import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INGESTION_SERVICE_DIR = PROJECT_ROOT / "services" / "ingestion-service"
for candidate in (PROJECT_ROOT, INGESTION_SERVICE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

from services.embedding_service import build_index
from parser import clean_ocr_text, extract_text_from_file
from shared.database import get_active_global_context, save_global_context

load_dotenv()

# Global pipeline variables for the active RAG session
conversational_router_chain = None
memory_store = {}
current_dataset_name = None
current_dataset_type = None
default_rag_chain = None
uploaded_rag_chain = None
global_rag_chain = None

# Exposed so the async upload endpoint can persist to DB on the main event loop
_last_vectorstore_path = None
_last_docs = None
_last_file_name = None


def process_laptop_answer(answer_text: str) -> str:
    """Removes duplicate laptop-name entries from a retrieved answer while preserving order."""
    if not answer_text:
        return answer_text

    cleaned_lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if len(cleaned_lines) <= 1:
        return answer_text.strip()

    unique_lines = []
    seen_signatures = set()

    for line in cleaned_lines:
        line_lower = line.lower()
        fields = []

        for field in ("company", "product", "typename", "model", "laptop"):
            match = re.search(rf"{field}\s*[\-:\-]\s*([^\n]+)", line_lower)
            if match:
                fields.append(match.group(1).strip())

        signature = " | ".join(fields).strip()
        if signature:
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

        unique_lines.append(line)

    return "\n".join(unique_lines).strip()


def _build_rag_chain_from_vectorstore(vectorstore, file_name: str, dataset_type: str):
    """Create a retrieval chain from a vectorstore and return the configured router chain."""
    global conversational_router_chain, memory_store
    global current_dataset_name, current_dataset_type, default_rag_chain, uploaded_rag_chain, global_rag_chain

    llm = ChatOpenAI(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=os.getenv("HF_TOKEN"),
        base_url="https://router.huggingface.co/v1",
        temperature=0,
        max_tokens=1500,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    router_prompt = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict query router.

Your job is to classify the user's query into exactly one of three categories:
- file_query
- laptop_query
- general_query

RULES (in order):

1. If a global context file has been uploaded for the app, or if one or more files are uploaded in the current conversation, classify as file_query whenever the question can be answered from that file context, unless the user clearly asks about laptops or asks something completely unrelated that cannot reasonably be answered from the uploaded file content.

2. If no uploaded or global context file exists and the query is about laptops, classify as laptop_query.

3. Otherwise classify as general_query.

Definitions:

file_query:
Any question that could reasonably be answered using the uploaded document(s), even if the user never mentions words like "file", "document", "PDF", "report", "table", "page", etc.

laptop_query:
Laptop recommendations, specifications, comparisons, troubleshooting, reviews, pricing, brands, upgrades, buying advice.

general_query:
Anything else.

Return only one word:
file_query
laptop_query
general_query
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
    router_chain = router_prompt | llm | StrOutputParser()

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and polite assistant. Answer the user's question directly."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    general_chain = general_prompt | llm

    laptop_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent document and laptop recommendation assistant.
RULES:
- Answer based directly on the context provided.
- Keep answers concise, factual, and strictly relevant to the user query.
- No conversational fluff or "Assistant:" prefixes.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, laptop_prompt)
    current_rag_chain = create_retrieval_chain(retriever, document_chain)

    if dataset_type == "default":
        default_rag_chain = current_rag_chain
    else:
        uploaded_rag_chain = current_rag_chain
        global_rag_chain = current_rag_chain

    def invoke_rag_chain(chain, user_text: str, chat_history):
        if chain is None:
            return None

        try:
            res = chain.invoke({"input": user_text, "history": chat_history})
            if isinstance(res, dict):
                answer_text = res.get("answer")
            else:
                answer_text = str(res)

            if not answer_text:
                return None

            processed_answer = process_laptop_answer(answer_text)
            answer_lower = processed_answer.lower()
            weak_answer_markers = [
                "i don't know",
                "i do not know",
                "not enough context",
                "not enough information",
                "cannot determine",
                "unable to answer",
                "sorry,",
            ]
            if any(marker in answer_lower for marker in weak_answer_markers):
                return None

            return processed_answer
        except Exception as e:
            print("RAG chain error:", e)
            return None

    def route_query(inputs: dict):
        user_text = inputs["input"]
        chat_history = inputs.get("history", [])

        try:
            raw_decision = router_chain.invoke({"input": user_text})
            if not isinstance(raw_decision, str):
                raw_decision = str(raw_decision)
            decision = raw_decision.strip().lower().replace("'", "").replace('"', "")
        except Exception as e:
            print("Router invoke error:", e)
            try:
                res = general_chain.invoke({"input": user_text, "history": chat_history})
                answer_text = res.content if hasattr(res, 'content') else str(res)
                return {"answer": answer_text}
            except Exception as e2:
                print("General chain fallback error:", e2)
                return {"answer": "Sorry, I'm having trouble processing your request right now."}

        print(f"\n[ROUTER] User input: '{user_text}' -> Classified as: '{decision}'")

        heuristic_terms = [
            "laptop", "model", "cpu", "ram", "storage", "ssd", "hdd", "gpu", "graphics",
            "screen", "display", "battery", "price", "brand", "spec", "specs", "performance",
            "recommend", "compare", "review", "memory", "touchscreen", "resolution", "weight",
        ]
        file_specific_markers = [
            "uploaded file", "in the uploaded file", "in this file", "in the file",
            "from the document", "from the uploaded document", "summary of the file",
            "summarize the uploaded document", "find the entry", "row", "record",
            "this document", "document contents", "uploaded document", "entry for",
            "image", "pdf", "scan", "ocr", "document text", "from the image", "from the pdf",
        ]
        lower_text = user_text.lower()

        if decision not in ("laptop_query", "file_query", "general_query"):
            if dataset_type == "uploaded" and any(marker in lower_text for marker in file_specific_markers):
                print(f"[ROUTER] Heuristic matched uploaded-file markers; forcing file_query for '{user_text}'")
                decision = "file_query"
            elif any(term in lower_text for term in heuristic_terms):
                print(f"[ROUTER] Heuristic matched laptop terms; forcing laptop_query for '{user_text}'")
                decision = "laptop_query"
            else:
                decision = "general_query"

        if decision in ("laptop_query", "file_query"):
            retrieval_candidates = []

            if global_rag_chain is not None:
                retrieval_candidates = [global_rag_chain, uploaded_rag_chain, default_rag_chain]
            elif dataset_type == "uploaded":
                retrieval_candidates = [uploaded_rag_chain, default_rag_chain]
            else:
                retrieval_candidates = [default_rag_chain, uploaded_rag_chain]

            for chain in retrieval_candidates:
                answer_text = invoke_rag_chain(chain, user_text, chat_history)
                if answer_text:
                    return {"answer": answer_text}

            try:
                res = general_chain.invoke({"input": user_text, "history": chat_history})
                answer_text = res.content if hasattr(res, 'content') else str(res)
                return {"answer": answer_text}
            except Exception as e:
                print("General chain fallback error:", e)
                return {"answer": "Sorry, I'm having trouble answering that right now."}

        try:
            res = general_chain.invoke({"input": user_text, "history": chat_history})
            answer_text = res.content if hasattr(res, 'content') else str(res)
            return {"answer": answer_text}
        except Exception as e:
            print("General chain error:", e)
            return {"answer": "Sorry, I'm having trouble answering that right now."}

    dynamic_router = RunnableLambda(route_query)

    def get_session_history(session_id: str):
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]

    conversational_router_chain = RunnableWithMessageHistory(
        dynamic_router,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
    print(f"Successfully loaded {file_name} into the pipeline!\n")


def initialize_rag_system(file_path: str):
    """
    Accepts any supported file, wipes the previous index/memory completely,
    and spins up a new RAG pipeline for the newly uploaded file.

    NOTE: Does NOT persist to MongoDB (no async DB calls). The caller
    should call ``persist_global_context()`` separately on the main event loop.
    """
    global conversational_router_chain, memory_store
    global current_dataset_name, current_dataset_type, default_rag_chain, uploaded_rag_chain, global_rag_chain
    global _last_vectorstore_path, _last_docs, _last_file_name

    memory_store = {}
    conversational_router_chain = None
    default_rag_chain = None
    uploaded_rag_chain = None
    global_rag_chain = None

    print(f"Processing newly uploaded file: {file_path}...")
    current_dataset_name = Path(file_path).name
    is_default_dataset = current_dataset_name.lower() == "dataset.csv"
    current_dataset_type = "default" if is_default_dataset else "uploaded"
    _last_file_name = current_dataset_name

    docs = extract_text_from_file(file_path)
    docs = [clean_ocr_text(doc) for doc in docs if clean_ocr_text(doc)]
    docs = list(dict.fromkeys(docs))

    if not docs:
        print("Warning: Parsed file contains no text data chunks.")
        return

    print(f"Building Vector Embeddings for {len(docs)} text segments...")
    vectorstore, vectorstore_path = build_index(file_path=file_path, source_type=current_dataset_type)
    if vectorstore is None or vectorstore_path is None:
        print("Warning: Failed to build persisted vector index.")
        return

    print(f"Persisted vector embeddings and metadata to: {vectorstore_path}")
    _last_vectorstore_path = str(vectorstore_path)
    _last_docs = docs

    _build_rag_chain_from_vectorstore(vectorstore, Path(file_path).name, current_dataset_type)


async def persist_global_context():
    """Persist the last uploaded file's context to MongoDB (call from main event loop)."""
    global _last_vectorstore_path, _last_docs, _last_file_name
    if not _last_vectorstore_path or not _last_docs or not _last_file_name:
        return

    try:
        context_text = "\n".join(_last_docs)
        docs_for_store = list(dict.fromkeys(_last_docs))
        await save_global_context(
            file_name=_last_file_name,
            context_chunks=docs_for_store,
            context_text=context_text,
            vectorstore_path=_last_vectorstore_path,
        )
        print(f"Persisted global context for {_last_file_name}")
    except Exception as exc:
        print("Warning: Failed to persist global context record:", exc)


async def restore_global_context_from_db():
    """Restore the persisted global uploaded context from MongoDB and make it available to all chats.

    This is an async function because get_active_global_context() uses Motor
    (async MongoDB driver).  Callers must ``await`` it directly — do NOT wrap
    it in ``asyncio.to_thread()`` or ``asyncio.run()``.
    """
    global conversational_router_chain, memory_store
    global current_dataset_name, current_dataset_type, default_rag_chain, uploaded_rag_chain, global_rag_chain

    try:
        context_doc = await get_active_global_context()
    except Exception as exc:
        print("Warning: Failed to read global context from DB:", exc)
        return False

    if not context_doc or not context_doc.get("vectorstore_path"):
        return False

    vectorstore_path = Path(context_doc["vectorstore_path"])
    if not vectorstore_path.exists():
        return False

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)
        current_dataset_name = context_doc.get("file_name", vectorstore_path.name)
        current_dataset_type = "uploaded"
        _build_rag_chain_from_vectorstore(vectorstore, current_dataset_name, current_dataset_type)
        return True
    except Exception as exc:
        print("Warning: Failed to restore persisted global context vectorstore:", exc)
        return False


def rag_answer(question: str, session_id: str) -> str:
    if conversational_router_chain is None:
        return "System Error: No active file uploaded. Please upload a file first."

    config = {"configurable": {"session_id": session_id}}
    response = conversational_router_chain.invoke({"input": question}, config=config)
    return response["answer"]
