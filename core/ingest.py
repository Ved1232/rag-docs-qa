# ─────────────────────────────────────────────────────────────────
# core/ingest.py — LangChain document ingestion with dual vector store
#
# KEY DESIGN: Deterministic chunk IDs via SHA256
#
# WHY SHA256 NOT UUID? (interview answer):
#   UUID generates a random ID every run — the same chunk gets a
#   different ID each time so the duplicate check never matches.
#   SHA256(source_name + content) always produces the SAME ID for
#   the same content — so re-ingesting the same document correctly
#   detects and skips existing chunks.
#
#   Before (broken):  chunk_id = uuid4()  → always new → always inserted
#   After  (fixed):   chunk_id = sha256(source+content) → same → skipped
# ─────────────────────────────────────────────────────────────────

import os
import hashlib
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()
CHROMA_PATH = os.getenv("CHROMA_PATH_OVERRIDE", "chroma_db")
COLLECTION_NAME = "documents"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs-qa")

# ── EMBEDDINGS ────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── TEXT SPLITTER ─────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def make_chunk_id(source_name: str, content: str) -> str:
    """
    Creates a deterministic chunk ID from source name + content.

    WHY DETERMINISTIC?
    The same document always produces the same set of IDs.
    When re-ingested, the duplicate check finds these IDs already
    in the DB and skips them — preventing unbounded growth.

    SHA256 gives a 64-char hex string — unique enough for any
    practical document collection with zero collision risk.
    """
    return hashlib.sha256(
        (source_name + "|" + content).encode("utf-8")
    ).hexdigest()


# ── VECTOR STORE FACTORY ──────────────────────────────────────────

def get_vectorstore(namespace=None):
    """
    Factory — returns ChromaDB or Pinecone based on VECTOR_STORE env var.
    Single point of configuration for the entire codebase.
    """
    if VECTOR_STORE == "pinecone":
        return _get_pinecone_store(namespace=namespace)
    return _get_chroma_store()


def _get_chroma_store():
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )


def _get_pinecone_store(namespace=None):
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    import time

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace or "default"
    )


# ── INGESTION ─────────────────────────────────────────────────────

def ingest_text(text: str, source_name: str = "pasted_text", namespace=None):
    """
    Splits text into chunks and stores them in the vector DB.
    Uses SHA256 chunk IDs to prevent duplicate ingestion.

    Args:
        text: raw document text
        source_name: label shown in citation badges
        namespace: Pinecone namespace for per-user isolation

    Returns:
        tuple: (chunks_added, total_chunks)
    """
    if not text or not text.strip():
        return 0, 0

    # Step 1 — split into chunks
    docs = [Document(page_content=text, metadata={"source": source_name})]
    chunks = text_splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return 0, 0

    # Step 2 — generate deterministic IDs
    chunk_ids = [
        make_chunk_id(source_name, chunk.page_content)
        for chunk in chunks
    ]

    vectorstore = get_vectorstore(namespace=namespace)

    if VECTOR_STORE == "pinecone":
        # Pinecone upsert is idempotent — same ID = update not duplicate
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)
        return len(chunks), len(chunks)

    # ChromaDB — check existing IDs before inserting
    try:
        existing = vectorstore._collection.get(ids=chunk_ids)
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    # Step 3 — filter to only new chunks
    new_chunks = []
    new_ids = []
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    # Step 4 — insert only new chunks
    if new_chunks:
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    return len(new_chunks), len(chunks)


def ingest_file(file_path: str, namespace=None):
    """
    Loads a TXT or PDF file from disk and ingests its content.

    Args:
        file_path: path to file on disk
        namespace: Pinecone namespace for per-user isolation

    Returns:
        tuple: (chunks_added, total_chunks)
    """
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    file_name = os.path.basename(file_path)
    ext = file_name.lower().split(".")[-1]

    if ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        return 0, 0

    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_name

    chunks = text_splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return 0, 0

    # Deterministic IDs based on filename + content
    chunk_ids = [
        make_chunk_id(file_name, chunk.page_content)
        for chunk in chunks
    ]

    vectorstore = get_vectorstore(namespace=namespace)

    if VECTOR_STORE == "pinecone":
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)
        return len(chunks), len(chunks)

    try:
        existing = vectorstore._collection.get(ids=chunk_ids)
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    new_chunks = []
    new_ids = []
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    return len(new_chunks), len(chunks)