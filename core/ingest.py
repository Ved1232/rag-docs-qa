# ─────────────────────────────────────────────────────────────────
# core/ingest.py — LangChain document ingestion with dual vector store
#
# VECTOR STORE STRATEGY:
#   VECTOR_STORE=chroma  (default) → local ChromaDB, dev/testing
#   VECTOR_STORE=pinecone          → cloud Pinecone, production
#
# WHY THIS PATTERN? (interview answer):
#   "The factory pattern abstracts the vector store behind a single
#    function get_vectorstore(). The rest of the codebase never
#    imports ChromaDB or Pinecone directly — they only call
#    get_vectorstore(). This means switching databases is one
#    environment variable change, zero code changes. It's the
#    same pattern used in production systems like LangChain itself."
#
# CHROMADB vs PINECONE (interview answer):
#   ChromaDB: runs locally, zero cost, zero network, perfect for dev.
#             Data lives in chroma_db/ folder on disk.
#             Limitation: single machine only, no horizontal scaling.
#
#   Pinecone: cloud-hosted, scales to billions of vectors,
#             built-in replication and backups, sub-10ms queries
#             at any scale. Costs money per vector stored.
#             Production choice for any serious deployment.
# ─────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────
# VECTOR_STORE env var controls which backend is used
# Default is chroma so local development works with no extra setup
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()

# ChromaDB settings
CHROMA_PATH = os.getenv("CHROMA_PATH_OVERRIDE", "chroma_db")
COLLECTION_NAME = "documents"

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs-qa")

# ── EMBEDDINGS ────────────────────────────────────────────────────
# Same embeddings model regardless of vector store backend.
# WHY KEEP EMBEDDINGS CONSISTENT?
# "The embedding model and vector store are independent concerns.
#  The embedding model converts text to vectors — that's fixed.
#  The vector store stores and searches those vectors — that's
#  swappable. Keeping them separate means I can change the DB
#  without re-embedding all my documents."
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── TEXT SPLITTER ─────────────────────────────────────────────────
# chunk_size=500: enough context per chunk without being too broad
# chunk_overlap=80: prevents context loss at chunk boundaries
# separators: tries paragraph → sentence → word boundaries in order
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ── VECTOR STORE FACTORY ──────────────────────────────────────────

def get_vectorstore(namespace=None):
    """
    Factory function — returns the correct vector store based on
    the VECTOR_STORE environment variable.

    This is the ONLY place in the codebase that knows about
    ChromaDB or Pinecone. Everything else calls this function.

    WHY A FACTORY FUNCTION? (interview answer):
    "A factory function is a design pattern where a single function
     decides which concrete implementation to return based on
     configuration. The caller gets back a LangChain VectorStore
     object — it doesn't know or care whether it's ChromaDB or
     Pinecone underneath. This is the Open/Closed principle:
     open for extension (add new backends), closed for modification
     (existing code doesn't change)."

    Args:
        namespace: Pinecone namespace for per-user isolation.
                   Ignored for ChromaDB (not needed locally).
                   When set, queries only search that user's vectors.

    Returns:
        LangChain VectorStore instance (Chroma or PineconeVectorStore)
    """
    if VECTOR_STORE == "pinecone":
        return _get_pinecone_store(namespace=namespace)
    else:
        return _get_chroma_store()


def _get_chroma_store():
    """
    Returns a local ChromaDB vector store.
    Used for development and testing — no API keys needed.
    """
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )


def _get_pinecone_store(namespace=None):
    """
    Returns a Pinecone cloud vector store.
    Used for production — scales to any size.

    HOW PINECONE WORKS (interview answer):
    "Pinecone organises vectors in indexes. An index is like a
     database table — it stores vectors of a fixed dimension.
     We use dimension=1536 to match text-embedding-3-small.
     Within an index, namespaces provide logical partitioning —
     user A's vectors are in namespace 'user_a', user B's in
     'user_b'. A query in namespace 'user_a' never touches
     user B's vectors — isolation enforced at the DB level."

    PINECONE INDEX CREATION:
    The index is created automatically on first use if it doesn't
    exist. In production you'd create it via Terraform or the
    Pinecone console to have it ready before the app starts.

    Args:
        namespace: optional string for per-user document isolation
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found in .env — "
            "add it or set VECTOR_STORE=chroma for local development"
        )

    # Initialise Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create the index if it doesn't exist yet
    # dimension=1536 must match text-embedding-3-small output size
    # metric=cosine: measures angle between vectors — best for text
    # WHY COSINE NOT EUCLIDEAN?
    # "Cosine similarity measures the angle between vectors, not
    #  their magnitude. For text embeddings, two sentences with
    #  similar meaning point in the same direction regardless of
    #  their length. Euclidean distance penalises longer texts
    #  which have larger magnitude vectors — cosine doesn't."
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            # ServerlessSpec: Pinecone manages the infrastructure
            # No servers to provision, pay only for what you use
            # WHY SERVERLESS?
            # "Serverless Pinecone auto-scales — at zero queries
            #  you pay near zero, at peak load it scales up
            #  automatically. A pod-based index has fixed cost
            #  regardless of usage."
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # free tier supported region
            )
        )
        # Wait for index to be ready
        import time
        print("Waiting for index to be ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("Index ready!")

    # Return the LangChain wrapper around Pinecone
    # namespace parameter scopes all operations to that user's data
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace or "default"
    )


# ── INGESTION FUNCTIONS ───────────────────────────────────────────

def ingest_text(text, source_name="pasted_text", namespace=None):
    """
    Ingests raw text into the active vector store.

    The namespace parameter enables per-user document isolation
    in Pinecone. In ChromaDB mode it is ignored.

    Args:
        text: raw string content to ingest
        source_name: label shown in citation badges in the UI
        namespace: Pinecone namespace (username for user isolation)

    Returns:
        tuple: (chunks_added, total_chunks)
    """
    if not text or not text.strip():
        return 0, 0

    # Wrap text in LangChain Document with source metadata
    doc = Document(
        page_content=text,
        metadata={"source": source_name}
    )

    # Split into chunks — paragraph boundaries preferred
    chunks = text_splitter.split_documents([doc])
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return 0, 0

    vectorstore = get_vectorstore(namespace=namespace)

    # Generate consistent IDs for deduplication
    # Same document ingested twice → same IDs → Pinecone upserts
    # (updates existing rather than creating duplicates)
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]

    if VECTOR_STORE == "pinecone":
        # Pinecone uses add_texts for better ID control
        # upsert=True means update if ID exists, insert if not
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks), len(chunks)

    else:
        # ChromaDB — check for existing IDs to prevent duplicates
        new_chunks = []
        new_ids = []

        for i, (chunk, chunk_id) in enumerate(zip(chunks, ids)):
            try:
                existing = vectorstore._collection.get(ids=[chunk_id])
                if not existing["ids"]:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)
            except Exception:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if not new_chunks:
            return 0, len(chunks)

        vectorstore.add_documents(documents=new_chunks, ids=new_ids)
        return len(new_chunks), len(chunks)


def ingest_file(file_path, namespace=None):
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

    # Ensure all chunks carry just the filename as source
    for doc in documents:
        doc.metadata["source"] = file_name

    chunks = text_splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        return 0, 0

    vectorstore = get_vectorstore(namespace=namespace)
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

    if VECTOR_STORE == "pinecone":
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks), len(chunks)

    else:
        new_chunks = []
        new_ids = []
        for chunk, chunk_id in zip(chunks, ids):
            try:
                existing = vectorstore._collection.get(ids=[chunk_id])
                if not existing["ids"]:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)
            except Exception:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if not new_chunks:
            return 0, len(chunks)

        vectorstore.add_documents(documents=new_chunks, ids=new_ids)
        return len(new_chunks), len(chunks)