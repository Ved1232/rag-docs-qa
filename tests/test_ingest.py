# ─────────────────────────────────────────────────────────────────
# tests/test_ingest.py — Tests for document ingestion pipeline
#
# KEY DESIGN DECISION:
#   Uses chromadb.EphemeralClient() (in-memory) instead of
#   PersistentClient (disk-based) for all tests.
#
#   WHY IN-MEMORY FOR TESTS?
#   "PersistentClient writes to disk. In CI environments like
#    GitHub Actions, the runner may have read-only filesystem
#    permissions for certain paths, causing InternalError 1032.
#    EphemeralClient runs entirely in RAM — no disk writes,
#    no permission issues, no cleanup needed, and 10x faster."
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import uuid
import shutil

# Force in-memory ChromaDB for all tests
os.environ["CHROMA_PATH_OVERRIDE"] = ":memory:"


@pytest.fixture(autouse=True)
def patch_vectorstore(monkeypatch):
    """
    Replaces get_vectorstore() with an in-memory ChromaDB instance.
    Runs automatically before every test in this file.

    This is the correct way to test ChromaDB — never write to disk
    in tests. Each test gets a fresh in-memory store that is
    discarded when the test ends.
    """
    import chromadb
    from langchain_chroma import Chroma

    def make_in_memory_store(namespace=None):
        # EphemeralClient = pure in-memory, no disk, no permissions needed
        client = chromadb.EphemeralClient()
        # Use a unique collection name per test to ensure isolation
        collection_name = f"test_{uuid.uuid4().hex[:8]}"
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=None  # overridden by mock_embeddings
        )

    monkeypatch.setattr("core.ingest.get_vectorstore", make_in_memory_store)
    monkeypatch.setattr("core.query.get_vectorstore", make_in_memory_store)


@pytest.fixture
def mock_embeddings(monkeypatch):
    """
    Replaces OpenAI embeddings with a fast, free fake.
    Returns deterministic vectors so retrieval is testable.
    """
    import numpy as np
    from langchain_core.embeddings import Embeddings

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                seed = abs(hash(text[:30])) % 10000
                np.random.seed(seed)
                vectors.append(np.random.rand(1536).tolist())
            return vectors

        def embed_query(self, text):
            seed = abs(hash(text[:30])) % 10000
            np.random.seed(seed)
            return np.random.rand(1536).tolist()

    fake = FakeEmbeddings()
    monkeypatch.setattr("core.ingest.embeddings", fake)


@pytest.fixture
def sample_text():
    return """
    Cloud computing is the delivery of computing services over the internet.
    It includes servers, storage, databases, networking, software, and analytics.

    There are three main types of cloud services:
    Infrastructure as a Service (IaaS) provides virtualised computing resources.
    Platform as a Service (PaaS) provides a development and deployment environment.
    Software as a Service (SaaS) delivers software applications over the internet.

    The main cloud providers are Amazon Web Services, Microsoft Azure, and Google Cloud.
    Each provider offers hundreds of services across compute, storage, and AI categories.
    """


@pytest.fixture
def sample_txt_file(tmp_path, sample_text):
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    return str(file_path)


# ── UNIT TESTS: TEXT SPLITTER ─────────────────────────────────────

class TestChunking:
    """Tests for chunking logic — no DB needed."""

    def test_text_splits_into_multiple_chunks(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        long_text = "This is a test sentence. " * 20
        doc = Document(page_content=long_text, metadata={"source": "test"})
        chunks = splitter.split_documents([doc])

        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunks_inherit_metadata(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        doc = Document(
            page_content="Test content " * 30,
            metadata={"source": "my_document.txt"}
        )
        chunks = splitter.split_documents([doc])

        for chunk in chunks:
            assert chunk.metadata.get("source") == "my_document.txt"

    def test_chunk_overlap_preserves_context(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=30, separators=[" "]
        )
        doc = Document(
            page_content=" ".join([f"word{i}" for i in range(50)]),
            metadata={"source": "test"}
        )
        chunks = splitter.split_documents([doc])

        if len(chunks) >= 2:
            chunk0_words = set(chunks[0].page_content.split())
            chunk1_words = set(chunks[1].page_content.split())
            overlap = chunk0_words.intersection(chunk1_words)
            assert len(overlap) > 0, "Adjacent chunks should share words"

    def test_empty_text_produces_no_chunks(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        doc = Document(page_content="   ", metadata={"source": "empty"})
        chunks = splitter.split_documents([doc])

        meaningful_chunks = [c for c in chunks if c.page_content.strip()]
        assert len(meaningful_chunks) == 0


# ── INTEGRATION TESTS: INGESTION ─────────────────────────────────

class TestIngestion:
    """Integration tests using in-memory ChromaDB."""

    def test_ingest_text_returns_chunk_count(self, mock_embeddings, sample_text, patch_vectorstore):
        """ingest_text() should return the number of chunks created."""
        from core.ingest import ingest_text

        chunks_added, total = ingest_text(sample_text, source_name="test_doc")

        assert total > 0, "Text splitter should produce at least one chunk"
        assert chunks_added >= 0, "chunks_added must be non-negative"

    def test_ingest_text_prevents_duplicates(self, mock_embeddings, sample_text, patch_vectorstore):
        """
        Ingesting the same document twice should not add duplicate chunks.
        """
        from core.ingest import ingest_text

        unique_source = f"test_doc_{uuid.uuid4().hex[:8]}"

        # First ingest
        first_count, total = ingest_text(sample_text, source_name=unique_source)
        assert total > 0, "Text splitter should produce chunks"

        # Second ingest — same source name, same chunk IDs → 0 new chunks
        second_count, _ = ingest_text(sample_text, source_name=unique_source)
        assert second_count == 0, "Re-ingesting same document should add 0 chunks"

    def test_ingest_file_loads_txt(self, mock_embeddings, sample_txt_file, patch_vectorstore):
        """ingest_file() should load and ingest a TXT file."""
        from core.ingest import ingest_file

        chunks_added, total = ingest_file(sample_txt_file)

        assert total > 0, "TXT file should produce at least one chunk"

    def test_ingest_unsupported_file_returns_zero(self, mock_embeddings, tmp_path, patch_vectorstore):
        """Unsupported file types should return (0, 0) without crashing."""
        from core.ingest import ingest_file

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4")

        chunks_added, total = ingest_file(str(csv_file))

        assert chunks_added == 0
        assert total == 0