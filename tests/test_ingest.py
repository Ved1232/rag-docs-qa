# ─────────────────────────────────────────────────────────────────
# tests/test_ingest.py — Tests for document ingestion pipeline
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import uuid
import numpy as np

os.environ["CHROMA_PATH_OVERRIDE"] = ":memory:"


# ── FIXTURES ──────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings(monkeypatch):
    """
    Fake embeddings — returns deterministic vectors, no OpenAI calls.
    Returned so patch_vectorstore can inject it into the Chroma store.
    """
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
    return fake  # returned so patch_vectorstore can use it


@pytest.fixture(autouse=True)
def patch_vectorstore(monkeypatch, mock_embeddings):
    """
    Replaces get_vectorstore() with an in-memory ChromaDB instance.
    Depends on mock_embeddings so the fake model is passed into Chroma.

    WHY embedding_function=mock_embeddings?
    When add_documents() is called, Chroma calls embed_documents()
    on the embedding_function. Without it, Chroma raises:
    'You must provide an embedding function'
    Passing mock_embeddings here mirrors production behaviour exactly.
    """
    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.EphemeralClient()
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=mock_embeddings  # required for add_documents()
    )

    def make_in_memory_store(namespace=None):
        return store

    monkeypatch.setattr("core.ingest.get_vectorstore", make_in_memory_store)
    monkeypatch.setattr("core.query.get_vectorstore", make_in_memory_store)


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
    """Tests for chunking logic — no DB writes needed."""

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
            assert len(overlap) > 0

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
    """Integration tests using in-memory ChromaDB + fake embeddings."""

    def test_ingest_text_returns_chunk_count(self, sample_text):
        from core.ingest import ingest_text

        chunks_added, total = ingest_text(sample_text, source_name="test_doc")

        assert total > 0, "Text splitter should produce at least one chunk"
        assert chunks_added > 0, f"Expected chunks to be added, got {chunks_added}"
        assert chunks_added == total, "All chunks should be new on first ingest"

    def test_ingest_text_prevents_duplicates(self, sample_text):
        from core.ingest import ingest_text

        unique_source = f"test_doc_{uuid.uuid4().hex[:8]}"

        # First ingest
        first_count, total = ingest_text(sample_text, source_name=unique_source)
        assert total > 0, "Text splitter should produce chunks"
        assert first_count > 0, f"First ingest should add chunks, got {first_count}"

        # Second ingest — same source name → same chunk IDs → 0 new chunks
        second_count, _ = ingest_text(sample_text, source_name=unique_source)
        assert second_count == 0, "Re-ingesting same document should add 0 new chunks"

    def test_ingest_stores_source_metadata(self, sample_text):
        from core.ingest import ingest_text, get_vectorstore

        ingest_text(sample_text, source_name="my_test_document.txt")

        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search("cloud computing", k=1)

        assert len(results) > 0, "Should find at least one result"
        assert results[0].metadata.get("source") == "my_test_document.txt"

    def test_ingest_file_loads_txt(self, sample_txt_file):
        from core.ingest import ingest_file

        chunks_added, total = ingest_file(sample_txt_file)

        assert total > 0, "TXT file should produce at least one chunk"
        assert chunks_added > 0

    def test_ingest_unsupported_file_returns_zero(self, tmp_path):
        from core.ingest import ingest_file

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4")

        chunks_added, total = ingest_file(str(csv_file))

        assert chunks_added == 0
        assert total == 0