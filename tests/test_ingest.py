# ─────────────────────────────────────────────────────────────────
# tests/test_ingest.py — Tests for document ingestion pipeline
#
# WHAT THIS FILE TESTS:
#   - Text chunking logic (correct sizes, overlap, boundary handling)
#   - Document ingestion into ChromaDB
#   - Duplicate prevention (same doc ingested twice = no duplicates)
#   - File ingestion for TXT files
#
# HOW TO RUN:
#   pytest tests/test_ingest.py -v
#
# WHY WE WRITE TESTS (interview answer):
#   "Tests give me confidence that changes don't break existing
#    behaviour. When I swap ChromaDB for Pinecone, I run the same
#    tests against the new backend. If they pass, the swap worked.
#    Without tests, I'd have to manually test every scenario after
#    every change — that doesn't scale."
#
# TESTING STRATEGY:
#   We use a SEPARATE test ChromaDB path (chroma_db_test/) so tests
#   never touch the real database. Tests clean up after themselves
#   using pytest fixtures with yield (setup → test → teardown).
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import tempfile
import shutil

# ── TEST CONFIGURATION ────────────────────────────────────────────
# Use a separate ChromaDB path for tests so we never touch real data
# This is set BEFORE importing core.ingest so it picks up the override
TEST_CHROMA_PATH = "chroma_db_test"
os.environ["CHROMA_PATH_OVERRIDE"] = TEST_CHROMA_PATH


# ── FIXTURES ──────────────────────────────────────────────────────
# pytest fixtures run setup code before each test and teardown after.
# The yield keyword separates setup (before yield) from teardown (after).

@pytest.fixture(autouse=True)
def clean_test_db():
    """
    Automatically runs before and after EVERY test in this file.
    autouse=True means we don't need to include it in each test signature.

    BEFORE test: ensures a fresh, empty test database
    AFTER test: deletes the test database to avoid pollution between tests

    WHY CLEAN BETWEEN TESTS?
    If test_A ingests a document and test_B checks the document count,
    test_B would fail because it inherited test_A's data. Each test
    must start from a known empty state.
    """
    # Setup — remove test DB if it exists from a previous failed run
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)

    yield  # test runs here

    # Teardown — clean up after the test
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)


@pytest.fixture
def sample_text():
    """
    Returns a sample text document for testing.
    Using a fixture means we write this once and reuse across tests.
    """
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
    """
    Creates a temporary TXT file for file ingestion tests.
    tmp_path is a pytest built-in fixture that provides a temp directory.
    The file is automatically deleted after the test.
    """
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    return str(file_path)


# ── UNIT TESTS: TEXT SPLITTER ─────────────────────────────────────
# These tests check the chunking logic in isolation
# without actually calling OpenAI or ChromaDB

class TestChunking:
    """
    Tests for the RecursiveCharacterTextSplitter configuration.

    WHY TEST CHUNKING SEPARATELY?
    Chunking is the foundation of retrieval quality. If chunks are
    too large, retrieval is imprecise. If too small, context is lost.
    Testing it in isolation lets us verify the behaviour without
    the cost of embedding calls.
    """

    def test_text_splits_into_multiple_chunks(self):
        """
        A long text should produce multiple chunks.
        If everything ends up in one chunk, retrieval can't be precise.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        # Create a text that's definitely longer than 200 characters
        long_text = "This is a test sentence. " * 20  # 500+ characters

        doc = Document(page_content=long_text, metadata={"source": "test"})
        chunks = splitter.split_documents([doc])

        # Should have more than 1 chunk
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunks_inherit_metadata(self):
        """
        Each chunk must carry the source metadata from the parent document.
        Without this, we can't show citations in the UI.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

        doc = Document(
            page_content="Test content " * 30,
            metadata={"source": "my_document.txt"}
        )
        chunks = splitter.split_documents([doc])

        for chunk in chunks:
            assert chunk.metadata.get("source") == "my_document.txt", \
                "Every chunk must inherit the source metadata"

    def test_chunk_overlap_preserves_context(self):
        """
        With overlap=20, adjacent chunks should share some words.
        This tests that context is not lost at chunk boundaries.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=30,
            separators=[" "]  # split on spaces for predictable behaviour
        )

        doc = Document(
            page_content=" ".join([f"word{i}" for i in range(50)]),
            metadata={"source": "test"}
        )
        chunks = splitter.split_documents([doc])

        if len(chunks) >= 2:
            # Last words of chunk 0 should appear in chunk 1
            # This verifies the overlap is working
            chunk0_words = set(chunks[0].page_content.split())
            chunk1_words = set(chunks[1].page_content.split())
            overlap = chunk0_words.intersection(chunk1_words)
            assert len(overlap) > 0, "Adjacent chunks should share words due to overlap"

    def test_empty_text_produces_no_chunks(self):
        """
        Empty text should not produce any chunks.
        This prevents empty embeddings being stored in ChromaDB.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        doc = Document(page_content="   ", metadata={"source": "empty"})
        chunks = splitter.split_documents([doc])

        # Whitespace-only text should produce 0 chunks
        meaningful_chunks = [c for c in chunks if c.page_content.strip()]
        assert len(meaningful_chunks) == 0


# ── INTEGRATION TESTS: INGESTION ─────────────────────────────────
# These tests call ingest_text() for real but use a mock embeddings
# to avoid OpenAI API costs during testing.
# We use pytest's monkeypatch to replace the real embeddings with a fake.

class TestIngestion:
    """
    Integration tests for the full ingestion pipeline.
    Uses mock embeddings to avoid OpenAI API calls.
    """

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """
        Replaces OpenAI embeddings with a deterministic fake.

        WHY MOCK EMBEDDINGS IN TESTS?
        "Real embedding calls cost money and add latency to every
         test run. By mocking the embeddings model, our tests run
         in milliseconds and don't require an API key. We test the
         pipeline logic — chunking, storage, retrieval — not
         OpenAI's API."

        The fake embedding returns a fixed-length list of zeros.
        This is enough to test that data is stored and retrieved —
        we don't need real semantic similarity for unit tests.
        """
        import numpy as np
        from langchain_core.embeddings import Embeddings

        class FakeEmbeddings(Embeddings):
            """Fake embeddings that return zeros — fast and free."""

            def embed_documents(self, texts):
                # Return a 1536-dim zero vector for each text
                # 1536 matches text-embedding-3-small dimensions
                return [np.zeros(1536).tolist() for _ in texts]

            def embed_query(self, text):
                return np.zeros(1536).tolist()

        # Patch the embeddings object in the ingest module
        monkeypatch.setattr("core.ingest.embeddings", FakeEmbeddings())
        monkeypatch.setattr("core.query.embeddings", FakeEmbeddings())

    def test_ingest_text_returns_chunk_count(self, mock_embeddings, sample_text):
        """
        ingest_text() should return the number of chunks created.
        This tells the UI how many chunks were indexed.
        """
        from core.ingest import ingest_text

        chunks_added, total = ingest_text(sample_text, source_name="test_doc")

        assert chunks_added > 0, "Should have added at least one chunk"
        assert total > 0, "Total chunks should be greater than 0"
        assert chunks_added == total, "All chunks should be new (first ingest)"

    def test_ingest_text_prevents_duplicates(self, mock_embeddings, sample_text):
        """
        Ingesting the same document twice should not add duplicate chunks.
        This prevents the vector store from growing unboundedly.
        """
        from core.ingest import ingest_text

        # First ingest
        first_count, _ = ingest_text(sample_text, source_name="test_doc")
        assert first_count > 0

        # Second ingest of the same document
        second_count, _ = ingest_text(sample_text, source_name="test_doc")

        # No new chunks should be added (duplicates prevented)
        assert second_count == 0, \
            "Re-ingesting the same document should add 0 new chunks"

    def test_ingest_stores_source_metadata(self, mock_embeddings, sample_text):
        """
        Ingested chunks must have the correct source metadata.
        This is used for citation badges in the UI.
        """
        from core.ingest import ingest_text, get_vectorstore

        ingest_text(sample_text, source_name="my_test_document.txt")

        # Query the vectorstore to verify metadata was stored
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search("cloud computing", k=1)

        assert len(results) > 0, "Should find at least one result"
        assert results[0].metadata.get("source") == "my_test_document.txt", \
            "Source metadata must be preserved in stored chunks"

    def test_ingest_file_loads_txt(self, mock_embeddings, sample_txt_file):
        """
        ingest_file() should successfully load and ingest a TXT file.
        """
        from core.ingest import ingest_file

        chunks_added, total = ingest_file(sample_txt_file)

        assert chunks_added > 0, "TXT file should produce at least one chunk"

    def test_ingest_unsupported_file_returns_zero(self, mock_embeddings, tmp_path):
        """
        Unsupported file types (e.g. .csv) should return (0, 0) gracefully.
        The app should handle this without crashing.
        """
        from core.ingest import ingest_file

        # Create a fake .csv file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4")

        chunks_added, total = ingest_file(str(csv_file))

        assert chunks_added == 0, "Unsupported file type should return 0 chunks"
        assert total == 0   