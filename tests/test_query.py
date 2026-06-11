# ─────────────────────────────────────────────────────────────────
# tests/test_query.py — Tests for retrieval + generation pipeline
# Uses in-memory ChromaDB — no disk writes, no CI permission issues
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import uuid
import numpy as np

os.environ["CHROMA_PATH_OVERRIDE"] = ":memory:"


@pytest.fixture(autouse=True)
def patch_vectorstore(monkeypatch):
    """In-memory ChromaDB for every test."""
    import chromadb
    from langchain_chroma import Chroma

    _store = {}  # shared store within a test

    def make_in_memory_store(namespace=None):
        key = namespace or "default"
        if key not in _store:
            client = chromadb.EphemeralClient()
            collection_name = f"test_{uuid.uuid4().hex[:8]}"
            _store[key] = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=None
            )
        return _store[key]

    monkeypatch.setattr("core.ingest.get_vectorstore", make_in_memory_store)
    monkeypatch.setattr("core.query.get_vectorstore", make_in_memory_store)


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Deterministic fake embeddings — no OpenAI API calls."""
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
def mock_llm(monkeypatch):
    """Fake LLM — returns predictable response, no OpenAI API calls."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class FakeLLM(BaseChatModel):
        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(
                    message=AIMessage(content="MOCKED ANSWER from fake LLM")
                )]
            )

    monkeypatch.setattr("core.query.llm", FakeLLM())


@pytest.fixture
def populated_db(mock_embeddings, patch_vectorstore):
    """Ingests two test documents into the in-memory store."""
    from core.ingest import ingest_text

    ingest_text(
        "IaaS provides virtual machines and storage. AWS EC2 is IaaS.",
        source_name="cloud_doc.txt"
    )
    ingest_text(
        "BGP is the routing protocol of the internet between autonomous systems.",
        source_name="networking_doc.txt"
    )
    return {"cloud": "cloud_doc.txt", "networking": "networking_doc.txt"}


class TestRetrieval:

    def test_retrieval_returns_results(self, mock_embeddings, populated_db):
        from core.query import get_retriever
        retriever = get_retriever(k=2)
        results = retriever.invoke("cloud computing")
        assert len(results) > 0, "Retriever should return at least one result"

    def test_retrieved_docs_have_source_metadata(self, mock_embeddings, populated_db):
        from core.query import get_retriever
        retriever = get_retriever(k=2)
        results = retriever.invoke("virtual machines")
        for doc in results:
            assert "source" in doc.metadata
            assert doc.metadata["source"] != ""

    def test_format_docs_produces_string(self, mock_embeddings, populated_db):
        from core.query import get_retriever, format_docs
        retriever = get_retriever(k=2)
        docs = retriever.invoke("cloud")
        context = format_docs(docs)
        assert isinstance(context, str)
        assert len(context) > 0


class TestQueryPipeline:

    def test_empty_db_returns_helpful_message(self, patch_vectorstore):
        """Most important test — empty DB must NOT hallucinate."""
        from core.query import query_pipeline
        answer, sources = query_pipeline("What is cloud computing?")
        assert (
            "no documents" in answer.lower() or
            "not ingested" in answer.lower() or
            "please upload" in answer.lower()
        ), f"Empty DB should return helpful message, got: {answer}"
        assert sources == []

    def test_query_returns_answer_and_sources(
        self, mock_embeddings, mock_llm, populated_db
    ):
        from core.query import query_pipeline
        answer, sources = query_pipeline("What is IaaS?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)

    def test_sources_contain_known_document(
        self, mock_embeddings, mock_llm, populated_db
    ):
        from core.query import query_pipeline
        answer, sources = query_pipeline("Tell me about virtual machines and IaaS")
        assert any("cloud" in s for s in sources), \
            f"Expected cloud_doc.txt in sources, got: {sources}"

    def test_chat_history_is_accepted(
        self, mock_embeddings, mock_llm, populated_db
    ):
        from core.query import query_pipeline
        chat_history = "Human: What is IaaS?\nAssistant: IaaS provides virtual machines."
        answer, sources = query_pipeline("Tell me more", chat_history=chat_history)
        assert isinstance(answer, str)

    def test_sources_are_deduplicated(
        self, mock_embeddings, mock_llm, populated_db
    ):
        from core.query import query_pipeline
        answer, sources = query_pipeline("cloud virtual machines")
        assert len(sources) == len(set(sources)), "Sources must not contain duplicates"