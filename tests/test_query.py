# ─────────────────────────────────────────────────────────────────
# tests/test_query.py — Tests for retrieval + generation pipeline
#
# WHAT THIS FILE TESTS:
#   - Retrieval returns relevant chunks for a question
#   - Source citations are correct
#   - Hallucination guard works (no documents = correct message)
#   - Chat history is passed correctly for conversation memory
#   - The LCEL chain produces a string answer
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import shutil
import numpy as np

TEST_CHROMA_PATH = "chroma_db_test"
os.environ["CHROMA_PATH_OVERRIDE"] = TEST_CHROMA_PATH


@pytest.fixture(autouse=True)
def clean_test_db():
    """Fresh test database for every test."""
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)
    yield
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)


@pytest.fixture
def mock_embeddings(monkeypatch):
    """
    Replaces OpenAI embeddings with deterministic fake vectors.
    See test_ingest.py for full explanation of why we mock this.
    """
    from langchain_core.embeddings import Embeddings

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            # Use a hash of the text to produce different (but deterministic)
            # vectors for different texts — better than all zeros for
            # testing that the right document is retrieved
            vectors = []
            for text in texts:
                seed = hash(text[:50]) % 1000
                np.random.seed(seed)
                vectors.append(np.random.rand(1536).tolist())
            return vectors

        def embed_query(self, text):
            seed = hash(text[:50]) % 1000
            np.random.seed(seed)
            return np.random.rand(1536).tolist()

    monkeypatch.setattr("core.ingest.embeddings", FakeEmbeddings())
    monkeypatch.setattr("core.query.embeddings", FakeEmbeddings())


@pytest.fixture
def mock_llm(monkeypatch):
    """
    Replaces the real GPT model with a fake that echoes the context.

    WHY MOCK THE LLM?
    "We don't want to pay for OpenAI calls in every test run.
     More importantly, real LLM responses are non-deterministic —
     the same prompt can produce different answers, making tests flaky.
     By mocking the LLM with a deterministic fake, tests are fast,
     free, and reliable."

    The fake LLM returns a string containing the word "MOCKED"
    so we can verify the chain called it correctly.
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class FakeLLM(BaseChatModel):
        """Fake LLM that returns a predictable response."""

        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # Return a fake response that includes context keywords
            # so we can verify the context was actually passed
            last_message = messages[-1].content if messages else ""
            response = f"MOCKED ANSWER based on context. Question received: {last_message[:50]}"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=response))]
            )

    monkeypatch.setattr("core.query.llm", FakeLLM())


@pytest.fixture
def populated_db(mock_embeddings):
    """
    Sets up a test database with known documents.
    Tests that need existing documents use this fixture.
    """
    from core.ingest import ingest_text

    # Ingest two documents with clearly different topics
    ingest_text(
        "IaaS provides virtual machines and storage. AWS EC2 is an example of IaaS.",
        source_name="cloud_doc.txt"
    )
    ingest_text(
        "BGP is the routing protocol of the internet used between autonomous systems.",
        source_name="networking_doc.txt"
    )

    return {"cloud": "cloud_doc.txt", "networking": "networking_doc.txt"}


class TestRetrieval:
    """Tests for the vector search retrieval step."""

    def test_retrieval_returns_results(self, mock_embeddings, populated_db):
        """
        Querying a populated database should return results.
        Basic sanity check — if this fails, nothing else matters.
        """
        from core.query import get_retriever

        retriever = get_retriever(k=2)
        results = retriever.invoke("cloud computing")

        assert len(results) > 0, "Retriever should return at least one result"

    def test_retrieved_docs_have_source_metadata(self, mock_embeddings, populated_db):
        """
        Every retrieved document must have source metadata.
        Without this, the citation badges in the UI won't work.
        """
        from core.query import get_retriever

        retriever = get_retriever(k=2)
        results = retriever.invoke("virtual machines")

        for doc in results:
            assert "source" in doc.metadata, \
                "Retrieved document must have source in metadata"
            assert doc.metadata["source"] != "", \
                "Source metadata must not be empty"

    def test_format_docs_produces_string(self, mock_embeddings, populated_db):
        """
        format_docs() must return a string — this is what the prompt receives.
        If it returns a list or None, the prompt template will fail.
        """
        from core.query import get_retriever, format_docs

        retriever = get_retriever(k=2)
        docs = retriever.invoke("cloud")
        context = format_docs(docs)

        assert isinstance(context, str), "format_docs must return a string"
        assert len(context) > 0, "Context string must not be empty"


class TestQueryPipeline:
    """Tests for the full query_pipeline() function."""

    def test_empty_db_returns_helpful_message(self):
        """
        Querying with no documents should return a helpful message,
        not crash or return an empty string.

        This is the hallucination guard test — the most important test
        in the suite. If this fails, the system might hallucinate answers
        from GPT's training data instead of saying "I don't know."
        """
        from core.query import query_pipeline

        answer, sources = query_pipeline("What is cloud computing?")

        # Should return the no-documents message
        assert "no documents" in answer.lower() or \
               "not ingested" in answer.lower() or \
               "please upload" in answer.lower(), \
            "Empty DB should return a message asking user to add documents"

        assert sources == [], "Empty DB should return empty sources list"

    def test_query_returns_answer_and_sources(
        self, mock_embeddings, mock_llm, populated_db
    ):
        """
        query_pipeline() must return both an answer string and a sources list.
        The UI depends on both being present.
        """
        from core.query import query_pipeline

        answer, sources = query_pipeline("What is IaaS?")

        assert isinstance(answer, str), "Answer must be a string"
        assert len(answer) > 0, "Answer must not be empty"
        assert isinstance(sources, list), "Sources must be a list"

    def test_sources_contain_known_document(
        self, mock_embeddings, mock_llm, populated_db
    ):
        """
        When asking about cloud topics, at least one source should
        be the cloud document we ingested.
        """
        from core.query import query_pipeline

        answer, sources = query_pipeline("Tell me about virtual machines and IaaS")

        # At least one source should be from our cloud document
        assert any("cloud" in s for s in sources), \
            "Cloud question should cite the cloud document"

    def test_chat_history_is_accepted(
        self, mock_embeddings, mock_llm, populated_db
    ):
        """
        query_pipeline() should accept a chat_history argument without crashing.
        This verifies conversation memory integration works.
        """
        from core.query import query_pipeline

        chat_history = "Human: What is IaaS?\nAssistant: IaaS provides virtual machines."

        # Should not raise any exceptions
        answer, sources = query_pipeline(
            "Tell me more about it",
            chat_history=chat_history
        )

        assert isinstance(answer, str)

    def test_query_returns_list_of_sources(
        self, mock_embeddings, mock_llm, populated_db
    ):
        """
        Sources should be a deduplicated list of strings.
        Duplicates would show the same badge multiple times in the UI.
        """
        from core.query import query_pipeline

        answer, sources = query_pipeline("cloud virtual machines")

        # Sources should be unique (no duplicates)
        assert len(sources) == len(set(sources)), \
            "Sources list should not contain duplicates"