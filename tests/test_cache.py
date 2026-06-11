# ─────────────────────────────────────────────────────────────────
# tests/test_cache.py — Tests for Redis cache layer
#
# WHAT THIS FILE TESTS:
#   - Cache stores and retrieves answers correctly
#   - Cache returns None on miss (question not seen before)
#   - Cache works when Redis is unavailable (graceful fallback)
#   - Cache TTL and key generation work correctly
#   - Clear cache removes all entries
# ─────────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock, patch


class TestCacheKeyGeneration:
    """
    Tests for cache key generation.
    Keys must be consistent — same question always produces same key.
    """

    def test_same_question_same_key(self):
        """
        The same question must always produce the same cache key.
        If keys are inconsistent, every query is a cache miss.
        """
        from core.cache import make_cache_key

        key1 = make_cache_key("What is IaaS?")
        key2 = make_cache_key("What is IaaS?")

        assert key1 == key2, "Same question must produce identical cache keys"

    def test_different_questions_different_keys(self):
        """
        Different questions must produce different cache keys.
        If all questions hash to the same key, every answer is wrong.
        """
        from core.cache import make_cache_key

        key1 = make_cache_key("What is IaaS?")
        key2 = make_cache_key("What is PaaS?")

        assert key1 != key2, "Different questions must produce different keys"

    def test_key_is_case_insensitive(self):
        """
        "What is IaaS?" and "what is iaas?" should hit the same cache entry.
        Users don't think about capitalisation when typing questions.
        """
        from core.cache import make_cache_key

        key1 = make_cache_key("What is IaaS?")
        key2 = make_cache_key("WHAT IS IAAS?")
        key3 = make_cache_key("what is iaas?")

        assert key1 == key2 == key3, \
            "Cache keys should be case-insensitive"

    def test_key_ignores_whitespace(self):
        """
        Leading/trailing whitespace should not produce different keys.
        Users might accidentally add spaces when typing.
        """
        from core.cache import make_cache_key

        key1 = make_cache_key("What is IaaS?")
        key2 = make_cache_key("  What is IaaS?  ")

        assert key1 == key2, "Cache keys should ignore leading/trailing whitespace"

    def test_key_has_correct_prefix(self):
        """
        Keys must start with 'cache:exact:' for proper Redis namespacing.
        This allows selective clearing of only cache keys.
        """
        from core.cache import make_cache_key

        key = make_cache_key("What is IaaS?")

        assert key.startswith("cache:exact:"), \
            "Cache key must start with 'cache:exact:' prefix"


class TestCacheWithRedis:
    """
    Tests for cache operations using a mocked Redis client.

    WHY MOCK REDIS IN TESTS?
    "We don't want tests to depend on a running Redis instance.
     Mocking Redis lets us test the cache logic in isolation —
     we verify our code calls Redis correctly, not that Redis works."
    """

    @pytest.fixture
    def mock_redis(self, monkeypatch):
        """
        Replaces the real Redis client with a mock that stores
        data in a Python dict — behaves like Redis but in-memory.
        """
        storage = {}  # in-memory store mimicking Redis

        mock_client = MagicMock()

        # Mock .get() — returns stored value or raises exception
        def fake_get(key):
            if key in storage:
                return storage[key]
            raise Exception("Key not found")

        # Mock .setex() — stores with TTL (we ignore TTL in tests)
        def fake_setex(key, ttl, value):
            storage[key] = value
            return True

        # Mock .ping() — simulates successful connection
        mock_client.ping.return_value = True
        mock_client.get.side_effect = fake_get
        mock_client.setex.side_effect = fake_setex
        mock_client.keys.return_value = list(storage.keys())
        mock_client.delete.return_value = True

        # Patch get_redis_client to return our mock
        monkeypatch.setattr("core.cache.get_redis_client", lambda: mock_client)

        return mock_client, storage

    def test_cache_miss_returns_none(self, mock_redis):
        """
        A question that has never been asked should return None.
        None tells the pipeline to run the full LLM query.
        """
        from core.cache import get_cached_answer

        result = get_cached_answer("What is a completely new question?")

        assert result is None, "Cache miss must return None"

    def test_cache_hit_returns_answer(self, mock_redis):
        """
        After storing an answer, retrieving it should return the same data.
        This is the core cache functionality.
        """
        from core.cache import get_cached_answer, set_cached_answer

        question = "What is IaaS?"
        answer = "IaaS provides virtualised computing resources."
        sources = ["cloud_doc.txt"]

        # Store the answer
        set_cached_answer(question, answer, sources)

        # Retrieve it
        cached = get_cached_answer(question)

        assert cached is not None, "Should find the cached answer"
        assert cached["answer"] == answer, "Cached answer must match stored answer"
        assert cached["sources"] == sources, "Cached sources must match stored sources"
        assert cached.get("cached") is True, "Cached flag must be True"

    def test_set_cached_answer_stores_all_fields(self, mock_redis):
        """
        set_cached_answer must store answer, sources, AND the cached flag.
        The UI uses the cached flag to show the ⚡ indicator.
        """
        import json
        from core.cache import set_cached_answer, make_cache_key

        mock_client, storage = mock_redis

        question = "What is BGP?"
        answer = "BGP is the routing protocol of the internet."
        sources = ["networking_doc.txt"]

        set_cached_answer(question, answer, sources)

        # Verify the stored data structure
        key = make_cache_key(question)
        stored = json.loads(storage[key])

        assert "answer" in stored
        assert "sources" in stored
        assert "cached" in stored
        assert stored["cached"] is True


class TestCacheGracefulFallback:
    """
    Tests for behaviour when Redis is unavailable.

    WHY TEST REDIS FAILURE?
    "Redis going down shouldn't crash the application. The cache
     is a performance optimisation — without it, queries are slower
     but still work. This is called graceful degradation."
    """

    @pytest.fixture
    def redis_unavailable(self, monkeypatch):
        """Makes Redis appear unavailable by returning None from get_redis_client."""
        monkeypatch.setattr("core.cache.get_redis_client", lambda: None)

    def test_get_cached_answer_returns_none_when_redis_down(
        self, redis_unavailable
    ):
        """
        When Redis is down, get_cached_answer should return None (cache miss).
        The pipeline then runs normally — slower but correct.
        """
        from core.cache import get_cached_answer

        result = get_cached_answer("What is cloud computing?")

        assert result is None, \
            "When Redis is unavailable, should return None (not crash)"

    def test_set_cached_answer_silent_when_redis_down(self, redis_unavailable):
        """
        When Redis is down, set_cached_answer should fail silently.
        The answer is still returned to the user — just not cached.
        """
        from core.cache import set_cached_answer

        # Should not raise any exception
        try:
            set_cached_answer(
                "What is IaaS?",
                "IaaS answer here",
                ["source.txt"]
            )
        except Exception as e:
            pytest.fail(
                f"set_cached_answer raised an exception when Redis is down: {e}"
            )

    def test_get_cache_stats_returns_none_when_redis_down(
        self, redis_unavailable
    ):
        """
        Cache stats should return None when Redis is down.
        The UI uses this to show the ⚠️ Redis unavailable message.
        """
        from core.cache import get_cache_stats

        stats = get_cache_stats()

        assert stats is None, \
            "Cache stats should be None when Redis is unavailable"

    def test_clear_cache_returns_false_when_redis_down(self, redis_unavailable):
        """
        clear_cache() should return False when Redis is down, not crash.
        """
        from core.cache import clear_cache

        result = clear_cache()

        assert result is False, \
            "clear_cache should return False when Redis is unavailable"