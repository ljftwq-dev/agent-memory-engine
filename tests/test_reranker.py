"""Tests for the optional cross-encoder reranker stage in recall.

Runs entirely offline (conftest forces HF_HUB_OFFLINE=1 and a nonexistent
embed model), so the real bge-reranker-v2-m3 can't load here. We verify:

1. The reranker is off by default (config), so recall keeps the RRF path.
2. Forcing rerank=True when the model can't load degrades gracefully (keeps
   RRF, no crash, no ``rerank`` field on results).
3. With a mocked cross-encoder, recall promotes candidates the model prefers
   and tags them with a normalized ``rerank`` score.

Run:  pytest -q tests/test_reranker.py
"""
import pytest

from engine import config, reranker
from engine.remember import remember
from engine.recall import recall


@pytest.fixture(autouse=True)
def _reset_reranker_singleton():
    """Reset the lazy singleton between tests so each starts clean."""
    reranker._MODEL = None
    reranker._MODE = None
    yield
    reranker._MODEL = None
    reranker._MODE = None


def test_reranker_disabled_by_default():
    """The reranker is opt-in (off by default in config)."""
    assert config.reranker_enable() is False
    assert config.reranker_pool() > 0
    # check the shipped default model directly (conftest overrides the env for tests)
    assert "bge-reranker" in config.DEFAULTS["RERANKER_MODEL"]


def test_reranker_off_keeps_rrf():
    """rerank=False keeps the pure hybrid path (rrf field, no rerank field)."""
    remember(topic="t", summary="python pytest fixture scope session")
    results = recall("python pytest fixture scope session", top_k=3, rerank=False)
    assert len(results) == 1
    assert "rrf" in results[0]
    assert "rerank" not in results[0]


def test_reranker_unavailable_degrades_to_rrf():
    """Forcing rerank=True when the model can't load keeps RRF, no crash.

    conftest pins HF offline + a nonexistent embed model, so loading the
    cross-encoder fails -> reranker returns None -> recall keeps the fused
    relevance exactly as if rerank were off.
    """
    remember(topic="t", summary="python pytest fixture scope session")
    results = recall("python pytest fixture scope session", top_k=3, rerank=True)
    assert reranker.mode() == "disabled"
    assert len(results) == 1
    # fell back to RRF, not cross-encoder scores
    assert "rrf" in results[0]
    assert "rerank" not in results[0]


def test_reranker_mocked_promotes_relevant():
    """A mocked cross-encoder re-orders results by its own scores.

    Two memories share the keyword 'python' (both recalled), but the fake
    cross-encoder prefers the one mentioning 'pandas'. The reranker must
    promote it to rank 1 and tag both with normalized ``rerank`` scores
    (best -> 1.0 within the batch).
    """
    remember(topic="t1", summary="python web framework django flask")
    remember(topic="t2", summary="python data science pandas numpy")

    class FakeCrossEncoder:
        def predict(self, pairs):
            # high score when the candidate text mentions 'pandas'
            return [10.0 if "pandas" in text else 1.0 for _, text in pairs]

    reranker._MODEL = FakeCrossEncoder()
    reranker._MODE = "cross-encoder"

    results = recall("python pandas data", top_k=2, rerank=True)
    assert len(results) == 2
    # the pandas memory is promoted to the top by the reranker
    assert results[0]["summary"] == "python data science pandas numpy"
    assert "rerank" in results[0]
    # min-max normalization: best in batch -> 1.0
    assert results[0]["rerank"] == pytest.approx(1.0, abs=1e-6)
    assert results[-1]["rerank"] == pytest.approx(0.0, abs=1e-6)
    # final score honors the new relevance
    alpha = config.alpha()
    expected = alpha * results[0]["strength_now"] + (1.0 - alpha) * results[0]["rerank"]
    assert results[0]["score"] == pytest.approx(expected, abs=1e-6)


def test_reranker_returns_none_for_empty():
    """rerank() short-circuits on an empty candidate list (no model load)."""
    assert reranker.rerank("anything", []) is None


def test_reranker_handles_identical_scores():
    """When the cross-encoder returns identical scores, relevance stays neutral
    (0.5) and ranking is left to the rest of the score."""
    remember(topic="t1", summary="python one thing")
    remember(topic="t2", summary="python two thing")

    class FlatCrossEncoder:
        def predict(self, pairs):
            return [3.0 for _ in pairs]

    reranker._MODEL = FlatCrossEncoder()
    reranker._MODE = "cross-encoder"

    results = recall("python", top_k=2, rerank=True)
    assert len(results) == 2
    for r in results:
        assert r["rerank"] == pytest.approx(0.5, abs=1e-9)
