"""Tests for the MCP server's IDE-facing result formatting.

The recall tool condenses engine results into a string for the IDE. The
classic trap this guards against: hybrid recall yields BM25-only candidates
with ``distance=None`` (no vector match), and a naive ``1 - distance`` does
``1 - None`` -> ``TypeError: int - NoneType``, breaking every recall that
returned a BM25-only hit. These are pure-function tests (no DB / model / HTTP).

Run:  pytest -q tests/test_mcp_server.py
"""
from engine.mcp_server import _format_recall


def test_format_recall_empty():
    assert _format_recall([]) == "(no relevant memories found)"


def test_format_recall_bm25_only_distance_none():
    """Regression: a BM25-only candidate has distance=None. Must NOT crash.

    Before the fix, ``1 - m.get("distance", 0)`` evaluated to ``1 - None``
    (the default 0 is ignored because the key *exists* with value None),
    raising TypeError and killing the whole recall tool.
    """
    results = [
        {"topic": "vector hit", "summary": "v", "distance": 0.3},
        {"topic": "bm25 only", "summary": "b", "distance": None, "rrf": 0.55},
    ]
    out = _format_recall(results)
    assert "vector hit" in out
    assert "bm25 only" in out
    assert "rrf=0.55" in out          # BM25-only relevance surfaces via rrf


def test_format_recall_distance_none_without_rel_field():
    """Defensive: distance=None and no rerank/rrf falls back to sim=0.0."""
    out = _format_recall([{"topic": "x", "summary": "y", "distance": None}])
    assert "x" in out
    assert "sim=0.00" in out


def test_format_recall_rerank_branch_with_none_distance():
    """A reranked result can also carry distance=None; rerank still wins."""
    out = _format_recall([{"topic": "t", "summary": "s", "rerank": 0.91, "distance": None}])
    assert "rerank=0.91" in out


def test_format_recall_vector_hit_uses_sim():
    out = _format_recall([{"topic": "t", "summary": "s", "distance": 0.25}])
    assert "sim=0.75" in out          # 1 - 0.25
