"""Tests for the MCP server's IDE-facing result formatting.

The recall tool condenses engine results into a string for the IDE. The
classic trap this guards against: hybrid recall yields BM25-only candidates
with ``distance=None`` (no vector match), and a naive ``1 - distance`` does
``1 - None`` -> ``TypeError: int - NoneType``, breaking every recall that
returned a BM25-only hit. These are pure-function tests (no DB / model / HTTP).

Run:  pytest -q tests/test_mcp_server.py
"""
import pytest

# The mcp package is an optional extra (pyproject: [project.optional-dependencies].mcp).
# Skip the whole module when it isn't installed (e.g. CI only installs ".[dev]");
# _format_recall itself is a pure function, but importing engine.mcp_server pulls
# in FastMCP at module level.
pytest.importorskip(
    "mcp", reason="mcp extra not installed (pip install '.[mcp]')"
)

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


# --- social tools (formatting via monkeypatched HTTP layer) ---------------

from engine import mcp_server


def test_social_get_fan_formats_card(monkeypatch):
    monkeypatch.setattr(mcp_server, "_get", lambda path, params=None: {
        "fan": {"nickname": "墨迹一号", "relation": "friend", "favor": 95,
                "impressions": "第一天打卡\n连投三硬币", "traits": "爱用🐶"}
    })
    out = mcp_server.social_get_fan("bilibili", "10086")
    assert "friend" in out and "95/100" in out
    assert "墨迹一号" in out and "连投三硬币" in out


def test_social_get_fan_missing(monkeypatch):
    monkeypatch.setattr(mcp_server, "_get", lambda path, params=None: {"fan": None})
    out = mcp_server.social_get_fan("bilibili", "ghost")
    assert "no card yet" in out


def test_social_update_fan_omits_defaults(monkeypatch):
    seen = {}

    def fake_post(path, payload):
        seen.update(payload)
        return {"fan": {"relation": "fan", "favor": 55}}

    monkeypatch.setattr(mcp_server, "_post", fake_post)
    out = mcp_server.social_update_fan("bilibili", "9",
                                       favor_delta=5, impression="好人")
    assert seen["favor_delta"] == 5 and seen["impression"] == "好人"
    assert "favor" not in seen and "nickname" not in seen, "defaults must not leak"
    assert "relation=fan" in out


def test_social_mood_read_and_set(monkeypatch):
    monkeypatch.setattr(mcp_server, "_get", lambda path, params=None: {
        "current": {"label": "炸毛", "valence": -40, "arousal": 80,
                    "updated_at": "2026-10-25T20:00:00"},
        "events": [{"ts": "2026-10-25T20:00:00", "label": "炸毛",
                    "reason": "被比较"}],
    })
    out = mcp_server.social_mood()
    assert "炸毛" in out and "被比较" in out

    posted = {}

    def fake_post(path, payload):
        posted.update(payload)
        return {"mood": {"label": payload["label"], "valence": 10, "arousal": 20}}

    monkeypatch.setattr(mcp_server, "_post", fake_post)
    out = mcp_server.social_mood(label="开心", valence=10, arousal=20, reason="涨粉")
    assert posted["label"] == "开心" and "mood set" in out
