"""P3a directory tests: engram upsert/refresh/lookup.

The engram layer is a LIVING directory (unlike letters): refresh rewrites
state_summary, unions member_rowids, and nudges confidence up. The stored
vector embeds ONLY the concept name (question -> name matching), so tests
query the concept name itself for a deterministic distance-0 hit.
"""
import pytest

from engine.engram import list_top, lookup, upsert


def test_upsert_create_and_lookup():
    concept, action = upsert("memory engine architecture",
                             "hybrid retrieval + typology + letters mailbox, 98 tests")
    assert action == "created"
    hits = lookup("memory engine architecture", top_k=3)
    assert len(hits) == 1
    assert hits[0]["concept"] == concept
    assert "typology" in hits[0]["state_summary"]


def test_refresh_rewrites_summary_and_unions_members():
    upsert("concept A", "state v1: early days", member_rowids=[1, 2])
    concept, action = upsert("concept A", "state v2: shipped",
                             member_rowids=[2, 3])
    assert action == "refreshed"
    hits = lookup("concept A", top_k=1)
    assert hits[0]["state_summary"] == "state v2: shipped"
    assert hits[0]["member_count"] == 3          # {1,2} u {2,3}
    assert hits[0]["confidence"] > 0.5           # nudged up on refresh


def test_confidence_explicit_override():
    upsert("concept B", "explicit confidence test", confidence=0.9)
    top = list_top(n=5)
    entry = [e for e in top if e["concept"] == "concept B"][0]
    assert entry["confidence"] == 0.9


def test_confidence_floor_protects_manual_trust():
    # consolidation path (confidence_floor=True) must never LOWER a manually
    # earned confidence when its cluster name lands on an existing concept
    upsert("concept C", "manually trusted", confidence=0.9)
    upsert("concept C", "auto consolidation refresh", confidence=0.4,
           confidence_floor=True)
    top = list_top(n=5)
    entry = [e for e in top if e["concept"] == "concept C"][0]
    assert entry["confidence"] == 0.9          # not diluted to 0.4
    # without the floor the explicit value still wins (manual override stays
    # a real override)
    upsert("concept C", "manual override down", confidence=0.2)
    top = list_top(n=5)
    entry = [e for e in top if e["concept"] == "concept C"][0]
    assert entry["confidence"] == 0.2


def test_lookup_miss_gated_by_distance():
    upsert("quantum needlework embroidery", "an unrelated concept for gating")
    assert lookup("glibc allocator internals 2026 changelog", top_k=3) == []


def test_list_top_orders_by_confidence():
    upsert("lo concept", "low confidence entry", confidence=0.2)
    upsert("hi concept", "high confidence entry", confidence=0.95)
    top = list_top(n=2)
    assert top[0]["concept"] == "hi concept"
    assert top[1]["concept"] == "lo concept"


def test_lookup_bumps_hits():
    upsert("hit counter concept", "hits before lookup")
    before = [e for e in list_top(n=10)
              if e["concept"] == "hit counter concept"][0]["hits"]
    lookup("hit counter concept", top_k=1)
    after = [e for e in list_top(n=10)
             if e["concept"] == "hit counter concept"][0]["hits"]
    assert after == before + 1


def test_empty_concept_rejected():
    with pytest.raises(ValueError):
        upsert("   ", "summary without a concept")
    with pytest.raises(ValueError):
        upsert("concept without summary", "  ")
