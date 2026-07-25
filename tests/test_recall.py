"""Tests for the two-stage recall pipeline, gating, dedup, and Ebbinghaus decay.

Runs entirely on the hash-fallback embedder (no BGE-m3 download needed), so it
works in any CI / offline environment.

Run:  pytest -q
"""
from datetime import datetime, timedelta

from engine import config, db, embed
from engine.remember import remember
from engine.recall import recall, _strength_now
from engine.forget import decay_all


def test_remember_creates():
    rid, action = remember(topic="t", summary="python pytest fixture scope session")
    assert action == "created"
    assert rid > 0


def test_recall_exact_match_returns_fields():
    """Same text in & out -> distance ~ 0, passes the gate, carries all fields."""
    remember(topic="t", summary="python pytest fixture scope session")
    results = recall("python pytest fixture scope session", top_k=3, update=False)
    assert len(results) == 1
    r = results[0]
    assert r["distance"] < 0.01          # identical text
    assert set(["sim", "score", "strength_now", "topic", "summary"]).issubset(r.keys())


def test_gate_filters_noise():
    """A fully unrelated query is blocked by the distance gate."""
    remember(topic="t", summary="python pytest fixture scope session")
    results = recall("zzz completely unrelated random xyzzy noise text 99999",
                     top_k=3, update=False)
    # Different hash vector => distance well above 0.9 => gated out.
    assert len(results) == 0


def test_dedup_merges_identical_summary():
    """Storing an identical summary again merges into the existing record."""
    rid1, a1 = remember(topic="t", summary="sqlite vec0 knn vector search")
    rid2, a2 = remember(topic="t", summary="sqlite vec0 knn vector search")
    assert a1 == "created"
    assert a2 == "merged"
    assert rid2 == rid1
    # Still only one row in the DB.
    conn = db.get_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_strength_fresh_is_near_one():
    now_ts = datetime.now().isoformat(timespec="seconds")
    s = _strength_now(1.0, 7.0, now_ts, now_ts, now_ts)
    assert 0.99 <= s <= 1.0


def test_strength_decays_over_time():
    old_ts = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    s = _strength_now(1.0, 7.0, old_ts, old_ts, old_ts)
    # exp(-30/7) ~ 0.014
    assert s < 0.05


def test_recall_update_reinforces_memory():
    """Recall with update=True should reset strength and grow tau."""
    old_ts = (datetime.now() - timedelta(days=20)).isoformat(timespec="seconds")
    remember(topic="t", summary="reinforce me please", ts=old_ts)

    conn = db.get_conn()
    try:
        before = conn.execute(
            "SELECT strength, tau FROM episodic WHERE summary = ?",
            ("reinforce me please",),
        ).fetchone()
    finally:
        conn.close()
    assert before["strength"] == 1.0  # stored default

    recall("reinforce me please", top_k=1, update=True)
    conn = db.get_conn()
    try:
        after = conn.execute(
            "SELECT strength, tau, last_recall_ts FROM episodic WHERE summary = ?",
            ("reinforce me please",),
        ).fetchone()
    finally:
        conn.close()
    assert after["strength"] == 1.0
    assert after["tau"] > before["tau"]            # tau grew (1.5x)
    assert after["last_recall_ts"] is not None     # stamp recorded


def test_forget_purges_weak_memories():
    old_ts = (datetime.now() - timedelta(days=100)).isoformat(timespec="seconds")
    remember(topic="ancient", summary="a very old memory to be purged", ts=old_ts)
    r = decay_all(threshold=0.5, purge=True, min_age_days=0)
    assert r["purged"] >= 1
    # And the row is gone.
    conn = db.get_conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_forget_keeps_fresh_memories():
    remember(topic="fresh", summary="just created memory")
    r = decay_all(threshold=0.5, purge=True, min_age_days=0)
    assert r["purged"] == 0
    assert r["total"] == 1


def test_embed_dim_consistency():
    """Stored vector dim must equal the configured embed dim."""
    remember(topic="t", summary="dimension check memory")
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT embedding FROM episodic_vec LIMIT 1").fetchone()
    finally:
        conn.close()
    assert len(row[0]) // 4 == embed.dim()


def test_hybrid_is_default():
    """Hybrid (vector + BM25) recall is on by default."""
    assert config.hybrid_enable() is True


def test_hybrid_bm25_rescues_keyword_match():
    """BM25 branch recalls a memory a strict vector gate would drop.

    Query shares the keyword 'python' with the stored summary but is otherwise
    different -> different hash vector -> large distance -> gated out in
    pure-vector mode. The BM25 branch still matches on the shared keyword, so
    hybrid recall rescues it. This is the core value of hybrid retrieval.
    """
    remember(topic="t", summary="python pytest fixture scope session")
    strict = 0.1
    r_vec = recall("python testing unrelated zzz", top_k=3,
                   hybrid=False, threshold=strict)
    assert len(r_vec) == 0                      # pure-vector: gated out
    r_hyb = recall("python testing unrelated zzz", top_k=3,
                   hybrid=True, threshold=strict)
    assert len(r_hyb) >= 1                      # hybrid: BM25 rescued
    assert r_hyb[0]["summary"] == "python pytest fixture scope session"
    assert "rrf" in r_hyb[0]


def test_hybrid_rrf_scores_are_normalized():
    """RRF relevance is normalized to [0, 1] and present on every result."""
    remember(topic="t1", summary="python testing basics")
    remember(topic="t2", summary="python advanced patterns")
    results = recall("python", top_k=5, hybrid=True)
    assert len(results) >= 1
    for r in results:
        assert "rrf" in r
        assert 0.0 <= r["rrf"] <= 1.0
        assert r["score"] >= 0.0


def test_vector_only_mode_has_no_rrf():
    """hybrid=False keeps the legacy pure-vector behavior (no rrf field)."""
    remember(topic="t", summary="python pytest fixture scope session")
    results = recall("python pytest fixture scope session", top_k=3, hybrid=False)
    assert len(results) == 1
    assert "rrf" not in results[0]
    assert "sim" in results[0]
