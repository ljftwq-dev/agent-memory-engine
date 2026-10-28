"""P1 typology tests: kind column, per-kind tau, purge protection, merge
promotion.

Lessons (error) used to decay like ordinary episodes and get purged, so they
were exiled to hand-maintained notes outside the engine. These tests pin the
typology that lets lessons live IN the engine.
"""
import pytest

from engine import db, embed
from engine.forget import PROTECTED_KINDS, decay_all
from engine.recall import recall
from engine.remember import KINDS, KIND_DEFAULT_TAU, remember


def _get_row(rowid):
    conn = db.get_conn()
    try:
        return conn.execute(
            "SELECT kind, tau, strength FROM episodic WHERE rowid = ?",
            (rowid,)).fetchone()
    finally:
        conn.close()


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        remember("t", "invalid kind test", kind="vibes")


def test_default_kind_is_episode():
    rid, _ = remember("t", "default kind test")
    assert _get_row(rid)["kind"] == "episode"


def test_each_kind_gets_its_tau():
    for kind in KINDS:
        rid, _ = remember("t", f"tau check {kind}", kind=kind)
        row = _get_row(rid)
        assert row["kind"] == kind
        assert row["tau"] == KIND_DEFAULT_TAU[kind]


def test_purge_protects_error_and_capability():
    # decay_all recomputes strength from timestamps (it ignores the stored
    # strength), so we age the rows by rewriting their clocks to 2020 with a
    # tiny tau: new_strength = exp(-dt/tau) ~ 0 for BOTH rows.
    OLD = "2020-01-01T00:00:00"
    rid_ep, _ = remember("t", "purge race episode", kind="episode")
    for kind in PROTECTED_KINDS:
        rid, _ = remember("t", f"purge race {kind}", kind=kind)
        conn = db.get_conn()
        with db.WRITE_LOCK:
            for target in (rid, rid_ep):
                conn.execute(
                    "UPDATE episodic SET tau = 0.01, ts = ?, created_ts = ?, "
                    "last_recall_ts = ? WHERE rowid = ?",
                    (OLD, OLD, OLD, target))
        conn.commit()
        conn.close()
        stats = decay_all(threshold=0.5, purge=True)
        assert stats["purged"] >= 1                  # the episode died
        assert _get_row(rid) is not None             # protected survived
        assert _get_row(rid)["strength"] < 0.5       # ...but still decayed
        # recreate the sacrificial episode for the next loop iteration
        rid_ep, _ = remember("t", "purge race episode")


def test_merge_promotes_kind_and_tau_floor():
    rid, _ = remember("merge promo", "identical merge promo summary", kind="episode")
    rid2, action = remember("merge promo", "identical merge promo summary",
                            kind="error")
    assert action == "merged" and rid2 == rid
    row = _get_row(rid)
    assert row["kind"] == "error"
    assert row["tau"] >= KIND_DEFAULT_TAU["error"]   # upgrade applies immediately


def test_merge_keeps_stronger_existing_kind():
    rid, _ = remember("merge keep", "identical merge keep summary", kind="error")
    rid2, action = remember("merge keep", "identical merge keep summary",
                            kind="episode")
    assert action == "merged" and rid2 == rid
    assert _get_row(rid)["kind"] == "error"          # no demotion


def test_migration_adds_kind_column():
    conn = db.get_conn()
    try:
        cols_before = {r["name"] for r in conn.execute("PRAGMA table_info(episodic)")}
    finally:
        conn.close()
    if "kind" in cols_before:
        pytest.skip("kind column already present (fresh schema)")
    db.init_db(dim=embed.dim())
    conn = db.get_conn()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(episodic)")}
    finally:
        conn.close()
    assert "kind" in cols


def test_recall_returns_kind():
    remember("kind echo", "kind echo summary", kind="fact")
    results = recall("kind echo summary", top_k=3)
    assert results
    assert results[0]["kind"] == "fact"
