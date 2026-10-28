"""P3b consolidation tests: clustering, rank-up-only retyping, degraded
(distinct-topic separation on the hash embedder via no-dedup inserts)."""
from engine import db
from engine.consolidate import (_apply_kind_upgrades, cluster, run)
from engine.engram import list_top
from engine.remember import remember

TOPIC_A = "cluster topic alpha shared"
# The hash test embedder has NO semantics: distinct texts are near-orthogonal
# (L2 ~1.4), so only IDENTICAL text (distance 0) clusters under the 0.8
# threshold. Real bge-m3 chains same-topic paraphrases; tests pin the
# mechanics with identical-text triplets instead.
SUM_A = "alpha cluster identical summary text for clustering mechanics"
SUM_B = SUM_A
SUM_C = SUM_A


def _seed():
    # no-dedup so near-identical texts stay separate rows (hash vectors:
    # identical text = distance 0, shared-tail text = very close)
    remember(TOPIC_A, SUM_A, dedup=False)
    remember(TOPIC_A, SUM_B, dedup=False)
    remember(TOPIC_A, SUM_C, dedup=False)
    remember("totally unrelated zebra topic",
             "quantum needlepoint embroidery hoop stash", dedup=False)


def test_cluster_groups_related_and_separates():
    from engine.consolidate import _load_window
    _seed()
    items = _load_window(days=7)
    clusters = cluster(items)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3]                    # the 3 alpha rows together, zebra alone


def test_dry_run_writes_nothing():
    _seed()
    stats = run(days=7, dry_run=True)
    assert stats["clusters"] >= 1
    assert list_top(n=50) == []               # no engram rows created
    conn = db.get_conn()
    n, = conn.execute("SELECT COUNT(*) FROM episodic WHERE kind != 'episode'").fetchone()
    conn.close()
    assert n == 0                             # no kind backfill


def test_run_degraded_path_creates_engrams():
    # no LLM configured in tests -> degraded naming (strongest member's topic),
    # but the cluster still lands in the directory
    _seed()
    stats = run(days=7, dry_run=False)
    assert stats["engrams"] >= 1
    top = list_top(n=10)
    assert any(e["member_count"] == 3 for e in top)


def test_kind_upgrades_rank_up_only():
    rid_ep, _ = remember("upgrade test", "upgrade test summary", kind="episode")
    rid_err, _ = remember("upgrade test 2", "upgrade test summary 2", kind="error")
    applied = _apply_kind_upgrades({rid_ep: "error", rid_err: "episode"})
    assert applied == 1                       # episode->error applied
    conn = db.get_conn()
    k1, = conn.execute("SELECT kind FROM episodic WHERE rowid=?", (rid_ep,)).fetchone()
    k2, = conn.execute("SELECT kind FROM episodic WHERE rowid=?", (rid_err,)).fetchone()
    conn.close()
    assert k1 == "error"                      # upgraded
    assert k2 == "error"                      # NOT demoted
