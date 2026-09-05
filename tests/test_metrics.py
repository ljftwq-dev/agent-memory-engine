"""Metrics tests (issue #2): /metrics counters + snapshot assembly."""
from engine import server


def _remember(topic, summary):
    from engine.remember import remember
    return remember(topic=topic, summary=summary)


class TestCounters:
    def test_bump_and_set(self):
        server._STATS.update(recalls_served=0, recall_results=0, remembers=0,
                             remember_merges=0, forgets=0, last_forget_purged=None)
        server._bump(recalls_served=1, recall_results=3)
        server._bump(recalls_served=1, recall_results=2)
        server._bump(remembers=1, remember_merges=1)
        server._bump(forgets=1)
        server._set_stat("last_forget_purged", 7)
        assert server._STATS["recalls_served"] == 2
        assert server._STATS["recall_results"] == 5
        assert server._STATS["remember_merges"] == 1
        assert server._STATS["last_forget_purged"] == 7

    def test_bump_ignores_unknown_keys(self):
        before = dict(server._STATS)
        server._bump(not_a_counter=5)
        assert server._STATS == before


class TestSnapshot:
    def test_snapshot_shape_and_counts(self):
        server._STATS.update(recalls_served=4, recall_results=10, remembers=2,
                             remember_merges=1, forgets=1, last_forget_purged=3)
        _remember("m1", "first memory for metrics")
        _remember("m2", "second memory for metrics")
        snap = server._metrics_snapshot()
        assert snap["ok"] is True
        assert snap["memories_total"] >= 2          # live COUNT(*) from SQLite
        assert snap["recalls_served"] == 4
        assert snap["avg_results_per_recall"] == 2.5  # 10 / 4
        assert snap["remembers"] == 2
        assert snap["remember_merges"] == 1
        assert snap["last_forget_purged"] == 3
        assert "embed_mode" in snap and "embed_dim" in snap

    def test_avg_none_when_no_recalls(self):
        server._STATS.update(recalls_served=0, recall_results=0)
        snap = server._metrics_snapshot()
        assert snap["avg_results_per_recall"] is None
