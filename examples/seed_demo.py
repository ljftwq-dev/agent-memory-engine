"""seed_demo.py - load demo data and verify the closed loop.

Generic, non-personal examples (so the repo ships clean sample data).
Run from the repo root:

    python examples/seed_demo.py            # insert demo memories
    python -m engine.recall "python testing"   # then query
"""
import os
import sys

# Allow running as a script: add repo root to sys.path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import config, db, embed          # noqa: E402
from engine.remember import remember          # noqa: E402

SEED_DATA = [
    {
        "ts": "2026-06-01T10:00:00",
        "topic": "pytest fixtures",
        "summary": (
            "Discussed pytest fixtures: function scope recreates per test, "
            "session scope shares across the whole run. Put shared fixtures in "
            "conftest.py. Fixtures keep tests isolated and DRY."
        ),
    },
    {
        "ts": "2026-06-05T14:00:00",
        "topic": "SQLite vector search",
        "summary": (
            "Explored sqlite-vec for vector search inside SQLite. A vec0 virtual "
            "table stores FLOAT[dim] embeddings; KNN via 'WHERE embedding MATCH ? "
            "AND k = N'. Zero extra service to run."
        ),
    },
    {
        "ts": "2026-06-10T09:00:00",
        "topic": "Transformer self-attention",
        "summary": (
            "Self-attention computes Q, K, V from inputs; attention weights = "
            "softmax(QK^T / sqrt(d_k)); output = weights * V. Every position "
            "attends to all others, avoiding RNN's long-path problem."
        ),
    },
    {
        "ts": "2026-06-15T16:00:00",
        "topic": "Ebbinghaus forgetting curve",
        "summary": (
            "Memory retention decays as exp(-t/tau). Spaced repetition slows "
            "decay by growing tau on each recall - the use-it-or-lose-it "
            "principle behind long-term memory systems."
        ),
    },
    {
        "ts": "2026-06-20T11:00:00",
        "topic": "Git interactive rebase",
        "summary": (
            "git rebase -i HEAD~3 squashes, rewords, or reorders the last 3 "
            "commits. Never rebase commits already pushed to a shared branch. "
            "Use git reflog to recover from mistakes."
        ),
    },
]


def seed(force=False):
    if force and os.path.exists(config.db_path()):
        os.remove(config.db_path())
        print(f"[seed] removed old {config.db_path()}")
    db.init_db(dim=embed.dim(), force=False)
    print(f"[seed] embed mode = {embed.mode()}, dim = {embed.dim()}")
    print(f"[seed] inserting {len(SEED_DATA)} episodes ...")
    for item in SEED_DATA:
        rowid, action = remember(
            topic=item["topic"], summary=item["summary"], ts=item["ts"],
        )
        print(f"  #{rowid} [{action}]  {item['ts'][:10]}  {item['topic']}")
    print(f"[seed] done. db at {config.db_path()}")


if __name__ == "__main__":
    seed(force="--force" in sys.argv)
