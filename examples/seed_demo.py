"""seed_demo.py - load demo data and verify the closed loop.

Generic, non-personal examples (so the repo ships clean sample data).
Timestamps are generated relative to *now* (last 5 days), so Ebbinghaus decay
doesn't accidentally hide them when someone clones and tries the demo fresh.

Run from the repo root:
    python examples/seed_demo.py                       # insert demo memories
    python -m engine.recall "vector search"            # then query
"""
import os
import sys
from datetime import datetime, timedelta

# Allow running as a script: add repo root to sys.path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import config, db, embed          # noqa: E402
from engine.remember import remember          # noqa: E402

# Each entry is N days ago; resolved to a real ISO ts at seed() time.
_SEED = [
    {"days_ago": 4, "hour": 10, "topic": "pytest fixtures",
     "summary": (
         "Discussed pytest fixtures: function scope recreates per test, "
         "session scope shares across the whole run. Put shared fixtures in "
         "conftest.py. Fixtures keep tests isolated and DRY.")},
    {"days_ago": 3, "hour": 14, "topic": "SQLite vector search",
     "summary": (
         "Explored sqlite-vec for vector search inside SQLite. A vec0 virtual "
         "table stores FLOAT[dim] embeddings; KNN via 'WHERE embedding MATCH ? "
         "AND k = N'. Zero extra service to run.")},
    {"days_ago": 2, "hour": 9, "topic": "Transformer self-attention",
     "summary": (
         "Self-attention computes Q, K, V from inputs; attention weights = "
         "softmax(QK^T / sqrt(d_k)); output = weights * V. Every position "
         "attends to all others, avoiding RNN's long-path problem.")},
    {"days_ago": 1, "hour": 16, "topic": "Ebbinghaus forgetting curve",
     "summary": (
         "Memory retention decays as exp(-t/tau). Spaced repetition slows "
         "decay by growing tau on each recall - the use-it-or-lose-it "
         "principle behind long-term memory systems.")},
    {"days_ago": 0, "hour": 11, "topic": "Git interactive rebase",
     "summary": (
         "git rebase -i HEAD~3 squashes, rewords, or reorders the last 3 "
         "commits. Never rebase commits already pushed to a shared branch. "
         "Use git reflog to recover from mistakes.")},
]


def _ts(days_ago, hour):
    now = datetime.now()
    return (now - timedelta(days=days_ago)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    ).isoformat(timespec="minutes")


def seed(force=False):
    if force and os.path.exists(config.db_path()):
        os.remove(config.db_path())
        print(f"[seed] removed old {config.db_path()}")
    db.init_db(dim=embed.dim(), force=False)
    print(f"[seed] embed mode = {embed.mode()}, dim = {embed.dim()}")
    print(f"[seed] inserting {len(_SEED)} episodes (ts = last {len(_SEED)} days) ...")
    for item in _SEED:
        ts = _ts(item["days_ago"], item["hour"])
        rowid, action = remember(topic=item["topic"], summary=item["summary"], ts=ts)
        print(f"  #{rowid} [{action}]  {ts[:10]}  {item['topic']}")
    print(f"[seed] done. db at {config.db_path()}")


if __name__ == "__main__":
    seed(force="--force" in sys.argv)
