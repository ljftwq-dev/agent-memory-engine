"""Concurrency tests (issue #5): WAL + busy_timeout under mixed read/write load.

Writers serialize through db.WRITE_LOCK; readers run concurrently against the
same SQLite file. With WAL enabled, readers must never see "database is
locked", and every write must land exactly once.
"""
import threading

from engine import db
from engine.recall import recall
from engine.remember import remember

WRITER_THREADS = 4
WRITES_PER_THREAD = 25
READER_THREADS = 4
READS_PER_THREAD = 25


def test_journal_mode_is_wal():
    conn = db.get_conn()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_mixed_readers_writers_no_locked_errors():
    errors = []

    def writer(tag):
        try:
            for i in range(WRITES_PER_THREAD):
                remember(topic=f"stress {tag}-{i}",
                         summary=f"concurrency stress row {tag} {i}")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(("writer", repr(e)))

    def reader():
        try:
            for _ in range(READS_PER_THREAD):
                recall("concurrency stress", top_k=5, update=False)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(("reader", repr(e)))

    threads = [threading.Thread(target=writer, args=(f"t{n}",))
               for n in range(WRITER_THREADS)]
    threads += [threading.Thread(target=reader)
                for _ in range(READER_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    conn = db.get_conn()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM episodic WHERE topic LIKE 'stress %'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == WRITER_THREADS * WRITES_PER_THREAD
