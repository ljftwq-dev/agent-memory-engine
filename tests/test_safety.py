"""Tests for write-lock serialization and DB backup safety.

Run: pytest -q tests/test_safety.py
"""
import os
import threading
import pytest

from engine import config, db
from engine.remember import remember


def test_concurrent_identical_remembers_merge_into_one():
    """N threads storing the SAME summary must end with 1 row, not N.

    Without WRITE_LOCK the dedup-read-then-write races: every thread reads
    'no near neighbor', every thread inserts -> N duplicates (and racing vec0
    writes can corrupt the DB, which is exactly what cost us a production DB).
    The lock serializes the writes so dedup actually fires."""
    summary = "concurrent dedup race test summary unique token zzz999"
    errors = []

    def store():
        try:
            remember(topic="t", summary=summary)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=store) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    conn = db.get_conn()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM episodic WHERE summary=?", (summary,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1   # 8 concurrent stores -> 1 created + 7 merged


def test_backup_produces_valid_copy():
    """do_backup() writes a consistent snapshot that can be opened and read."""
    remember(topic="t", summary="a memory to back up safely")
    dst = db.do_backup()
    assert dst and os.path.exists(dst)

    import sqlite3
    bak = sqlite3.connect(dst)
    try:
        n = bak.execute(
            "SELECT COUNT(*) FROM episodic WHERE summary='a memory to back up safely'"
        ).fetchone()[0]
    finally:
        bak.close()
    assert n == 1


def test_rotate_keeps_only_n_backups():
    """rotate_backups keeps only the newest BACKUP_KEEP snapshots."""
    import time as _t
    bdir = db.backups_dir()
    if os.path.isdir(bdir):
        for f in os.scandir(bdir):
            if f.name.endswith(".db"):
                os.remove(f.path)
    keep = config.backup_keep()
    for _ in range(keep + 3):
        db.do_backup()
        _t.sleep(0.1)   # distinct mtimes
    db.rotate_backups()
    files = [f for f in os.listdir(bdir) if f.endswith(".db")]
    assert len(files) <= keep
