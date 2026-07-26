"""Tests for the additive schema migration (_migrate_episodic).

A DB created before ``session_id`` was introduced is missing the column, and
recall crashes with ``no such column: e.session_id``. init_db must add it via
ALTER TABLE so old DBs keep working.
"""
import sqlite3

import pytest
import sqlite_vec
from datetime import datetime

from engine import db, embed
from engine.recall import recall


def _episodic_cols():
    conn = db.get_conn()
    try:
        return {row["name"] for row in conn.execute("PRAGMA table_info(episodic)")}
    finally:
        conn.close()


def _make_old_schema_episodic(summary="python migration test summary"):
    """Recreate episodic in its pre-a6389f0 shape (no session_id) + one row
    with a vector so recall can find it."""
    conn = db.get_conn()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("DROP TABLE IF EXISTS episodic")
        conn.execute("""CREATE TABLE episodic (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, topic TEXT NOT NULL, summary TEXT NOT NULL,
            raw TEXT, strength REAL DEFAULT 1.0, tau REAL DEFAULT 7.0,
            last_recall_ts TEXT, created_ts TEXT NOT NULL
        )""")
        cur = conn.execute(
            "INSERT INTO episodic (ts, topic, summary, raw, created_ts) "
            "VALUES (?, 't', ?, ?, ?)",
            (now, summary, summary, now),
        )
        rid = cur.lastrowid
        blob = sqlite_vec.serialize_float32(embed.encode(summary))
        conn.execute(
            "INSERT INTO episodic_vec (rowid, embedding) VALUES (?, ?)",
            (rid, blob),
        )
        conn.commit()
    finally:
        conn.close()


def test_init_adds_missing_session_id():
    """An episodic table missing session_id gets it added on init_db."""
    _make_old_schema_episodic()
    assert "session_id" not in _episodic_cols()
    db.init_db(dim=embed.dim())
    assert "session_id" in _episodic_cols()
    # idempotent: running again is a no-op
    db.init_db(dim=embed.dim())
    assert "session_id" in _episodic_cols()


def test_recall_works_on_migrated_old_db():
    """The bug this fixes: recall crashed on a pre-session_id DB. After
    migration it returns results with session_id NULL on old rows."""
    summary = "python migration test summary"
    _make_old_schema_episodic(summary=summary)
    # before migration: recall hits `e.session_id` and crashes (the original bug)
    with pytest.raises(sqlite3.OperationalError):
        recall(summary, top_k=3)
    # migrate
    db.init_db(dim=embed.dim())
    # after migration: recall works; the old row has no session tag (NULL)
    results = recall(summary, top_k=3)
    assert len(results) == 1
    assert results[0]["summary"] == summary
    assert results[0]["session_id"] is None
