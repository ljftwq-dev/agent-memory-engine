"""db.py - SQLite + sqlite-vec storage layer.

One .db file holds both structured metadata and the vector index
(no separate service to run).

- ``episodic`` table: metadata (topic / summary / raw / strength / tau / ...)
- ``episodic_vec`` virtual table: vec0 index (FLOAT[dim]), joined to episodic by rowid
- ``episodic_fts`` virtual table: FTS5 full-text index over (topic, summary),
  used by the BM25 branch of hybrid recall. Auto-created if FTS5 is available.
- ``session`` table: registry of active agent sessions for multi-agent
  collaboration (each opencode/agent registers its current task so siblings
  can see what the others are doing).
"""
import os
import sqlite3
from datetime import datetime, timedelta

import sqlite_vec

from . import config


def get_conn(db_path=None):
    """Open a connection, auto-loading the sqlite-vec extension."""
    path = db_path or config.db_path()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(dim=None, force=False):
    """Initialize schema. ``force=True`` drops existing tables (dev reset).

    Returns the dimension in use.
    """
    dim = dim or config.embed_dim()
    conn = get_conn()
    try:
        if force:
            conn.execute("DROP TABLE IF EXISTS episodic")
            conn.execute("DROP TABLE IF EXISTS episodic_vec")
            conn.execute("DROP TABLE IF EXISTS episodic_fts")
            conn.execute("DROP TABLE IF EXISTS session")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw TEXT,
                strength REAL DEFAULT 1.0,
                tau REAL DEFAULT 7.0,
                last_recall_ts TEXT,
                created_ts TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vec USING vec0(
                embedding FLOAT[{dim}]
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_ts ON episodic(ts)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session (
                session_id  TEXT PRIMARY KEY,
                task        TEXT NOT NULL,
                progress    TEXT,
                status      TEXT DEFAULT 'active',
                started_ts  TEXT NOT NULL,
                updated_ts  TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_updated ON session(updated_ts)"
        )
        if fts5_available():
            init_fts(conn)
        conn.commit()
        return dim
    finally:
        conn.close()


def get_dim(db_path=None):
    """Read vector dim from an existing DB (to align the embed module).

    Returns the configured default if the DB or vec table is empty/missing.
    """
    path = db_path or config.db_path()
    if not os.path.exists(path):
        return config.embed_dim()
    conn = get_conn(path)
    try:
        row = conn.execute("SELECT embedding FROM episodic_vec LIMIT 1").fetchone()
        if row is None:
            return config.embed_dim()
        return len(row[0]) // 4  # float32 = 4 bytes
    except sqlite3.Error:
        return config.embed_dim()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FTS5 full-text index (for the BM25 branch of hybrid recall)
# ---------------------------------------------------------------------------

_FTS5_AVAILABLE = None


def fts5_available():
    """Cache whether the underlying SQLite engine supports FTS5."""
    global _FTS5_AVAILABLE
    if _FTS5_AVAILABLE is None:
        try:
            probe = sqlite3.connect(":memory:")
            probe.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
            probe.close()
            _FTS5_AVAILABLE = True
        except sqlite3.Error:
            _FTS5_AVAILABLE = False
    return _FTS5_AVAILABLE


def init_fts(conn):
    """Create the FTS5 external-content index over episodic + sync triggers.

    Idempotent. No-op if FTS5 is unavailable. Safe to call on every init_db.
    """
    if not fts5_available():
        return
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
            topic, summary,
            content='episodic', content_rowid='rowid',
            tokenize='unicode61'
        )
    """)
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS episodic_fts_ai AFTER INSERT ON episodic BEGIN
            INSERT INTO episodic_fts(rowid, topic, summary)
            VALUES (new.rowid, new.topic, new.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS episodic_fts_ad AFTER DELETE ON episodic BEGIN
            INSERT INTO episodic_fts(episodic_fts, rowid, topic, summary)
            VALUES('delete', old.rowid, old.topic, old.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS episodic_fts_au AFTER UPDATE ON episodic BEGIN
            INSERT INTO episodic_fts(episodic_fts, rowid, topic, summary)
            VALUES('delete', old.rowid, old.topic, old.summary);
            INSERT INTO episodic_fts(rowid, topic, summary)
            VALUES (new.rowid, new.topic, new.summary);
        END;
    """)
    n = conn.execute("SELECT COUNT(*) FROM episodic_fts").fetchone()[0]
    if n == 0:
        conn.execute(
            "INSERT INTO episodic_fts(rowid, topic, summary) "
            "SELECT rowid, topic, summary FROM episodic"
        )
    conn.commit()


def fts5_search(conn, query, k):
    """BM25 recall via FTS5. Returns ``[(rowid, bm25_score), ...]``.

    ``bm25_score`` is FTS5's ``bm25()`` (negative; smaller = more relevant).
    Empty list on no match or error. Tokenization mirrors unicode61
    (ASCII words + one CJK char per token) so MATCH works on Chinese.
    """
    if not fts5_available():
        return []
    toks = _fts_tokenize(query)
    if not toks:
        return []
    fts_query = " OR ".join(toks)  # OR => wide recall, RRF + strength refine later
    try:
        rows = conn.execute(
            "SELECT rowid, bm25(episodic_fts) AS score "
            "FROM episodic_fts WHERE episodic_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (fts_query, k),
        ).fetchall()
        return [(r["rowid"], r["score"]) for r in rows]
    except sqlite3.Error:
        return []


def _fts_tokenize(text):
    """ASCII words + one CJK char per token (matches unicode61 CJK behavior)."""
    import re
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9]+", text)
    toks += re.findall(r"[\u4e00-\u9fff]", text)
    return toks


# ---------------------------------------------------------------------------
# Session registry (multi-agent collaboration)
# ---------------------------------------------------------------------------
# Each running agent (an opencode window, a claude-code session, ...) registers
# here so its siblings can see "who is doing what, how far along". This is the
# shared "working state" layer on top of the shared "episodic memory" layer.

def register_session(session_id, task):
    """Register a session, or re-register with a new task (re-activates it)."""
    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    try:
        conn.execute(
            "INSERT INTO session (session_id, task, progress, status, "
            "started_ts, updated_ts) VALUES (?, ?, NULL, 'active', ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "task=excluded.task, status='active', updated_ts=excluded.updated_ts",
            (session_id, task, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def heartbeat_session(session_id, progress=None, task=None):
    """Refresh a session's liveness; optionally update task/progress.

    Returns False if the session_id is unknown (not registered).
    """
    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    try:
        cur = conn.execute(
            "UPDATE session SET updated_ts = ?, status = 'active' "
            "WHERE session_id = ?", (now, session_id),
        )
        if cur.rowcount == 0:
            return False
        if task is not None:
            conn.execute("UPDATE session SET task = ? WHERE session_id = ?",
                         (task, session_id))
        if progress is not None:
            conn.execute("UPDATE session SET progress = ? WHERE session_id = ?",
                         (progress, session_id))
        conn.commit()
        return True
    finally:
        conn.close()


def finish_session(session_id, result=None):
    """Mark a session finished; optionally stash a result summary in progress.

    Returns False if the session_id is unknown.
    """
    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    try:
        cur = conn.execute(
            "UPDATE session SET status = 'finished', updated_ts = ? "
            "WHERE session_id = ?", (now, session_id),
        )
        if cur.rowcount == 0:
            return False
        if result is not None:
            conn.execute("UPDATE session SET progress = ? WHERE session_id = ?",
                         (result, session_id))
        conn.commit()
        return True
    finally:
        conn.close()


def list_active_sessions(timeout_hours=None):
    """All active sessions with a heartbeat newer than ``now - timeout``.

    Sessions silent longer than the timeout are treated as stale and excluded.
    Defaults to ``config.SESSION_TIMEOUT_HOURS``.
    """
    if timeout_hours is None:
        timeout_hours = config.session_timeout_hours()
    cutoff = (datetime.now() - timedelta(hours=timeout_hours)).isoformat(timespec="seconds")
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT session_id, task, progress, status, started_ts, updated_ts "
            "FROM session WHERE status = 'active' AND updated_ts >= ? "
            "ORDER BY updated_ts DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    d = init_db(force=force)
    print(f"DB ready at: {config.db_path()}")
    print(f"vector dim: {d}")
    print(f"fts5: {'available' if fts5_available() else 'unavailable (BM25 fallback = pure Python)'}")
