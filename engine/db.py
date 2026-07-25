"""db.py - SQLite + sqlite-vec storage layer.

One .db file holds both structured metadata and the vector index
(no separate service to run).

- ``episodic`` table: metadata (topic / summary / raw / strength / tau / ...)
- ``episodic_vec`` virtual table: vec0 index (FLOAT[dim]), joined to episodic by rowid
"""
import os
import sqlite3

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


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    d = init_db(force=force)
    print(f"DB ready at: {config.db_path()}")
    print(f"vector dim: {d}")
