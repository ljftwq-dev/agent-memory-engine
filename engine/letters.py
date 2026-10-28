"""letters.py - append-only mailbox: letters to the future self.

The episodic store keeps fragments (each self-contained, mergeable, purgable
- by design). What it cannot keep is the first-person narrative arc: how the
agent's judgment, strategies and relationships actually evolved. Letters are
that arc, one short note per day, written by today's self to tomorrow's.

Three rules, enforced structurally rather than by convention:
  1. Append-only. BEFORE UPDATE/DELETE triggers RAISE(ABORT) — the past
     physically cannot be rewritten or deleted, not even by a future admin
     session. (Regenerating one's past self is explicitly out of scope: it
     invites information overload and confusion.)
  2. Bounded. <= MAX_LETTER_CHARS (500). Oversized letters raise ValueError
     instead of being silently truncated — a letter cut mid-sentence loses
     its point; the writer should distill, not delegate the cutting.
  3. Not plans. A plan says "what should happen" and expires when reality
     disagrees; a letter records "what I thought and why" and never expires.
     The write tool's docstring steers callers accordingly.

Lazy-init like the social layer: deployments that never touch letters keep
a pristine DB. A missed day may be back-filled by passing ``day`` explicitly
— honest catch-up: ``written_ts`` always records the true write time.

CLI:
  python -m engine.letters write "letter body text"
  python -m engine.letters read            # latest 3, oldest-first
  python -m engine.letters read --n 5
"""
import argparse
import sqlite3
import sys
from datetime import datetime

from . import db

MAX_LETTER_CHARS = 500

_ready = False


def _ensure_ready():
    """Create the letters table + guards once per process (idempotent)."""
    global _ready
    if _ready:
        return
    conn = db.get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS letters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day TEXT NOT NULL,
                written_ts TEXT NOT NULL,
                body TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_letters_day ON letters(day)")
        # The single-direction guarantee: time flows one way. These fire on
        # ANY UPDATE/DELETE, from any code path, forever.
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS letters_no_update
            BEFORE UPDATE ON letters
            BEGIN
                SELECT RAISE(ABORT,
                    'letters are append-only: the past cannot be rewritten');
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS letters_no_delete
            BEFORE DELETE ON letters
            BEGIN
                SELECT RAISE(ABORT,
                    'letters are append-only: the past cannot be deleted');
            END
        """)
        conn.commit()
        _ready = True
    finally:
        conn.close()


def write_letter(body, day=None):
    """Append one letter. Returns ``(id, day)``.

    ``day`` defaults to today's local date (YYYY-MM-DD); pass it explicitly
    only for honest back-fill of a missed day — ``written_ts`` keeps the
    true write time either way. Raises ValueError over the length cap.
    """
    body = (body or "").strip()
    if not body:
        raise ValueError("letter body is empty")
    if len(body) > MAX_LETTER_CHARS:
        raise ValueError(
            f"letter is {len(body)} chars; max is {MAX_LETTER_CHARS} — "
            "distill it, don't get truncated")
    if day is None:
        day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().isoformat(timespec="seconds")

    _ensure_ready()
    conn = db.get_conn()
    try:
        with db.WRITE_LOCK:
            cur = conn.execute(
                "INSERT INTO letters (day, written_ts, body) VALUES (?, ?, ?)",
                (day, ts, body))
            conn.commit()
            return cur.lastrowid, day
    finally:
        conn.close()


def read_recent(n=3):
    """The latest ``n`` letters, returned OLDEST-first (narrative order:
    a cold-start reader should meet yesterday before today)."""
    _ensure_ready()
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT id, day, written_ts, body FROM letters "
            "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def read_by_day(day):
    """All letters written for one day (there may be several), oldest-first."""
    _ensure_ready()
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT id, day, written_ts, body FROM letters WHERE day = ? "
            "ORDER BY id", (day,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count():
    _ensure_ready()
    conn = db.get_conn()
    try:
        return conn.execute("SELECT COUNT(*) FROM letters").fetchone()[0]
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Append-only letters to the future self")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="append a letter (<=500 chars)")
    w.add_argument("body")
    w.add_argument("--day", help="explicit day YYYY-MM-DD (honest back-fill only)")
    r = sub.add_parser("read", help="read the latest letters, oldest-first")
    r.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "write":
        lid, day = write_letter(args.body, day=args.day)
        print(f"letter #{lid} sealed for day {day}")
    else:
        letters = read_recent(n=args.n)
        if not letters:
            print("(mailbox is empty)")
            return
        for l in letters:
            print(f"--- {l['day']} (written {l['written_ts']}) #{l['id']} ---")
            print(l["body"])


if __name__ == "__main__":
    main()
