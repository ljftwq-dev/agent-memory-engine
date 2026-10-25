"""social.py - social memory layer: per-fan cards + continuous mood state.

Extends the episodic core with structured social memory, aimed at persona
agents that interact with an audience (e.g. a Bilibili creator account):

- ``social_card`` table: one row per (platform, uid) fan card — relation
  level, favor score (0-100), impressions, blacklist flag. Inspired by the
  favor systems of open-source audience bots, but layered on this engine's
  retrieval/decay stack instead of a plain vector store.
- ``mood_state`` table: singleton "current mood" row, plus a ``mood_event``
  log. Gives an agent emotional continuity across sessions: save on exit,
  restore on wake, replay recent events when context allows.

All functions accept an explicit ``conn`` (dependency injection), so tests
can point them at a throwaway database. Every write path acquires
``db.WRITE_LOCK`` for the same cross-thread safety reasons as db.py.
"""
from datetime import datetime

from . import db

RELATIONS = ("stranger", "fan", "regular", "friend", "master", "blocked")


def relation_from_favor(favor: int) -> str:
    """Map a 0-100 favor score to the default relation tier.

    master / blocked are manual overrides and are never auto-assigned here.
    """
    if favor >= 90:
        return "friend"
    if favor >= 70:
        return "regular"
    if favor >= 40:
        return "fan"
    return "stranger"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_social(conn=None):
    """Create the social tables if they do not exist yet."""
    conn = conn or db.get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS social_card (
            platform    TEXT NOT NULL,
            uid         TEXT NOT NULL,
            nickname    TEXT DEFAULT '',
            relation    TEXT DEFAULT 'stranger',
            favor       INTEGER DEFAULT 50,
            impressions TEXT DEFAULT '',
            traits      TEXT DEFAULT '',
            first_seen  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (platform, uid)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_state (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            label      TEXT NOT NULL,
            valence    INTEGER DEFAULT 0,
            arousal    INTEGER DEFAULT 0,
            reason     TEXT DEFAULT '',
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_event (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT NOT NULL,
            label   TEXT NOT NULL,
            valence INTEGER NOT NULL,
            arousal INTEGER NOT NULL,
            reason  TEXT DEFAULT ''
        )
    """)
    conn.commit()


def _card_row(conn, platform: str, uid: str):
    return conn.execute(
        "SELECT * FROM social_card WHERE platform=? AND uid=?",
        (platform, uid)).fetchone()


def upsert_fan(platform: str, uid: str, nickname=None, favor=None,
               favor_delta=None, impression=None, trait=None,
               relation=None, blacklist=None, conn=None) -> dict:
    """Create or merge-update one fan card and return the stored card.

    ``favor_delta`` is applied on top of the stored (or default) score and
    clamped to [0, 100]; ``favor`` (absolute) wins if both are given. The
    relation tier auto-refreshes from the final favor score unless an
    explicit ``relation`` (or ``blacklist=True``) is passed.
    """
    conn = conn or db.get_conn()
    now = _now()
    with db.WRITE_LOCK:
        row = _card_row(conn, platform, uid)
        if row is None:
            new_favor = 50 if favor is None else favor
            if favor_delta is not None and favor is None:
                new_favor = 50 + favor_delta
            new_favor = max(0, min(100, int(new_favor)))
            new_relation = relation if relation else relation_from_favor(new_favor)
            if blacklist:
                new_relation = "blocked"
            conn.execute(
                """INSERT INTO social_card
                   (platform, uid, nickname, relation, favor, impressions,
                    traits, first_seen, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (platform, uid, nickname or "", new_relation, new_favor,
                 impression or "", trait or "", now, now))
        else:
            cur = dict(row)
            if favor is not None:
                new_favor = int(favor)
            elif favor_delta is not None:
                new_favor = int(cur["favor"]) + favor_delta
            else:
                new_favor = int(cur["favor"])
            new_favor = max(0, min(100, new_favor))
            if relation:
                new_relation = relation
            elif blacklist is True:
                new_relation = "blocked"
            elif cur["relation"] == "blocked" and blacklist is not False:
                new_relation = "blocked"
            else:
                new_relation = relation_from_favor(new_favor)
            impressions = cur["impressions"]
            if impression:
                lines = [l for l in impressions.splitlines() if l.strip()]
                lines.append(impression.strip())
                impressions = "\n".join(lines[-20:])
            traits = cur["traits"]
            if trait:
                lines = [l for l in traits.splitlines() if l.strip()]
                lines.append(trait.strip())
                traits = "\n".join(lines[-20:])
            conn.execute(
                """UPDATE social_card SET nickname=?, relation=?, favor=?,
                   impressions=?, traits=?, updated_at=? WHERE platform=? AND uid=?""",
                (nickname if nickname is not None else cur["nickname"],
                 new_relation, new_favor, impressions, traits, now,
                 platform, uid))
        conn.commit()
        return dict(_card_row(conn, platform, uid))


def get_fan(platform: str, uid: str, conn=None):
    """Return the fan card as a dict, or None if unknown."""
    conn = conn or db.get_conn()
    row = _card_row(conn, platform, uid)
    return dict(row) if row else None


def list_fans(platform=None, limit=100, conn=None):
    """List fan cards, most-favored first. Optional platform filter."""
    conn = conn or db.get_conn()
    if platform:
        rows = conn.execute(
            "SELECT * FROM social_card WHERE platform=? "
            "ORDER BY favor DESC LIMIT ?", (platform, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM social_card ORDER BY favor DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def bump_favor(platform: str, uid: str, delta: int, reason="", conn=None) -> dict:
    """Shift a fan's favor by ``delta`` (clamped 0-100), logging the reason
    as an impression line. Creates the card if it does not exist yet."""
    imp = f"favor {delta:+d}: {reason}" if reason else f"favor {delta:+d}"
    return upsert_fan(platform, uid, favor_delta=delta, impression=imp,
                      conn=conn)


def set_mood(label: str, valence=0, arousal=0, reason="", conn=None) -> dict:
    """Overwrite the singleton current mood and append a mood event."""
    conn = conn or db.get_conn()
    now = _now()
    with db.WRITE_LOCK:
        conn.execute(
            """INSERT INTO mood_state (id, label, valence, arousal, reason, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET label=excluded.label,
               valence=excluded.valence, arousal=excluded.arousal,
               reason=excluded.reason, updated_at=excluded.updated_at""",
            (label, int(valence), int(arousal), reason, now))
        conn.execute(
            "INSERT INTO mood_event (ts, label, valence, arousal, reason) "
            "VALUES (?,?,?,?,?)",
            (now, label, int(valence), int(arousal), reason))
        conn.commit()
        return get_mood(conn=conn)


def get_mood(conn=None):
    """Return the current mood dict, or None if never set."""
    conn = conn or db.get_conn()
    row = conn.execute("SELECT * FROM mood_state WHERE id=1").fetchone()
    return dict(row) if row else None


def recent_mood_events(limit=10, conn=None):
    """Return the most recent mood events (newest first)."""
    conn = conn or db.get_conn()
    rows = conn.execute(
        "SELECT * FROM mood_event ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
