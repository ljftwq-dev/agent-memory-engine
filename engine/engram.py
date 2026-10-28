"""engram.py - the concept-level directory over episodic memories.

Recall answers "which fragments match this query"; it cannot answer "what do
we know about X, overall". The engram table is the missing layer: one row per
concept ("memory engine", "biligirl channel", "thesis"), each with a one-line
current-state summary and the rowids of the member memories. Retrieval then
becomes two-stage: find the concept in the directory, drill into members on
demand — an index over memory, not more memory.

Deliberate contrast with letters (append-only history): an engram is a LIVING
summary. Consolidation may rewrite its state_summary, merge concepts, or
raise/lower confidence. The past lives in letters + episodic rows; the
directory only describes the present state of what they add up to.

Writers:
  - manual upsert (this module, today): the agent registers/refreshes a
    concept at the end of a meaningful work session — already useful alone.
  - nightly consolidation (consolidate.py, later): clusters recent episodic
    memories and distills/refreshes concepts automatically.

Lazy-init like social/letters: untouched deployments keep a pristine DB.

CLI:
  python -m engine.engram upsert "concept" "state summary"
  python -m engine.engram lookup "natural language query"
  python -m engine.engram list --n 10
"""
import argparse
import json
import sqlite3
from datetime import datetime

import sqlite_vec

from . import db, embed

# Distance gate in sqlite-vec's L2 space (vectors are normalized, so
# L2 1.0 ~= cosine similarity 0.5). Measured on production bge-m3:
# correct question->concept pairs land 0.86-1.00, wrong concepts 1.06+.
# Question-vs-name matching is naturally farther than dedup-style matching;
# a cosine-intuition threshold (0.6) gates out every real hit.
_lookup_threshold = 1.05

_ready = False


def _ensure_ready():
    """Create engram + engram_vec once per process (idempotent)."""
    global _ready
    if _ready:
        return
    conn = db.get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engram (
                concept TEXT PRIMARY KEY,
                state_summary TEXT NOT NULL,
                member_rowids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.5,
                hits INTEGER NOT NULL DEFAULT 0,
                created_ts TEXT NOT NULL,
                updated_ts TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS engram_vec USING vec0(
                embedding FLOAT[{embed.dim()}]
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_engram_updated ON engram(updated_ts)")
        conn.commit()
        _ready = True
    finally:
        conn.close()


def upsert(concept, state_summary, member_rowids=None, confidence=None,
           confidence_floor=False):
    """Register or refresh a concept. Returns ``(concept, action)`` where
    action is "created" or "refreshed".

    member_rowids: optional iterable of episodic rowids belonging to this
    concept; on refresh the union with existing members is kept (members are
    facts about the past and are never dropped here).
    confidence: optional 0-1 override; on refresh defaults to nudging the
    stored value toward 1.0 (a human/agent touch is evidence the concept is
    real) — consolidation may lower it instead.
    confidence_floor: when True (the consolidation path), an explicit
    confidence never LOWERS the stored value — automated nightly runs must
    not dilute the trust a manual registration has earned.
    """
    concept = (concept or "").strip()
    state_summary = (state_summary or "").strip()
    if not concept or not state_summary:
        raise ValueError("concept and state_summary are both required")
    new_members = sorted(set(int(r) for r in (member_rowids or [])))

    # Embed ONLY the concept name: directory lookup is question -> concept
    # name matching; stuffing the (long, detail-diluted) state_summary into
    # the vector pushed real queries past the distance gate in production.
    vec = embed.encode(concept)
    blob = sqlite_vec.serialize_float32(vec)
    now = datetime.now().isoformat(timespec="seconds")

    _ensure_ready()
    conn = db.get_conn()
    try:
        with db.WRITE_LOCK:
            row = conn.execute(
                "SELECT member_rowids, confidence FROM engram WHERE concept = ?",
                (concept,)).fetchone()
            if row is None:
                conf = 0.5 if confidence is None else float(confidence)
                conn.execute(
                    "INSERT INTO engram (concept, state_summary, member_rowids, "
                    "confidence, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?, ?)",
                    (concept, state_summary, json.dumps(new_members), conf,
                     now, now))
                action = "created"
            else:
                old_members = set(json.loads(row["member_rowids"] or "[]"))
                merged = sorted(old_members | set(new_members))
                old_conf = float(row["confidence"])
                if confidence is None:
                    conf = old_conf + (1.0 - old_conf) * 0.2
                elif confidence_floor:
                    conf = max(old_conf, float(confidence))
                else:
                    conf = float(confidence)
                conn.execute(
                    "UPDATE engram SET state_summary = ?, member_rowids = ?, "
                    "confidence = ?, updated_ts = ? WHERE concept = ?",
                    (state_summary, json.dumps(merged), conf, now, concept))
                action = "refreshed"
            # vector always tracks the latest text (delete + insert, same
            # pattern as the remember merge path); two plain statements —
            # vec0's DELETE parser is best kept away from subqueries
            rowid = conn.execute(
                "SELECT rowid FROM engram WHERE concept = ?", (concept,)).fetchone()[0]
            conn.execute("DELETE FROM engram_vec WHERE rowid = ?", (rowid,))
            conn.execute("INSERT INTO engram_vec (rowid, embedding) VALUES (?, ?)",
                         (rowid, blob))
            conn.commit()
            return concept, action
    finally:
        conn.close()


def lookup(query, top_k=3):
    """Find directory concepts matching a natural-language query.

    Returns up to top_k entries (closest first): concept, state_summary,
    member count, confidence, updated_ts. Also bumps ``hits`` on matches —
    hot concepts earn their place in the cold-start bootstrap view later.
    """
    _ensure_ready()
    vec = embed.encode(query)
    blob = sqlite_vec.serialize_float32(vec)
    conn = db.get_conn()
    try:
        rows = conn.execute(
            """
            SELECT e.rowid, e.concept, e.state_summary, e.member_rowids,
                   e.confidence, e.updated_ts, v.distance
            FROM engram_vec v JOIN engram e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance
            """,
            (blob, max(top_k * 3, 6))).fetchall()
        out = []
        for r in rows:
            if r["distance"] > _lookup_threshold:
                continue
            members = json.loads(r["member_rowids"] or "[]")
            out.append({
                "concept": r["concept"],
                "state_summary": r["state_summary"],
                "member_count": len(members),
                "confidence": round(float(r["confidence"]), 3),
                "updated_ts": r["updated_ts"],
                "distance": round(float(r["distance"]), 3),
            })
            with db.WRITE_LOCK:
                try:
                    conn.execute("UPDATE engram SET hits = hits + 1 WHERE concept = ?",
                                 (r["concept"],))
                except sqlite3.OperationalError:
                    pass    # counter write lost to lock contention: the lookup
                            # result itself must never fail because of a hit counter
        if out:
            try:
                conn.commit()
            except sqlite3.OperationalError:
                pass
        return out[:top_k]
    finally:
        conn.close()


def list_top(n=10):
    """Directory overview for cold-start: most-confident first, recency as
    tiebreaker (engram rows carry no decay — the directory describes the
    present, not a fading trace)."""
    _ensure_ready()
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT concept, state_summary, member_rowids, confidence, hits, "
            "updated_ts FROM engram ORDER BY confidence DESC, updated_ts DESC "
            "LIMIT ?", (n,)).fetchall()
        return [{
            "concept": r["concept"],
            "state_summary": r["state_summary"],
            "member_count": len(json.loads(r["member_rowids"] or "[]")),
            "confidence": round(float(r["confidence"]), 3),
            "hits": r["hits"],
            "updated_ts": r["updated_ts"],
        } for r in rows]
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Concept-level engram directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upsert", help="register/refresh a concept")
    up.add_argument("concept")
    up.add_argument("state_summary")
    lk = sub.add_parser("lookup", help="find concepts by query")
    lk.add_argument("query")
    lk.add_argument("--k", type=int, default=3)
    ls = sub.add_parser("list", help="directory overview")
    ls.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "upsert":
        concept, action = upsert(args.concept, args.state_summary)
        print(f"{action} engram: {concept}")
    elif args.cmd == "lookup":
        for i, e in enumerate(lookup(args.query, top_k=args.k), 1):
            print(f"[{i}] {e['concept']} (conf={e['confidence']}, "
                  f"members={e['member_count']}, d={e['distance']})\n"
                  f"    {e['state_summary']}")
    else:
        for e in list_top(n=args.n):
            print(f"- {e['concept']} (conf={e['confidence']}, "
                  f"members={e['member_count']}, hits={e['hits']})\n"
                  f"    {e['state_summary']}")


if __name__ == "__main__":
    main()
