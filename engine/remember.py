"""remember.py - store an episodic memory.

Key features:
- Optional LLM summary: calls an OpenAI-compatible endpoint to condense the
  dialogue into a semantic sentence before embedding -> much better retrieval.
  Disabled automatically if AME_LLM_BASE_URL / AME_LLM_API_KEY are unset.
- Auto dedup: if the new summary's nearest neighbor distance <= threshold,
  merge into the existing record (refresh ts, reset strength, tau *= 1.5,
  append raw, replace vector) instead of creating a duplicate.
- session_id: tags the memory with the agent session that wrote it, so
  multi-agent collaboration can tell memories apart (anti cross-talk).

CLI:
  python -m engine.remember -t "topic" -s "summary"
  echo "long text" | python -m engine.remember -t "topic" -s "summary" --summarize --session-id oc-abc123
"""
import argparse
import json
import sqlite3
import sys
import urllib.request
from datetime import datetime

import sqlite_vec

from . import config, db, embed


_SCHEMA_READY = False


def _ensure_schema():
    """Make sure the schema exists. init_db is idempotent but opens its own
    connection + re-checks triggers every call, so we gate it on a flag and do
    it only once per process (the server already inits at startup; this covers
    the CLI / first-call path). A `force=True` reset elsewhere (e.g. tests)
    recreates the schema regardless."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    db.init_db(dim=embed.dim())
    _SCHEMA_READY = True


def _llm_summarize(text):
    """Condense the dialogue into one semantic sentence via an OpenAI-compatible API.

    Returns None on any failure (graceful fallback to the original text).
    """
    base_url = config.llm_base_url()
    api_key = config.llm_api_key()
    model = config.llm_model()
    if not base_url or not api_key:
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a memory summarizer. Condense the user-assistant dialogue "
                "into ONE semantic sentence covering the topic and key conclusion. "
                "Rules: keep core keywords for retrieval; <=80 chars; no structural "
                "words like 'user said'; output only the summary."
            )},
            {"role": "user", "content": (text or "")[:2000]},
        ],
        "temperature": 0.3,
        "max_tokens": 120,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[remember] LLM summarize failed: {e!r}", file=sys.stderr)
        return None


# P0 size guards: the dedup-merge path appends raw text without bound
# (old_raw + new_raw, forever), so a hot memory merged hundreds of times
# grows its raw into a token bomb; and long summaries ride along in every
# injected recall result. Both are capped at write time.
MAX_RAW_CHARS = 8192
MAX_SUMMARY_CHARS = 500


# P1 memory typology: what kind of thing a memory IS determines how it
# forgets. Lessons learned from 4 months in production: error lessons were
# being decayed and purged like ordinary episodes, so they had to live in a
# hand-maintained external file instead of the engine. Types fix that:
#   episode    - ordinary episodic context (default; classic Ebbinghaus)
#   fact       - stable facts, slower decay
#   strategy   - operational strategies, may evolve over time
#   capability - things the agent learned to DO; shouldn't quietly vanish
#   error      - lessons/pitfalls; slowest decay, never purged
KINDS = ("episode", "fact", "strategy", "capability", "error")
KIND_DEFAULT_TAU = {
    "episode": 7.0,        # same as the historical table default
    "fact": 30.0,
    "strategy": 90.0,
    "capability": 180.0,
    "error": 365.0,        # ~a year to decay to 0.37; recall resets & boosts tau
}
# Merge precedence when a new memory lands on an existing row: keep the
# stronger (more protected) kind. An episode merging into an error row must
# not demote the lesson back to purgable.
KIND_RANK = {k: i for i, k in enumerate(KINDS)}


def _cap_raw(text, limit=MAX_RAW_CHARS):
    """Cap stored raw text at ``limit`` chars, keeping the newest tail.

    FIFO drop of the oldest content: merged raws accumulate chronologically,
    and recent dialogue matters more than ancient history. The one-line
    marker keeps the truncation visible instead of silently rewriting the
    past. Final length stays <= limit + len(marker).
    """
    if text is None or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"[...{dropped} chars truncated, oldest dropped...]\n{text[dropped:]}"


def remember(topic, summary, raw=None, ts=None,
             dedup=True, dedup_threshold=None, summarize=False, session_id=None,
             kind="episode"):
    """Store one episodic memory. Returns ``(rowid, action)`` where
    ``action`` is ``"created"`` or ``"merged"``.

    With ``summarize=True`` (and LLM configured): the original ``summary`` is
    condensed into a semantic sentence used for embedding/storage; the original
    is appended to ``raw``. LLM failure falls back to the original text without
    blocking the store.

    ``session_id`` tags which agent session wrote this memory (anti cross-talk
    in multi-agent setups). Left None for backward compatibility.

    ``kind`` types the memory (see KINDS); it sets the initial tau and, on
    merge, upgrades to the stronger kind. Invalid values raise ValueError —
    a lesson stored under the wrong type loses its purge protection, so
    silently falling back to "episode" would be the dangerous choice.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if dedup_threshold is None:
        dedup_threshold = config.dedup_threshold()
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")

    if summarize and config.llm_enabled():
        summary_new = _llm_summarize(summary)
        if summary_new:
            raw = summary if not raw else (raw + "\n\n---\n" + summary)
            summary = summary_new

    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS]
    raw = _cap_raw(raw)

    conn = db.get_conn()
    try:
        with db.WRITE_LOCK:
            _ensure_schema()
            vec = embed.encode(summary)
            blob = sqlite_vec.serialize_float32(vec)

            if dedup:
                try:
                    row = conn.execute(
                        "SELECT v.rowid AS rowid, v.distance AS distance, "
                        "e.raw AS raw, e.kind AS kind "
                        "FROM episodic_vec v "
                        "JOIN episodic e ON e.rowid = v.rowid "
                        "WHERE v.embedding MATCH ? AND v.k = 1 "
                        "ORDER BY v.distance",
                        (blob,),
                    ).fetchone()
                    if row is not None and row["distance"] <= dedup_threshold:
                        rid = row["rowid"]
                        old_raw = row["raw"]
                        old_kind = row["kind"] or "episode"
                        # merge keeps the stronger kind; tau floor is the new
                        # kind's default so an upgrade protects immediately
                        merged_kind = (old_kind if KIND_RANK[old_kind] >= KIND_RANK[kind]
                                       else kind)
                        if old_raw and raw:
                            new_raw = old_raw + "\n\n---\n" + raw
                        else:
                            new_raw = old_raw or raw
                        # cap AFTER concatenation: the merge path is the one
                        # that grows unboundedly across calls
                        new_raw = _cap_raw(new_raw)
                        conn.execute(
                            "UPDATE episodic SET ts = ?, topic = ?, summary = ?, "
                            "raw = ?, strength = 1.0, kind = ?, "
                            "tau = MAX(COALESCE(tau, 7.0) * 1.5, ?), session_id = ? "
                            "WHERE rowid = ?",
                            (ts, topic, summary, new_raw, merged_kind,
                             KIND_DEFAULT_TAU[merged_kind], session_id, rid),
                        )
                        conn.execute("DELETE FROM episodic_vec WHERE rowid = ?", (rid,))
                        conn.execute(
                            "INSERT INTO episodic_vec (rowid, embedding) VALUES (?, ?)",
                            (rid, blob),
                        )
                        conn.commit()
                        return rid, "merged"
                except sqlite3.Error:
                    pass  # dedup query failed -> fall through to normal insert

            cur = conn.execute(
                "INSERT INTO episodic (ts, topic, summary, raw, created_ts, "
                "session_id, kind, tau) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, topic, summary, raw, ts, session_id, kind,
                 KIND_DEFAULT_TAU[kind]),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO episodic_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
            conn.commit()
            return rowid, "created"
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Store an episodic memory")
    parser.add_argument("-t", "--topic", required=True, help="one-line topic")
    parser.add_argument("-s", "--summary", required=True, help="summary (retrieval key)")
    parser.add_argument("--raw", help="full original text (optional)")
    parser.add_argument("--ts", help="custom ISO timestamp (optional)")
    parser.add_argument("--no-dedup", action="store_true", help="force create, skip dedup")
    parser.add_argument("--dedup-threshold", type=float)
    parser.add_argument("--summarize", action="store_true",
                        help="call LLM to summarize before storing")
    parser.add_argument("--session-id", help="which agent session wrote this (anti cross-talk)")
    parser.add_argument("--kind", default="episode", choices=KINDS,
                        help="memory type: sets initial tau and purge protection "
                             "(error/capability are never purged)")
    args = parser.parse_args()

    raw = args.raw
    if not raw and not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            raw = stdin_data

    rowid, action = remember(
        args.topic, args.summary, raw=raw, ts=args.ts,
        dedup=not args.no_dedup, dedup_threshold=args.dedup_threshold,
        summarize=args.summarize, session_id=args.session_id, kind=args.kind,
    )
    tag = "MERGED into" if action == "merged" else "CREATED"
    print(f"OK {tag} episode #{rowid}")
    print(f"   topic:   {args.topic}")
    print(f"   summary: {args.summary}")


if __name__ == "__main__":
    main()
