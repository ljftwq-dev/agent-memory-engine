"""remember.py - store an episodic memory.

Key features:
- Optional LLM summary: calls an OpenAI-compatible endpoint to condense the
  dialogue into a semantic sentence before embedding -> much better retrieval.
  Disabled automatically if AME_LLM_BASE_URL / AME_LLM_API_KEY are unset.
- Auto dedup: if the new summary's nearest neighbor distance <= threshold,
  merge into the existing record (refresh ts, reset strength, tau *= 1.5,
  append raw, replace vector) instead of creating a duplicate.

CLI:
  python -m engine.remember -t "topic" -s "summary"
  echo "long text" | python -m engine.remember -t "topic" -s "summary" --summarize
"""
import argparse
import json
import sqlite3
import sys
import urllib.request
from datetime import datetime

import sqlite_vec

from . import config, db, embed


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


def remember(topic, summary, raw=None, ts=None,
             dedup=True, dedup_threshold=None, summarize=False):
    """Store one episodic memory. Returns ``(rowid, action)`` where
    ``action`` is ``"created"`` or ``"merged"``.

    With ``summarize=True`` (and LLM configured): the original ``summary`` is
    condensed into a semantic sentence used for embedding/storage; the original
    is appended to ``raw``. LLM failure falls back to the original text without
    blocking the store.
    """
    if dedup_threshold is None:
        dedup_threshold = config.dedup_threshold()
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")

    if summarize and config.llm_enabled():
        summary_new = _llm_summarize(summary)
        if summary_new:
            raw = summary if not raw else (raw + "\n\n---\n" + summary)
            summary = summary_new

    conn = db.get_conn()
    try:
        db.init_db(dim=embed.dim())
        vec = embed.encode(summary)
        blob = sqlite_vec.serialize_float32(vec)

        if dedup:
            try:
                row = conn.execute(
                    "SELECT v.rowid AS rowid, v.distance AS distance, e.raw AS raw "
                    "FROM episodic_vec v "
                    "JOIN episodic e ON e.rowid = v.rowid "
                    "WHERE v.embedding MATCH ? AND v.k = 1 "
                    "ORDER BY v.distance",
                    (blob,),
                ).fetchone()
                if row is not None and row["distance"] <= dedup_threshold:
                    rid = row["rowid"]
                    old_raw = row["raw"]
                    if old_raw and raw:
                        new_raw = old_raw + "\n\n---\n" + raw
                    else:
                        new_raw = old_raw or raw
                    conn.execute(
                        "UPDATE episodic SET ts = ?, topic = ?, summary = ?, "
                        "raw = ?, strength = 1.0, tau = tau * 1.5 WHERE rowid = ?",
                        (ts, topic, summary, new_raw, rid),
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
            "INSERT INTO episodic (ts, topic, summary, raw, created_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, topic, summary, raw, ts),
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
    args = parser.parse_args()

    raw = args.raw
    if not raw and not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            raw = stdin_data

    rowid, action = remember(
        args.topic, args.summary, raw=raw, ts=args.ts,
        dedup=not args.no_dedup, dedup_threshold=args.dedup_threshold,
        summarize=args.summarize,
    )
    tag = "MERGED into" if action == "merged" else "CREATED"
    print(f"OK {tag} episode #{rowid}")
    print(f"   topic:   {args.topic}")
    print(f"   summary: {args.summary}")


if __name__ == "__main__":
    main()
