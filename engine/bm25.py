"""bm25.py - pure-Python BM25 fallback (used only when SQLite FTS5 is unavailable).

Tokenization is intentionally simple and dependency-free:
- ASCII runs -> lowercased word tokens
- CJK chars  -> one token per character (matches FTS5 unicode61's CJK behavior)

Standard BM25 (k1=1.5, b=0.75). Score is positive (larger = more relevant).
RRF in recall.py only uses rank, so the sign difference vs FTS5's negative bm25()
does not matter.
"""
import math
import re

from . import db


def _tokenize(text):
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens += re.findall(r"[\u4e00-\u9fff]", text)
    return tokens


def search(query, k):
    """BM25 recall over all memories. Returns [(rowid, score), ...] sorted desc.

    O(N * len(query)) per call - fine for hundreds-to-low-thousands of memories.
    """
    import os
    if not os.path.exists(db.config.db_path()):
        return []
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT rowid, topic, summary FROM episodic"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []

    docs = [(r["rowid"],
             _tokenize((r["topic"] or "") + " " + (r["summary"] or "")))
            for r in rows]
    N = len(docs)
    avgdl = sum(len(toks) for _, toks in docs) / max(N, 1)

    # document frequency per token
    df = {}
    for _, toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    q_toks = _tokenize(query)
    if not q_toks:
        return []

    k1, b = 1.5, 0.75
    scored = []
    for rowid, toks in docs:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        s = 0.0
        for qt in q_toks:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            n = df.get(qt, 0)
            if n == 0:
                continue
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
        if s > 0:
            scored.append((rowid, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
