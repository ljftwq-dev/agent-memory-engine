"""recall.py - two-stage retrieval + gating (the core selling point).

Pipeline:
  Stage A (wide recall): KNN pulls ``recall_pool`` candidates.
  Gating                : drop noise with distance > threshold.
  Stage B (rerank)      : score = alpha*strength_now + (1-alpha)*sim,
                          take top-k in descending order.

``strength_now = exp(-dt/tau)`` is a real-time Ebbinghaus decay - frequently
recalled memories stay strong (use-it-or-lose-it). This is our lightweight
substitute for MemRL's Q-value: it gives signal without needing reward data.

CLI:
  python -m engine.recall "query"
  python -m engine.recall "vector search" -k 3 --json
"""
import argparse
import json
import math
import os
from datetime import datetime

import sqlite_vec

from . import config, db, embed


def _strength_now(strength, tau, last_recall_ts, created_ts, ts, now=None):
    """Real-time Ebbinghaus strength ``exp(-dt/tau)``.

    ``dt`` is days since last recall (or creation).
    """
    now = now or datetime.now()

    def parse(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    ref = parse(last_recall_ts) or parse(created_ts) or parse(ts)
    if ref is None:
        return float(strength) if strength is not None else 0.0
    delta_days = (now - ref).total_seconds() / 86400.0
    tau_days = max(tau or 7.0, 0.001)
    return math.exp(-delta_days / tau_days)


def recall(query, top_k=3, threshold=None, update=False, min_strength=None,
           recall_pool=None, alpha=None):
    """Two-stage retrieval, returns up to ``top_k`` dicts.

    Each result additionally carries: ``sim``, ``strength_now``, ``score``.
    Any of threshold/min_strength/recall_pool/alpha left as None uses the
    configured default.
    """
    if not os.path.exists(config.db_path()):
        return []

    if threshold is None:
        threshold = config.recall_threshold()
    if min_strength is None:
        min_strength = config.min_strength()
    if recall_pool is None:
        recall_pool = config.recall_pool()
    if alpha is None:
        alpha = config.alpha()

    conn = db.get_conn()
    try:
        q_vec = embed.encode(query)
        blob = sqlite_vec.serialize_float32(q_vec)

        pool = max(recall_pool, top_k)
        sql = """
            SELECT e.rowid, e.ts, e.topic, e.summary, e.raw,
                   e.strength, e.tau, e.last_recall_ts, e.created_ts,
                   v.distance
            FROM episodic_vec v
            JOIN episodic e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND v.k = ?
            ORDER BY v.distance
        """
        rows = conn.execute(sql, [blob, pool]).fetchall()

        now = datetime.now()
        candidates = []
        for r in rows:
            d = dict(r)
            d["sim"] = 1.0 - d["distance"]
            d["strength_now"] = _strength_now(
                d.get("strength"), d.get("tau"),
                d.get("last_recall_ts"), d.get("created_ts"),
                d.get("ts"), now=now,
            )
            if d["strength_now"] < min_strength:
                continue
            if threshold is not None and d["distance"] > threshold:
                continue
            d["score"] = alpha * d["strength_now"] + (1.0 - alpha) * d["sim"]
            candidates.append(d)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:top_k]

        if update and results:
            now_iso = now.isoformat(timespec="seconds")
            for r in results:
                conn.execute(
                    "UPDATE episodic SET last_recall_ts = ?, "
                    "strength = 1.0, tau = tau * 1.5 WHERE rowid = ?",
                    (now_iso, r["rowid"]),
                )
            conn.commit()
        return results
    finally:
        conn.close()


def _format(results, fmt="text"):
    if fmt == "json":
        return json.dumps(results, ensure_ascii=False, indent=2)
    if not results:
        return "(no match)"
    lines = []
    for i, r in enumerate(results, 1):
        sim = r.get("sim", 1 - r["distance"])
        score = r.get("score", sim)
        ts_short = (r["ts"] or "")[:16].replace("T", " ")
        s_now = r.get("strength_now", 1.0)
        lines.append(
            f"[{i}] score={score:.3f}  sim={sim:.3f}  str={s_now:.2f}  {ts_short}  #{r['rowid']}\n"
            f"    topic:   {r['topic']}\n"
            f"    summary: {r['summary']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Recall top-k episodic memories")
    parser.add_argument("query", help="query text")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, help="distance gate (smaller = stricter)")
    parser.add_argument("--min-strength", type=float)
    parser.add_argument("--recall-pool", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--update", action="store_true",
                        help="update last_recall/strength after recall (use-it-or-lose-it)")
    args = parser.parse_args()

    results = recall(
        args.query, top_k=args.top_k, threshold=args.threshold,
        update=args.update, min_strength=args.min_strength,
        recall_pool=args.recall_pool, alpha=args.alpha,
    )
    print(_format(results, "json" if args.json else "text"))


if __name__ == "__main__":
    main()
