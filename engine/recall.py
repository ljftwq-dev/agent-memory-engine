"""recall.py - hybrid (vector + BM25) retrieval with gating and Ebbinghaus rerank.

Pipeline (hybrid mode, the default):
  Stage A (wide recall):
    - vector branch: KNN pulls ``recall_pool`` candidates (with distance)
    - BM25 branch  : FTS5 (or pure-Python fallback) pulls ``bm25_pool``
                     candidates by keyword relevance
  Gating : drop vector candidates whose distance > threshold. BM25-only
           candidates bypass the distance gate (they're the whole point of
           hybrid recall - catch keyword matches the vector path missed).
  Rerank : Reciprocal Rank Fusion (RRF) merges the two rankings into one
           relevance score in [0, 1]; final ``score = α·strength + (1-α)·rel``.
           ``strength_now = exp(-dt/τ)`` is the real-time Ebbinghaus decay.

RRF replaces the old ``sim = 1 - distance`` as the relevance term. Pure-vector
mode (``hybrid=False``) keeps the original single-branch behavior.

CLI:
  python -m engine.recall "query"
  python -m engine.recall "vector search" -k 3 --json
  python -m engine.recall "keyword" --no-hybrid     # vector-only mode
"""
import argparse
import json
import math
import os
from datetime import datetime

import sqlite_vec

from . import config, db, embed

_RRF_K = 60  # standard RRF constant


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


def _bm25_recall(conn, query, k):
    """BM25 branch: prefer FTS5, fall back to pure-Python BM25."""
    if db.fts5_available():
        return db.fts5_search(conn, query, k)
    from . import bm25 as _bm25_mod
    return _bm25_mod.search(query, k)


def recall(query, top_k=3, threshold=None, update=False, min_strength=None,
           recall_pool=None, alpha=None, hybrid=None, bm25_pool=None):
    """Hybrid retrieval, returns up to ``top_k`` dicts.

    Each result additionally carries: ``strength_now``, ``score``, and either
    ``sim`` (pure-vector mode) or ``rrf`` (hybrid mode). Any of the tuning
    args left as None uses the configured default.
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
    if hybrid is None:
        hybrid = config.hybrid_enable()
    if bm25_pool is None:
        bm25_pool = config.bm25_pool()

    conn = db.get_conn()
    try:
        q_vec = embed.encode(query)
        blob = sqlite_vec.serialize_float32(q_vec)

        # ---- vector branch: KNN ----
        pool_v = max(recall_pool, top_k)
        v_rows = conn.execute(
            """
            SELECT e.rowid, e.ts, e.topic, e.summary, e.raw,
                   e.strength, e.tau, e.last_recall_ts, e.created_ts,
                   v.distance
            FROM episodic_vec v
            JOIN episodic e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND v.k = ?
            ORDER BY v.distance
            """,
            [blob, pool_v],
        ).fetchall()

        v_data = {}        # rowid -> full row dict (carries distance)
        v_ranked = []      # rowids in ascending-distance order
        for r in v_rows:
            d = dict(r)
            v_data[d["rowid"]] = d
            v_ranked.append(d["rowid"])

        # ---- BM25 branch (hybrid only) ----
        b_ranked = []      # rowids in BM25 relevance order
        if hybrid:
            bm25_hits = _bm25_recall(conn, query, bm25_pool)
            b_ranked = [rid for rid, _ in bm25_hits]
            # fetch full rows for BM25-only candidates not already in v_data
            missing = [rid for rid in b_ranked if rid not in v_data]
            if missing:
                placeholders = ",".join("?" * len(missing))
                extra = conn.execute(
                    f"SELECT rowid, ts, topic, summary, raw, strength, tau, "
                    f"last_recall_ts, created_ts FROM episodic "
                    f"WHERE rowid IN ({placeholders})",
                    missing,
                ).fetchall()
                for r in extra:
                    d = dict(r)
                    d["distance"] = None   # no vector distance for BM25-only
                    v_data[d["rowid"]] = d   # merge into the same map

        # ---- fuse + rerank ----
        now = datetime.now()
        v_rank = {rid: i for i, rid in enumerate(v_ranked)}
        b_rank = {rid: i for i, rid in enumerate(b_ranked)}
        # theoretical max RRF = 2 / (K+1) when ranked #1 in both branches
        rrf_norm = (2.0 / (_RRF_K + 1))

        candidates = []
        for rid in set(v_rank) | set(b_rank):
            d = v_data.get(rid)
            if d is None:
                continue

            # distance gate applies to vector candidates; in hybrid mode a
            # BM25 hit bypasses the gate (keyword match rescues it)
            if d.get("distance") is not None and d["distance"] > threshold:
                if not (hybrid and rid in b_rank):
                    continue

            s_now = _strength_now(
                d.get("strength"), d.get("tau"),
                d.get("last_recall_ts"), d.get("created_ts"),
                d.get("ts"), now=now,
            )
            if s_now < min_strength:
                continue
            d["strength_now"] = s_now

            if hybrid:
                rrf = 0.0
                if rid in v_rank:
                    rrf += 1.0 / (_RRF_K + v_rank[rid] + 1)
                if rid in b_rank:
                    rrf += 1.0 / (_RRF_K + b_rank[rid] + 1)
                rel = rrf / rrf_norm           # normalized to [0, 1]
                d["rrf"] = rel
            else:
                rel = 1.0 - d["distance"]      # pure-vector similarity
            d["sim"] = rel                     # unified relevance field

            d["score"] = alpha * s_now + (1.0 - alpha) * rel
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


def _format(results, fmt="text", hybrid=True):
    if fmt == "json":
        return json.dumps(results, ensure_ascii=False, indent=2)
    if not results:
        return "(no match)"
    lines = []
    for i, r in enumerate(results, 1):
        rel = r.get("rrf", r.get("sim", 0.0))
        rel_tag = "rrf" if (hybrid and "rrf" in r) else "sim"
        score = r.get("score", rel)
        ts_short = (r["ts"] or "")[:16].replace("T", " ")
        s_now = r.get("strength_now", 1.0)
        lines.append(
            f"[{i}] score={score:.3f}  {rel_tag}={rel:.3f}  str={s_now:.2f}  {ts_short}  #{r['rowid']}\n"
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
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=None,
                        help="enable/disable vector+BM25 hybrid recall (default: config)")
    parser.add_argument("--bm25-pool", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--update", action="store_true",
                        help="update last_recall/strength after recall (use-it-or-lose-it)")
    args = parser.parse_args()

    results = recall(
        args.query, top_k=args.top_k, threshold=args.threshold,
        update=args.update, min_strength=args.min_strength,
        recall_pool=args.recall_pool, alpha=args.alpha,
        hybrid=args.hybrid, bm25_pool=args.bm25_pool,
    )
    is_hybrid = args.hybrid if args.hybrid is not None else config.hybrid_enable()
    print(_format(results, "json" if args.json else "text", hybrid=is_hybrid))


if __name__ == "__main__":
    main()
