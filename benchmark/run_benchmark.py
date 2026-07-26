"""run_benchmark.py - retrieval-quality ablation for the memory engine.

Compares three retrieval configurations on a hand-labeled evaluation set:

  A. pure-vector      hybrid=False               (sim = 1 - distance)
  B. hybrid           hybrid=True                (vector + BM25, fused by RRF)
  C. hybrid + reranker hybrid=True, rerank=True   (cross-encoder precision rerank)

Decay (Ebbinghaus strength) is NEUTRALIZED so the numbers reflect pure
retrieval/reranking quality:
  - every memory is seeded with the same fresh timestamp -> strength_now ~= 1
  - alpha=0  -> score is the relevance signal alone
  - min_strength=0, threshold=2.0 -> nothing filtered out by decay or the gate

So this answers: "given a fixed candidate pool, how good is each relevance
signal at ranking the right memories first?" It does NOT measure the decay
mechanism (that's a separate benchmark).

Requires real embeddings (BGE-m3) and, for condition C, the reranker model.
Not run in CI - the hash fallback has no semantics, so it would be meaningless.

Run from the repo root:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --verbose     # per-query breakdown
    python benchmark/run_benchmark.py --skip-reranker  # skip the heavy condition
"""
import argparse
import os
import sys
import tempfile

# Pin a throwaway DB BEFORE importing the engine, so we never touch the user's
# real memory. config reads AME_DB_PATH lazily.
_TMP = tempfile.mkdtemp(prefix="ame_bench_")
os.environ["AME_DB_PATH"] = os.path.join(_TMP, "bench.db")
# Make sure we use the real models, not the nonexistent test override.
os.environ.pop("AME_RERANKER_MODEL", None)
os.environ.pop("AME_EMBED_MODEL", None)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import db, embed                       # noqa: E402
from engine.remember import remember               # noqa: E402
from engine.recall import recall                   # noqa: E402

from benchmark.dataset import MEMORIES, QUERIES    # noqa: E402
from benchmark import metrics                       # noqa: E402

# Isolate retrieval quality: loose gate, no decay filtering, relevance-only score.
RECALL_KW = dict(top_k=10, threshold=2.0, min_strength=0.0, alpha=0.0,
                 recall_pool=40, bm25_pool=40, update=False)
EVAL_K = 5


def seed():
    """Seed all memories with a uniform fresh timestamp (strength ~= 1)."""
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    db.init_db(dim=embed.dim(), force=True)
    for topic, summary in MEMORIES:
        remember(topic=topic, summary=summary, ts=now, dedup=False)


def run_condition(label, rerank, do_hybrid, skip=False):
    """Run recall over all queries under one config; return per-query (ranked, rel)."""
    if skip:
        return None
    per_query = []
    for query, relevance in QUERIES:
        results = recall(query, hybrid=do_hybrid, rerank=rerank, **RECALL_KW)
        ranked = [r["rowid"] for r in results]
        per_query.append((ranked, relevance))
    return per_query


def main():
    ap = argparse.ArgumentParser(description="Retrieval-quality benchmark")
    ap.add_argument("--verbose", action="store_true", help="per-query breakdown")
    ap.add_argument("--skip-reranker", action="store_true",
                    help="skip the cross-encoder condition (heavy model)")
    args = ap.parse_args()

    print("=" * 64)
    print("  Agent Memory Engine - Retrieval Benchmark")
    print("=" * 64)
    print(f"  dataset    : {len(MEMORIES)} memories, {len(QUERIES)} queries")
    print(f"  eval @ {EVAL_K}    : nDCG@{EVAL_K}, Recall@{EVAL_K}")
    print(f"  settings   : threshold=2.0 alpha=0 min_strength=0 (decay neutralized)")
    print(f"  embed      : {embed.mode()} ({embed.dim()}d)")
    print("=" * 64)
    print("  seeding ...")
    seed()

    conditions = [
        ("pure-vector",           False, False, False),
        ("hybrid (vector+BM25)",  False, True,  False),
        ("hybrid + reranker",     True,  True,  args.skip_reranker),
    ]
    rows = []
    for label, rerank, do_hybrid, skip in conditions:
        print(f"  running: {label} ...")
        per_query = run_condition(label, rerank, do_hybrid, skip=skip)
        if per_query is None:
            rows.append((label, None, None))
            continue
        ndcg = metrics.average(metrics.ndcg_at_k, per_query, EVAL_K)
        rec = metrics.average(metrics.recall_at_k, per_query, EVAL_K)
        rows.append((label, ndcg, rec))
        if args.verbose:
            print(f"    {'q#':>3}  {'nDCG@5':>7} {'Recall@5':>8}  query")
            for i, (ranked, rel) in enumerate(per_query, 1):
                q = QUERIES[i - 1][0]
                print(f"    {i:>3}  {metrics.ndcg_at_k(ranked, rel, EVAL_K):>7.3f} "
                      f"{metrics.recall_at_k(ranked, rel, EVAL_K):>8.3f}  {q[:48]}")

    print("\n" + "=" * 48)
    print(f"  {'Condition':<24}{'nDCG@5':>10}{'Recall@5':>11}")
    print("  " + "-" * 44)
    for label, ndcg, rec in rows:
        if ndcg is None:
            print(f"  {label:<24}{'skipped':>10}{'':>11}")
        else:
            print(f"  {label:<24}{ndcg:>10.3f}{rec:>11.3f}")
    print("=" * 48)


if __name__ == "__main__":
    main()
