"""metrics.py - IR ranking metrics for the benchmark.

- nDCG@k: normalized discounted cumulative gain, handles graded relevance.
- Recall@k: fraction of relevant items retrieved in the top-k.

Both take a ranked list of item ids (the system's output, best first) and a
relevance map {item_id: grade} (grade >= 1 counts as relevant).
"""
import math


def dcg_at_k(grades_in_rank_order, k):
    """DCG given the relevance grades of items already in rank order.

    Uses the 2^rel - 1 gain (standard for graded relevance).
    """
    total = 0.0
    for i, g in enumerate(grades_in_rank_order[:k], start=1):
        total += (2 ** g - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_ids, relevance, k):
    """nDCG@k for one query.

    ranked_ids: item ids in the system's predicted order (best first).
    relevance:  {item_id: grade} for THIS query (only graded items present).
    """
    ideal_grades = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal_grades, k)
    if idcg == 0:
        return 0.0
    gains = [relevance.get(i, 0) for i in ranked_ids[:k]]
    return dcg_at_k(gains, k) / idcg


def recall_at_k(ranked_ids, relevance, k):
    """Recall@k: |relevant in top-k| / |all relevant|. Relevant = grade >= 1."""
    relevant = {iid for iid, g in relevance.items() if g >= 1}
    if not relevant:
        return 0.0
    hits = len(relevant & set(ranked_ids[:k]))
    return hits / len(relevant)


def average(metric_fn, per_query_results, k):
    """Mean of metric_fn over all queries at cutoff k."""
    vals = []
    for ranked_ids, relevance in per_query_results:
        vals.append(metric_fn(ranked_ids, relevance, k))
    return sum(vals) / len(vals) if vals else 0.0
