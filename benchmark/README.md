# Retrieval Benchmark

Quantifies how much each retrieval stage is worth, on a hand-labeled evaluation
set. Replaces "we claim hybrid + reranker is better" with actual nDCG / Recall
numbers.

## What it measures

Ranking **quality of the relevance signal**, with the Ebbinghaus decay
neutralized so the numbers reflect pure retrieval:

- every memory seeded with the same fresh timestamp -> `strength_now ~= 1`
- `alpha=0` -> score is the relevance signal alone
- `min_strength=0`, `threshold=2.0` -> nothing filtered by decay or the gate

So this answers: *given a fixed candidate pool, how good is each relevance
signal at ranking the right memories first?* It does **not** measure the decay
mechanism (that's a separate concern).

## Dataset

`dataset.py`: 40 coding-agent-style memories across 5 domains (Python, Git/Tools,
ML/DL, Quant/Finance, Math) + 24 queries with **graded relevance** (2 = highly
relevant, 1 = partial). The last 6 queries are deliberately colloquial /
indirect ("stop my docker image from being bloated") - realistic user phrasings
with low lexical overlap, which is where a strong bi-encoder starts to slip and
a cross-encoder should pull ahead.

## Run

```bash
pip install -e ".[reranker]"        # needs real embeddings + the cross-encoder
python benchmark/run_benchmark.py
python benchmark/run_benchmark.py --verbose    # per-query breakdown
```

Not run in CI - the hash-fallback embedder has no semantics, so a benchmark on
it would be meaningless. This is a manual, model-dependent analysis tool.

## Results (2026-07-26, BGE-m3 + bge-reranker-v2-m3)

```
Condition                   nDCG@5   Recall@5
--------------------------------------------
pure-vector                  0.842      0.875
hybrid (vector+BM25)         0.868      0.896
hybrid + reranker            0.927      0.979
```

Each stage earns its keep:

| step | nDCG@5 gain | Recall@5 gain |
|---|---|---|
| +hybrid (RRF fusion) | +0.026 | +0.021 |
| +reranker (cross-encoder) | +0.059 | +0.083 |

The cross-encoder is the biggest single win - reading `(query, candidate)`
jointly beats encoding them separately, exactly as IR theory predicts.

## A bug the benchmark caught (and fixed)

The first run had the reranker *losing* on Recall (0.771). Root cause: the
reranker was re-scoring only the **top-20** candidates with min-max
normalization, so an item normalized to 0.0 sank below the un-reranked items
(which kept their RRF score) - wrongly demoting borderline-relevant memories.
Switching to **rerank all gated candidates** (`RERANKER_POOL=0`, the new
default) removed the boundary artifact and Recall jumped 0.771 -> 0.979.

That's the point of a benchmark: it doesn't just confirm what you hoped, it
surfaces what's actually broken.

## Extending it

- **More / harder queries** are the highest-leverage improvement - adversarial
  lexical traps and more colloquial phrasings widen the reranker's lead.
- **External baselines**: adapting this harness to score Mem0 / a plain
  Chroma cosine retriever on the same dataset would make the comparison
  apples-to-apples.
- **Larger memories** stress the gate and the pool sizing.

`MEMORIES` and `QUERIES` in `dataset.py` are plain Python lists - append and
re-run. Metrics are in `metrics.py` (nDCG, Recall, plus an `average` helper).
