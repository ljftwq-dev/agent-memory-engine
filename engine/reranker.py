"""reranker.py - optional cross-encoder precision re-ranker.

Sits between hybrid fusion (RRF) and final top-k selection in recall.py:

    wide recall (vector + BM25)
        |
    gate + RRF fusion  -> candidates (~dozens, each with an ``rrf`` score)
        |
    [THIS MODULE] cross-encoder re-scores (query, candidate) pairs
        |
    final score = alpha*strength + (1-alpha)*rerank_rel ; top-k

Design (mirrors embed.py):
- Singleton, lazy-loaded on first call, then reused (one model in memory).
- Default: BAAI/bge-reranker-v2-m3 (multilingual incl. Chinese, ~568M).
- First call downloads to the modelscope/HF cache (on this box that points at
  D:\\AIModels via HF_HOME / MODELSCOPE_CACHE).
- Graceful degradation: if disabled in config OR the model fails to load
  (heavy dep absent, CI offline, etc.), ``rerank()`` returns None and the
  caller keeps the RRF relevance untouched. The engine runs unchanged without
  it - the reranker is a pure precision upgrade, never a hard requirement.
- Normalization: cross-encoder raw scores are min-max normalized within the
  candidate batch to [0, 1], so they blend on the same scale as Ebbinghaus
  strength. This monotonic transform preserves the model's ranking exactly;
  only the blend with strength is affected.

CLI:
    python -m engine.reranker            # show load mode
    python -m engine.reranker "query"    # probe on a toy pair
"""
import argparse
import threading

from . import config

_MODEL = None
_MODE = None  # None = untried; "cross-encoder" = loaded; "disabled" = off/unavailable
_LOAD_LOCK = threading.Lock()


def _load_model():
    global _MODEL, _MODE
    if _MODE is not None:          # fast path: already attempted (no lock)
        return
    # NB: the enable/disable decision lives in recall.py (the ``rerank`` param,
    # default from config.reranker_enable()). This module just loads on demand
    # once asked, and degrades to None if the heavy dep is unavailable.
    # Double-checked locking: under ThreadingHTTPServer concurrent recalls must
    # not load the cross-encoder twice.
    with _LOAD_LOCK:
        if _MODE is not None:      # re-check under the lock
            return
        model_name = config.reranker_model()
        try:
            from sentence_transformers import CrossEncoder
            path = None
            # Prefer local ModelScope cache, then let sentence-transformers resolve HF.
            try:
                from modelscope import snapshot_download
                path = snapshot_download(model_name)
            except Exception:
                path = model_name  # falls back to HF cache / download inside ST
            _MODEL = CrossEncoder(path)
            _MODE = "cross-encoder"
            print(f"[reranker] loaded {model_name} (mode={_MODE})")
        except Exception as e:
            _MODEL = None
            _MODE = "disabled"
            print(f"[reranker] WARNING: model load failed: {e}")
            print(f"[reranker] disabled, keeping RRF relevance for recall")


def _candidate_text(d):
    """Text fed to the cross-encoder: prefer the full ``raw`` memory, then
    fall back to topic + summary so short records still get scored."""
    raw = d.get("raw")
    if raw:
        return raw
    parts = [d.get("topic", ""), d.get("summary", "")]
    return " ".join(p for p in parts if p)


def rerank(query, candidates):
    """Score (query, candidate) pairs with the cross-encoder.

    Returns a list of [0, 1] relevance scores aligned with ``candidates``
    (best in the batch -> 1.0, worst -> 0.0). Returns None if the reranker is
    disabled or unavailable, so the caller keeps the RRF relevance as-is.
    """
    if not candidates:
        return None
    _load_model()
    if _MODE != "cross-encoder":
        return None
    pairs = [(query, _candidate_text(c)) for c in candidates]
    raw = _MODEL.predict(pairs)
    lo = float(min(raw))
    hi = float(max(raw))
    span = hi - lo
    if span < 1e-9:
        # All identical scores: keep a neutral relevance (rank preserved trivially).
        return [0.5 for _ in candidates]
    return [(float(s) - lo) / span for s in raw]


def mode():
    _load_model()
    return _MODE


def cached_mode():
    """The cached mode WITHOUT triggering a load.

    Used by /health for observability: we don't want a health check to download
    a 1.1GB cross-encoder. Returns None until the first recall actually
    requests reranking.
    """
    return _MODE


def _main():
    parser = argparse.ArgumentParser(description="Probe the optional cross-encoder reranker")
    parser.add_argument("query", nargs="?", default=None, help="optional probe query")
    parser.add_argument("--doc", default="a candidate memory about python pandas",
                        help="candidate text to score against the query")
    args = parser.parse_args()
    print(f"mode: {mode()}")
    if args.query is None:
        return
    scores = rerank(args.query, [{"raw": args.doc}])
    if scores is None:
        print("(reranker unavailable, no score)")
    else:
        print(f"rerank({args.query!r}, {args.doc!r}) = {scores[0]:.4f}")


if __name__ == "__main__":
    _main()
