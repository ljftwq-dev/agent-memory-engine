"""embed.py - shared embedding provider.

Design:
- Singleton model (loaded once on first call, then reused -> low memory).
- Default: BGE-m3 (``BAAI/bge-m3``)
    * multilingual (incl. Chinese), 1024-dim
    * first call downloads to ~/.cache/modelscope or ~/.cache/huggingface
    * local & free, runnable on CPU
- Hash-based fallback: if the model fails to load, a deterministic
  pseudo-random vector is produced from the text hash. The same text always
  yields the same vector (reproducible), so the pipeline still runs end-to-end
  in dev/test environments without the real model. It has NO semantic power.
"""
import hashlib
import os
import threading

import numpy as np

from . import config

_MODEL = None
_MODE = None
_LOAD_LOCK = threading.Lock()


def _local_modelscope_path(model_name):
    """Resolve a model to its local ModelScope snapshot dir without network.

    Layout: {MODELSCOPE_CACHE}/models/{ns}--{name}/snapshots/{rev}/
    Returns the revision dir path if it has config.json, else None.
    """
    cache = os.environ.get("MODELSCOPE_CACHE")
    if not cache:
        return None
    ns_name = model_name.replace("/", "--")
    snap_root = os.path.join(cache, "models", ns_name, "snapshots")
    if not os.path.isdir(snap_root):
        return None
    try:
        revs = sorted(os.listdir(snap_root))
    except OSError:
        return None
    ordered = (["master"] if "master" in revs else []) + [r for r in revs if r != "master"]
    for rev in ordered:
        cand = os.path.join(snap_root, rev)
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return None


def _load_model():
    global _MODEL, _MODE
    if _MODE is not None:          # fast path: already attempted (no lock)
        return
    # Double-checked locking: under ThreadingHTTPServer two concurrent recalls
    # could both miss the fast path and load the model twice. The lock makes
    # the (expensive) load happen exactly once across threads.
    with _LOAD_LOCK:
        if _MODE is not None:      # re-check under the lock
            return
        model_name = config.embed_model()
        try:
            from sentence_transformers import SentenceTransformer
            path = _local_modelscope_path(model_name)
            if path:
                print(f"[embed] local cache hit: {path}")
            else:
                try:
                    from modelscope import snapshot_download
                    path = snapshot_download(model_name)
                except Exception:
                    path = model_name
            _MODEL = SentenceTransformer(path)
            _MODE = "sentence-transformers"
            print(f"[embed] loaded {model_name} (mode={_MODE})")
        except Exception as e:
            _MODEL = None
            _MODE = "hash-fallback"
            print(f"[embed] WARNING: model load failed: {e}")
            print(f"[embed] falling back to hash pseudo-embedding (no semantics)")


def encode(text):
    """Encode a string into a vector (default 1024-dim), np.float32."""
    _load_model()
    if _MODE == "sentence-transformers":
        vec = _MODEL.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    return _hash_encode(text)


def encode_batch(texts):
    """Batch encode (faster). Returns (N, dim) np.float32."""
    _load_model()
    if _MODE == "sentence-transformers":
        vecs = _MODEL.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
    return np.stack([_hash_encode(t) for t in texts])


def _hash_encode(text, dim=None):
    """Deterministic pseudo-random vector: same text -> same vector, no semantics.

    For fallback use only.
    """
    dim = dim or config.embed_dim()
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")   # 32-bit seed for np RandomState
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def dim():
    """Actual vector dim: read from the loaded model, else the configured default."""
    if _MODEL is not None:
        try:
            return _MODEL.get_sentence_embedding_dimension()
        except Exception:
            pass
    return config.embed_dim()


def mode():
    _load_model()
    return _MODE


if __name__ == "__main__":
    print(f"mode: {mode()}")
    print(f"dim: {dim()}")
    v1 = encode("how to optimize a momentum factor")
    v2 = encode("how to optimize a momentum factor")
    v3 = encode("the weather is nice today")
    print(f"v1 shape: {v1.shape}, dtype: {v1.dtype}")
    print(f"same text same vec: {np.allclose(v1, v2)}")
    print(f"cos(v1, v3) = {float(v1 @ v3):.4f}  (near 0 = unrelated)")
