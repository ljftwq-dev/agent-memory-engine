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

import numpy as np

from . import config

_MODEL = None
_MODE = None


def _load_model():
    global _MODEL, _MODE
    if _MODE is not None:          # already attempted (success or fallback)
        return
    model_name = config.embed_model()
    try:
        from sentence_transformers import SentenceTransformer
        path = None
        # Prefer local ModelScope cache, then let sentence-transformers resolve HF.
        try:
            from modelscope import snapshot_download
            path = snapshot_download(model_name)
        except Exception:
            path = model_name  # falls back to HF cache / download inside ST
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
