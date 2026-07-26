"""pytest configuration: isolate tests from the user's real environment.

- Points the DB at a temp file (never touches ~/.agent-memory/memory.db).
- Forces the embed model to a nonexistent name so encode() falls back to the
  deterministic hash embedder. This keeps tests fast, offline, and reproducible
  in CI without downloading BGE-m3.
- Disables LLM summarization.
"""
import os
import sys
import tempfile

# Make the repo root importable so `from engine import ...` works.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TMP = tempfile.mkdtemp(prefix="ame_test_")
os.environ["AME_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["AME_EMBED_MODEL"] = "__nonexistent_model_for_test__"
os.environ["AME_EMBED_DIM"] = "64"          # small = fast hash vectors
os.environ["AME_RERANKER_MODEL"] = "__nonexistent_reranker_for_test__"  # fail fast, no network
os.environ["AME_LLM_BASE_URL"] = ""
os.environ["AME_LLM_API_KEY"] = ""
# Force HF offline so sentence-transformers fails fast (no network wait) and
# the embedder drops straight into the deterministic hash fallback.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest  # noqa: E402

from engine import db, embed  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    """Wipe + re-init the DB before every test for full isolation."""
    db.init_db(dim=embed.dim(), force=True)
    yield
