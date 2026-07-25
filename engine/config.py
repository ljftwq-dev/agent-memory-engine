"""config.py - centralize all settings (read from .env / env vars).

No hardcoded paths / secrets. Every user-specific knob comes from here.

Env var convention: prefix ``AME_`` (e.g. ``AME_DB_PATH``).
A lightweight .env loader is bundled so users don't need python-dotenv.
"""
import os

DEFAULTS = {
    "DB_PATH": os.path.join(os.path.expanduser("~"), ".agent-memory", "memory.db"),
    "EMBED_MODEL": "BAAI/bge-m3",
    "EMBED_DIM": "1024",
    "LLM_BASE_URL": "",          # empty = LLM summarization disabled
    "LLM_API_KEY": "",
    "LLM_MODEL": "glm-4-flash",
    "RECALL_THRESHOLD": "0.9",   # distance upper bound (larger = looser gate)
    "RECALL_POOL": "15",         # stage-A wide recall count
    "ALPHA": "0.5",              # strength weight in rerank score
    "MIN_STRENGTH": "0.05",
    "DEDUP_THRESHOLD": "0.45",   # distance <= this => merge into existing memory
}

_loaded = False


def _load_dotenv():
    """Minimal .env parser (no python-dotenv dependency)."""
    candidates = []
    env_override = os.environ.get("AME_ENV_FILE")
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    for env_path in candidates:
        if not env_path or not os.path.isfile(env_path):
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
        return


def _ensure():
    global _loaded
    if not _loaded:
        _load_dotenv()
        _loaded = True


def get(key, default=None):
    """Read AME_{KEY} env var > DEFAULTS[key] > default."""
    _ensure()
    env_key = "AME_" + key
    val = os.environ.get(env_key)
    if val is not None and val != "":
        return val
    return DEFAULTS.get(key, default)


def db_path():
    return get("DB_PATH")


def embed_model():
    return get("EMBED_MODEL")


def embed_dim():
    return int(get("EMBED_DIM"))


def llm_base_url():
    return get("LLM_BASE_URL")


def llm_api_key():
    return get("LLM_API_KEY")


def llm_model():
    return get("LLM_MODEL")


def llm_enabled():
    return bool(llm_base_url() and llm_api_key())


def recall_threshold():
    return float(get("RECALL_THRESHOLD"))


def recall_pool():
    return int(get("RECALL_POOL"))


def alpha():
    return float(get("ALPHA"))


def min_strength():
    return float(get("MIN_STRENGTH"))


def dedup_threshold():
    return float(get("DEDUP_THRESHOLD"))
