"""agent-memory-engine: two-stage retrieval + gating memory engine for coding agents."""

from . import config  # noqa: F401  (re-export so `from engine import config` works)

__version__ = "0.1.0"

__all__ = ["config", "__version__"]
