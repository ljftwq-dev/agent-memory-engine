# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-07-28

Project "storefront" pass — the repo now shows what it does, not just describes it.

### Docs
- **Architecture diagram**: 4-layer overview (agent host → interface → engine core → storage) with the two-stage retrieve + write pipelines. Reproducible matplotlib source (`docs/make_architecture.py`).
- **Web dashboard screenshot** on the README — recall results, recent memories, and live multi-agent sessions, rendered from `seed_demo` data.
- **Benchmark results surfaced on the README** — nDCG@5 / Recall@5 for pure-vector vs hybrid vs hybrid+reranker, plus per-stage gains.
- **Social preview card** (1200×630) + generator (`docs/make_social_card.py`) for GitHub social preview / OG image.

## [0.1.0] - 2026-07-25

Initial public release. A long-term memory engine for coding agents.

### Added
- **Two-stage retrieval + gating** — wide KNN recall → drop pure noise → rerank by `score = α·strength + (1-α)·relevance` → top-k. Stops semantically-adjacent-but-useless junk polluting the prompt.
- **Hybrid recall** — vector KNN + FTS5 BM25, fused via reciprocal rank fusion (RRF). Catches keyword hits the vector path misses.
- **Cross-encoder precision rerank** (optional) — `bge-reranker-v2-m3` re-scores `(query, candidate)` pairs after fusion. The classic two-stage IR pattern.
- **Ebbinghaus decay** — `strength = exp(-Δt/τ)`, each recall does `τ *= 1.5`. Use-it-or-lose-it, no RL training needed.
- **Web dashboard** — `GET /` serves a single-page HTML UI (recall / recent / active sessions). No new deps.
- **Multi-agent collaboration** — sessions register their task; siblings query `/sessions/active` to see who's doing what. Ships with an opencode plugin reference (`examples/opencode/memory.ts`).
- **MCP server** (stdio) — expose the engine to MCP-capable IDEs (opencode / ZCode / Claude Code).
- **Retrieval benchmark** — 40 coding-agent memories × 24 hand-labeled queries, graded relevance, nDCG@5 / Recall@5 ablation across the three configs.
- **LLM summarization** (optional) — condense each turn into a semantic sentence before embedding. Any OpenAI-compatible endpoint.
- **Single SQLite file** storage via `sqlite-vec` — structured data + vector index in one `.db`, zero ops.
- **CI** (GitHub Actions, pytest) + **Chinese README** (`README-zh.md`).

### Fixed
- Reranker now reranks **all** gated candidates instead of a top-N subset (Recall jumped 0.771 → 0.979).
- Additive schema migration for databases from older versions.
- Serialized DB writes + automatic periodic backups (safe snapshots via SQLite online backup).
- Multi-agent `session_id` tagging to prevent cross-talk between sessions.

[Unreleased]: https://github.com/ljftwq-dev/agent-memory-engine/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ljftwq-dev/agent-memory-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ljftwq-dev/agent-memory-engine/releases/tag/v0.1.0
