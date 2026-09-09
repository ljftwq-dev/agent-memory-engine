# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-09-09

Hardening & platforms pass — security, concurrency, and Windows are now first-class.

### Added
- **Bearer-token auth** for the HTTP API (`AME_API_TOKEN`, opt-in): when set, every endpoint requires `Authorization: Bearer` — except `/health` and `/metrics`, which stay open for health checks. Constant-time compare, `WWW-Authenticate` on 401. Docker example now publishes to localhost only.
- **`GET /metrics`** endpoint — usage counters (`memories_total`, `recalls_served`, `avg_results_per_recall`, `remember_merges`, `forgets`, `embed_mode`, ...) for lightweight observability.
- **Windows support** — one-line `install.ps1` installer + `windows-latest` in CI (hash-fallback path).
- **Pluggable jieba tokenization** — word-level Chinese BM25 indexing (`[jieba]` extra; per-char fallback stays default); fixes CJK FTS indexing.
- **Cursor MCP integration example** (`examples/cursor/` — mcp.json + agent rules).
- **Auth & concurrency contract documented** in README (when to enable the token, WAL semantics).

### Changed
- Server binds to `127.0.0.1` only by default — nothing outside the machine can reach it unless you explicitly publish further.
- README positioning: "agent-agnostic HTTP memory API, MCP-ready".

### Fixed
- **WAL + `busy_timeout` on every connection** — concurrent readers never block behind a writer; cross-process access (e.g. MCP stdio against the same DB file) degrades gracefully instead of raising `database is locked`. Ships with an N-thread write stress test.
- Engine robustness: validate `k` param, honor read-only mode on recall, auto-repair NULL `tau` rows.

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

[Unreleased]: https://github.com/ljftwq-dev/agent-memory-engine/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ljftwq-dev/agent-memory-engine/compare/v0.2.0...v0.3.0
[0.1.1]: https://github.com/ljftwq-dev/agent-memory-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ljftwq-dev/agent-memory-engine/releases/tag/v0.1.0
