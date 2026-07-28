# Contributing

Thanks for your interest in improving agent-memory-engine! This is a small
project, but contributions are welcome — bug reports, fixes, docs, benchmark
cases, and new integration examples all help.

## Reporting bugs / suggesting features

Open an [issue](https://github.com/ljftwq-dev/agent-memory-engine/issues). A
good report includes: what you did, what you expected, what happened, and the
engine version (check `GET /health` or `python -c "from engine import config; print(config.db_path())"`).

## Development setup

```bash
git clone https://github.com/ljftwq-dev/agent-memory-engine
cd agent-memory-engine
pip install -e ".[all]"        # core + real embeddings + reranker + dev deps
cp .env.example .env           # defaults work out of the box
pytest -q                      # should be green before you start
```

Tests run on the hash-fallback embedder (no model download), so they're fast
and CI-friendly.

## Pull requests

1. Fork → feature branch (`feat/...`, `fix/...`, `docs/...`).
2. Keep commits focused; follow the existing
   [Conventional Commits](https://www.conventionalcommits.org/) style
   (`feat:`, `fix:`, `docs:`, `chore:`, ...).
3. Add or update tests for behavior changes.
4. Make sure `pytest -q` passes locally.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a PR describing **what** and **why** (link any related issue).

## Code style

- Python: PEP 8; type hints welcome where they aid clarity.
- Keep the engine dependency-light (stdlib first; heavy models stay optional
  behind the `[embed]` / `[reranker]` extras).
- Every config knob lives in `engine/config.py` — no hardcoded paths or secrets.
- Secrets never go in code.

## Adding to the benchmark

The benchmark lives in `benchmark/`. Memories and queries are plain Python
lists in `dataset.py` — append and re-run `python benchmark/run_benchmark.py`.
Hand-labeled graded relevance (2 = highly relevant, 1 = partial) is what makes
the numbers meaningful, so label carefully.
