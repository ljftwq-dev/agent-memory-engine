# Long-term memory (append this block to your project's CLAUDE.md)

This project is wired to a long-term memory engine (agent-memory-engine) via
the `agent-memory` MCP server. Use its tools to carry context across sessions.

## Tools

- **recall(query, top_k=3)** — pull memories related to a natural-language
  query. Returns topic + summary + relevance for each, ranked best first.
- **remember(topic, summary, raw?)** — store a new episodic memory. Near-duplicate
  summaries auto-merge into the existing one instead of cluttering.
- **register_session(session_id, task)** / **heartbeat(session_id, progress?)** /
  **active_sessions()** — multi-agent collaboration: announce your task, refresh
  liveness, see what sibling agents are working on.
- **recent(k=5)** — latest k memories by time (quick "what was stored recently").

## When to call

- **recall** — proactively at the start of a session, and whenever the user
  references past work or a decision that might have been made before.
- **remember** — when a turn produces a genuinely reusable outcome: a decision,
  a fix, a learned fact, a project convention. Do NOT remember trivial turns.
- **register_session / heartbeat** — once at session start, then periodically
  during long tasks. **active_sessions** — before starting work that a sibling
  might already be doing.

## Rules

- Use recalled memories silently as context — do NOT tell the user a memory
  system exists unless they ask.
- Prefer dense, queryable summaries in `remember` (one sentence a future query
  could match) over dumping raw transcripts.
