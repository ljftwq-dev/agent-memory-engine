# Cursor integration

Long-term memory for Cursor sessions, via the engine's MCP server.

## Setup

1. **Install + run the engine** (loads BGE-m3 once, stays up as an HTTP server):
   ```bash
   pip install "agent-memory-engine-ljf[all]"
   python -m engine.server            # listens on 127.0.0.1:8765
   ```

2. **Register the MCP server** in Cursor — either way works:

   - **Project-level (recommended)**: copy [`mcp.json`](mcp.json) to
     `.cursor/mcp.json` in your project root. Cursor auto-discovers it.
     (If you already have one, merge the `agent-memory` entry into
     `mcpServers`.)

   - **Global**: Cursor Settings → MCP → *Add new global MCP server*, paste the
     contents of `mcp.json`.

3. **Add the memory rules** from [`agent-rules.md`](agent-rules.md) to
   Cursor Settings → Rules for AI (or `.cursor/rules/` in newer versions).
   This tells the agent *when* to recall/remember.

4. **Restart Cursor.** Check Settings → MCP: `agent-memory` should show as
   running (green).

## Verify

```
you: do you remember what we decided about the retry logic last week?
cursor: [calls recall("retry logic")] → uses the returned memory to answer
```

Sanity-check the engine itself at any time:

```bash
curl -s http://127.0.0.1:8765/health    # {"ok": true, ...}
```

## How it works

Cursor calls the engine's MCP tools directly:

| tool | when the agent calls it |
|---|---|
| `recall` | at session start, or when past context is relevant |
| `remember` | when a turn produces a reusable decision / fix / fact |
| `register_session` / `heartbeat` / `active_sessions` | multi-agent collaboration |

This is **model-driven**: the agent decides when memory is relevant, guided by
the rules you added in step 3. Contrast with the [opencode plugin](../opencode/memory.ts),
which is **hook-driven** — every turn auto-recalls and auto-remembers
deterministically. Both styles are valid; the model-driven one keeps the agent
in charge of relevance.

## Notes

- The MCP server (`engine/mcp_server.py`) is a thin stdio proxy to the HTTP
  server — it starts fast and never loads the embedding model itself, so it
  won't slow Cursor's startup.
- All traffic stays on localhost; nothing leaves your machine.
- Windows note: if Cursor can't find `agent-memory-mcp`, use the full path
  from `pip show agent-memory-engine-ljf` (Scripts dir), e.g.
  `"command": "C:\\...\\Scripts\\agent-memory-mcp.exe"`.
