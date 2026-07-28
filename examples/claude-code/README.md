# Claude Code integration

Long-term memory for Claude Code sessions, via the engine's MCP server.

## Setup

1. **Install + run the engine** (loads BGE-m3 once, stays up as an HTTP server):
   ```bash
   pip install "agent-memory-engine-ljf[all]"
   python -m engine.server            # listens on 127.0.0.1:8765
   ```

2. **Copy `.mcp.json`** into your project root. Claude Code auto-discovers it
   on startup and connects the `agent-memory` MCP server. (If you already have
   an `.mcp.json`, merge the `agent-memory` entry into `mcpServers`.)

3. **Append the memory block** from [`CLAUDE.md`](CLAUDE.md) to your project's
   `CLAUDE.md` (or create one). This tells Claude *when* to recall/remember.

4. **Restart Claude Code.** Run `/mcp` to confirm `agent-memory` is connected.

## How it works

Claude Code calls the engine's MCP tools directly:

| tool | when Claude calls it |
|---|---|
| `recall` | at session start, or when past context is relevant |
| `remember` | when a turn produces a reusable decision / fix / fact |
| `register_session` / `heartbeat` / `active_sessions` | multi-agent collaboration |

This is **model-driven**: Claude decides when memory is relevant, guided by the
`CLAUDE.md` instructions. Contrast with the [opencode plugin](../opencode/memory.ts),
which is **hook-driven** — every turn auto-recalls and auto-remembers
deterministically. Both styles are valid; the model-driven one is more
Claude-native (the model judges relevance rather than a fixed hook firing every
turn).

## Verify

```
you: do you remember what we decided about the retry logic last week?
claude: [calls recall("retry logic")] → uses the returned memory to answer
```

## Notes

- The MCP server (`engine/mcp_server.py`) is a thin stdio proxy to the HTTP
  server — it starts fast and never loads the embedding model itself, so it
  won't slow down Claude Code's startup.
- All traffic stays on localhost; nothing leaves your machine.
