"""mcp_server.py - expose the memory engine to MCP-capable IDEs (stdio).

Thin HTTP proxy: the IDE launches this as a subprocess, and it forwards tool
calls to the engine's persistent HTTP server (server.py). That server is the
single model host (loads BGE-m3 / reranker once); this subprocess stays
lightweight and never touches the models - so it starts fast, uses little RAM,
and crucially never writes model-load progress to stdout (which would corrupt
MCP's JSON-RPC stream).

Same local-only philosophy: stdio transport, calls 127.0.0.1, your SQLite file,
nothing leaves the machine.

Setup:
  1. Run the engine HTTP server once (it loads the models and stays up):
         python -m engine.server --port 8765
  2. Point IDEs at this MCP server. Configure in an MCP client:
         {"command": "python", "args": ["-m", "engine.mcp_server"],
          "env": {"AME_HTTP_URL": "http://127.0.0.1:8765"}}

Tools exposed:
    recall            - pull memories related to a query (call at session start)
    remember          - store a memory (call when a turn yields something worth keeping)
    register_session  - announce this agent's task so siblings see it
    heartbeat         - refresh liveness / report progress
    finish_session    - mark the session done
    active_sessions   - see what OTHER agents are working on (multi-agent awareness)
    recent            - latest k memories by time
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP


def _base():
    return os.environ.get("AME_HTTP_URL", "http://127.0.0.1:8765").rstrip("/")


def _get(path, params=None):
    url = _base() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _base() + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _server_down(e):
    return (f"engine HTTP server not reachable at {_base()} ({e}). "
            f"Start it with:  python -m engine.server")


mcp = FastMCP("agent-memory-engine")


@mcp.tool()
def recall(query: str, top_k: int = 3) -> str:
    """Recall memories related to a query. Call this at the start of a session
    or whenever context from past work would help. Returns up to top_k memories
    ranked by relevance, with their topic, summary, and a relevance score.

    Args:
        query: What you're working on or wondering about, in natural language.
        top_k: How many memories to return (default 3).
    """
    try:
        r = _get("/recall", {"q": query, "k": top_k, "update": "true"})
    except Exception as e:
        return _server_down(e)
    results = r.get("results", [])
    if not results:
        return "(no relevant memories found)"
    lines = []
    for i, m in enumerate(results, 1):
        rel = m.get("rerank", m.get("rrf", 1 - m.get("distance", 0)))
        tag = "rerank" if "rerank" in m else ("rrf" if "rrf" in m else "sim")
        lines.append(
            f"[{i}] {tag}={rel:.2f}  {m.get('topic', '')}\n"
            f"    {m.get('summary', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def remember(topic: str, summary: str, raw: str = "", session_id: str = "") -> str:
    """Store a new episodic memory. Call this when a turn produces something
    worth remembering later (a decision, a fix, a fact learned). Near-duplicate
    summaries auto-merge into the existing memory instead of creating clutter.

    Args:
        topic: Short label for the memory (a few words).
        summary: The semantic retrieval key - one sentence a future query could
            match. This is what gets embedded, so make it information-dense.
        raw: Optional full original text (the dialogue / code). Stored but not
            embedded; useful for later reference.
        session_id: Optional tag for which agent session wrote this (helps tell
            memories apart in multi-agent setups).
    """
    payload = {"topic": topic, "summary": summary, "dedup": True, "summarize": False}
    if raw:
        payload["raw"] = raw
    if session_id:
        payload["session_id"] = session_id
    try:
        r = _post("/remember", payload)
    except Exception as e:
        return _server_down(e)
    return f"{r.get('action', '?')} memory #{r.get('id')}  |  topic: {topic}"


@mcp.tool()
def register_session(session_id: str, task: str) -> str:
    """Announce that THIS agent session is active and what it's working on, so
    sibling agents can see it (multi-agent collaboration). Call once near the
    start of a session. Re-registering with a new task updates it.

    Args:
        session_id: A stable id for this agent session (e.g. the IDE's session
            id, or any unique string you reuse across the session).
        task: One line describing what you're doing.
    """
    try:
        _post("/session/register", {"session_id": session_id, "task": task})
    except Exception as e:
        return _server_down(e)
    return f"registered session '{session_id}': {task}"


@mcp.tool()
def heartbeat(session_id: str, progress: str = "", task: str = "") -> str:
    """Refresh this session's liveness and optionally report progress. Sessions
    that go silent longer than the timeout are treated as stale. Call
    periodically during long tasks.

    Args:
        session_id: The id you registered with.
        progress: Optional one-line status update.
        task: Optional new task if the focus changed.
    """
    payload = {"session_id": session_id}
    if progress:
        payload["progress"] = progress
    if task:
        payload["task"] = task
    try:
        _post("/session/heartbeat", payload)
    except Exception as e:
        return _server_down(e)
    return "ok"


@mcp.tool()
def finish_session(session_id: str, result: str = "") -> str:
    """Mark this session finished, optionally stashing a result summary. Call
    when the task is done so siblings know the agent is no longer active.

    Args:
        session_id: The id you registered with.
        result: Optional one-line summary of the outcome.
    """
    payload = {"session_id": session_id}
    if result:
        payload["result"] = result
    try:
        _post("/session/finish", payload)
    except Exception as e:
        return _server_down(e)
    return "finished"


@mcp.tool()
def active_sessions() -> str:
    """See what OTHER agents are currently working on (multi-agent awareness).
    Returns active sessions that sent a heartbeat recently. Use this to discover
    siblings' in-progress work and avoid duplicating effort.
    """
    try:
        r = _get("/sessions/active")
    except Exception as e:
        return _server_down(e)
    sessions = r.get("sessions", [])
    if not sessions:
        return "(no active sessions)"
    lines = []
    for s in sessions:
        lines.append(f"[{s.get('session_id', '?')}] {s.get('task', '')}")
        if s.get("progress"):
            lines.append(f"    progress: {s['progress']}")
    return "\n".join(lines)


@mcp.tool()
def recent(k: int = 5) -> str:
    """Return the latest k memories by time (newest first), regardless of query
    relevance. Useful for a quick 'what was stored recently' glance.
    """
    try:
        r = _get("/recent", {"k": k})
    except Exception as e:
        return _server_down(e)
    items = r.get("results", [])
    if not items:
        return "(no memories yet)"
    return "\n".join(
        f"#{m.get('rowid', '?')}  {(m.get('ts', '') or '')[:16]}  {m.get('topic', '')}\n"
        f"    {m.get('summary', '')}" for m in items
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
