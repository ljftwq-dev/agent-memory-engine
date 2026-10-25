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
    social_get_fan    - one fan's card (relation/favor/impressions) for personalized replies
    social_update_fan - create/update a fan card (favor +/-, impressions, blacklist)
    social_mood       - get/set the persona agent's persistent mood
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


def _format_recall(results):
    """Format recall results as a readable, numbered string for the IDE.

    None-safe on ``distance``: BM25-only hybrid candidates carry
    ``distance=None`` (they have no vector match). A naive
    ``1 - m.get("distance", 0)`` is a trap - ``dict.get(k, default)`` evaluates
    ``default`` *eagerly*, and when the key exists with value ``None`` the
    default (0) is ignored, so it does ``1 - None`` and crashes the whole tool.
    Branch explicitly instead. See tests/test_mcp_server.py.
    """
    if not results:
        return "(no relevant memories found)"
    lines = []
    for i, m in enumerate(results, 1):
        if "rerank" in m:
            rel, tag = m["rerank"], "rerank"
        elif "rrf" in m:
            rel, tag = m["rrf"], "rrf"
        else:
            dist = m.get("distance")
            rel = (1.0 - dist) if dist is not None else 0.0
            tag = "sim"
        lines.append(
            f"[{i}] {tag}={rel:.2f}  {m.get('topic', '')}\n"
            f"    {m.get('summary', '')}"
        )
    return "\n".join(lines)


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
    return _format_recall(r.get("results", []))


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


@mcp.tool()
def social_get_fan(platform: str, uid: str) -> str:
    """Look up one fan's social card (relation tier, favor score 0-100,
    impressions, traits). Use before replying to a commenter/follower so the
    persona agent can personalize the response ('she remembers you').

    Args:
        platform: Audience platform id, e.g. 'bilibili'.
        uid: The user's id on that platform (mid for Bilibili).
    """
    try:
        r = _get("/social/fan", {"platform": platform, "uid": uid})
    except Exception as e:
        return _server_down(e)
    fan = r.get("fan")
    if not fan:
        return f"(no card yet for {platform}:{uid})"
    return (
        f"{platform}:{uid}  {fan.get('nickname', '')}\n"
        f"    relation: {fan.get('relation')}   favor: {fan.get('favor')}/100\n"
        f"    impressions:\n" + "\n".join(
            f"      - {line}" for line in (fan.get("impressions") or "").splitlines()
        ) + ("\n    traits:\n" + "\n".join(
            f"      - {line}" for line in (fan.get("traits") or "").splitlines()
        ) if fan.get("traits") else "")
    )


@mcp.tool()
def social_update_fan(platform: str, uid: str, nickname: str = "",
                      favor: int = -1, favor_delta: int = 0,
                      impression: str = "", trait: str = "",
                      relation: str = "", blacklist: bool = False) -> str:
    """Create or merge-update a fan card. Pass favor_delta for incremental
    changes (+/-), or favor for an absolute set; impressions/traits append as
    log lines (last 20 kept). Relation tier auto-derives from favor unless an
    explicit relation is given; blacklist=True marks 'blocked' (persists until
    blacklist=False).

    Args:
        platform: Audience platform id, e.g. 'bilibili'.
        uid: The user's id on that platform.
        nickname: Display name (optional, kept if empty).
        favor: Absolute favor set (0-100). Ignored when negative.
        favor_delta: Increment applied to current favor (0 = no change).
        impression: One-line impression to append to this fan's history.
        trait: One-line trait/note (memes they use, preferences).
        relation: Manual override ('stranger'|'fan'|'regular'|'friend'|'master').
        blacklist: True to block, False to unblock, omit to leave as-is.
    """
    payload = {"platform": platform, "uid": uid}
    if nickname:
        payload["nickname"] = nickname
    if favor >= 0:
        payload["favor"] = favor
    if favor_delta:
        payload["favor_delta"] = favor_delta
    if impression:
        payload["impression"] = impression
    if trait:
        payload["trait"] = trait
    if relation:
        payload["relation"] = relation
    payload["blacklist"] = blacklist
    try:
        r = _post("/social/fan", payload)
    except Exception as e:
        return _server_down(e)
    fan = r.get("fan", {})
    return (f"updated {platform}:{uid} -> relation={fan.get('relation')} "
            f"favor={fan.get('favor')}/100")


@mcp.tool()
def social_mood(label: str = "", valence: int = 0, arousal: int = 0,
                reason: str = "") -> str:
    """Get (label empty) or set (label given) the persona agent's current mood.
    Mood persists across sessions, giving emotional continuity; a short event
    log is kept. Valence -100..100 (bad..good), arousal 0..100 (calm..excited).

    Args:
        label: Mood name to set (e.g. 'happy', 'grumpy'). Empty = just read.
        valence: How positive the mood is (-100..100).
        arousal: How energized the mood is (0..100).
        reason: Why the mood changed (logged).
    """
    try:
        if label:
            r = _post("/social/mood", {"label": label, "valence": valence,
                                       "arousal": arousal, "reason": reason})
            cur = r.get("mood", {})
            return f"mood set: {cur.get('label')} (v={cur.get('valence')}, a={cur.get('arousal')})"
        r = _get("/social/mood", {"limit": 5})
    except Exception as e:
        return _server_down(e)
    cur = r.get("current")
    if not cur:
        return "(no mood set yet)"
    events = r.get("events", [])
    lines = [f"current: {cur.get('label')} (v={cur.get('valence')}, a={cur.get('arousal')}) "
             f"updated {cur.get('updated_at', '')[:16]}"]
    for ev in events:
        lines.append(f"  [{(ev.get('ts', '') or '')[:16]}] {ev.get('label')} - {ev.get('reason', '')}")
    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
