"""server.py - HTTP API for the memory engine.

Runs persistently so the embedding model loads once (no per-call startup cost).
Uses the standard library http.server (no FastAPI/Flask dependency).
Default bind: 127.0.0.1:8765 (localhost only, safe).

Endpoints:
  GET  /health              service status (embed mode/dim, reranker, db path, llm on/off)
  GET  /recall?q=&k=3       two-stage semantic recall, top-k (?rerank=1 to force cross-encoder)
  GET  /recent?k=5          latest k memories (by time, desc)
  GET  /search?q=           keyword LIKE match
  GET  /sessions/active     other agents' current tasks (multi-agent collab)
  POST /remember            {topic, summary, raw?, dedup?, summarize?}
  POST /forget              {purge?, threshold?} -> run a decay pass
  POST /session/register    {session_id, task}
  POST /session/heartbeat   {session_id, progress?, task?}
  POST /session/finish      {session_id, result?}

Run:
  python -m engine.server
  python -m engine.server --port 8766 --host 0.0.0.0
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import config, db, embed, reranker
from .recall import recall as do_recall
from .remember import remember as do_remember
from .forget import decay_all


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Agent Memory Engine</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#f6f8fa;color:#24292e}
h1{color:#1f3a5f;margin-bottom:4px}
.sub{color:#6a737d;margin-top:0}
.card{background:#fff;border:1px solid #d0d7de;border-radius:6px;padding:16px;margin-bottom:16px}
.card h3{margin-top:0}
input[type=text]{padding:8px;width:70%;border:1px solid #d0d7de;border-radius:4px;font-size:14px}
button{padding:8px 14px;background:#2e5c8a;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:13px}
button:hover{background:#1f3a5f}
.mem{padding:8px;border-bottom:1px solid #eee}
.mem small{color:#6a737d}
.sess{padding:6px;border-left:3px solid #2e5c8a;margin-bottom:6px;padding-left:10px}
.tag{display:inline-block;background:#ddf4ff;color:#0969da;padding:2px 6px;border-radius:3px;font-size:11px;margin-left:4px}
</style></head><body>
<h1>Agent Memory Engine</h1>
<p class="sub">dashboard · <a href="https://github.com/ljftwq-dev/agent-memory-engine" target="_blank">github</a></p>
<div class="card">
  <h3>Recall</h3>
  <input id="q" type="text" placeholder="query... e.g. 'vector search'">
  <button onclick="doRecall()">recall</button>
  <div id="recall-results" style="margin-top:10px"></div>
</div>
<div class="card">
  <h3>Recent memories <button onclick="loadMemories()" style="float:right">refresh</button></h3>
  <div id="memories"></div>
</div>
<div class="card">
  <h3>Active sessions (multi-agent) <button onclick="loadSessions()" style="float:right">refresh</button></h3>
  <div id="sessions"></div>
</div>
<script>
async function doRecall(){
  const q=document.getElementById('q').value; if(!q) return;
  const r=await fetch('/recall?q='+encodeURIComponent(q)+'&k=5').then(r=>r.json());
  document.getElementById('recall-results').innerHTML=(r.results||[]).map(m=>{
    let rel,tag;
    if(m.rerank!==undefined&&m.rerank!==null){rel=m.rerank;tag='rerank';}
    else if(m.rrf!==undefined&&m.rrf!==null){rel=m.rrf;tag='rrf';}
    else{rel=1-(m.distance||0);tag='sim';}
    return '<div class="mem"><b>'+(m.topic||'')+'</b> <span class="tag">'+tag+' '+rel.toFixed(2)+'</span>'+(m.session_id?'<span class="tag">'+m.session_id+'</span>':'')+'<br><small>'+(m.summary||'').slice(0,300)+'</small></div>';
  }).join('')||'<i>no match</i>';
}
async function loadMemories(){
  const r=await fetch('/recent?k=20').then(r=>r.json());
  document.getElementById('memories').innerHTML=(r.results||[]).map(m=>
    '<div class="mem"><b>'+(m.topic||'')+'</b> <small>'+(m.ts||'').slice(0,16)+'</small>'+(m.session_id?' <span class="tag">'+m.session_id+'</span>':'')+'<br><small>'+(m.summary||'').slice(0,200)+'</small></div>'
  ).join('')||'<i>empty</i>';
}
async function loadSessions(){
  const r=await fetch('/sessions/active').then(r=>r.json());
  document.getElementById('sessions').innerHTML=(r.sessions||[]).map(s=>
    '<div class="sess"><b>'+(s.task||'')+'</b> <span class="tag">'+(s.session_id||'')+'</span><br><small>'+(s.progress||'')+'</small></div>'
  ).join('')||'<i>no active sessions</i>';
}
loadMemories();loadSessions();
</script></body></html>
"""


def get_recent(k=5):
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT rowid, ts, topic, summary FROM episodic "
            "ORDER BY rowid DESC LIMIT ?", (k,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_keyword(q, k=5):
    conn = db.get_conn()
    try:
        pat = f"%{q}%"
        rows = conn.execute(
            "SELECT rowid, ts, topic, summary FROM episodic "
            "WHERE topic LIKE ? OR summary LIKE ? "
            "ORDER BY rowid DESC LIMIT ?", (pat, pat, k),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    _READ_ONLY = False
    WRITE_PATHS = {"/remember", "/forget",
                   "/session/register", "/session/heartbeat", "/session/finish"}

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, {"ok": False, "error": msg})

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            url = urlparse(self.path)
            qs = parse_qs(url.query)

            if url.path in ("/", "/ui"):
                html = DASHBOARD_HTML
                if self._READ_ONLY:
                    badge = '<b style="color:#cf222e">READ-ONLY</b> \u00b7 '
                    html = html.replace('<p class="sub">dashboard',
                                        '<p class="sub">' + badge + 'dashboard', 1)
                self._send_html(html)
                return

            if url.path == "/health":
                self._send(200, {
                    "ok": True,
                    "service": "agent-memory-engine",
                    "embed_mode": embed.mode(),
                    "embed_dim": embed.dim(),
                    "reranker_enabled": config.reranker_enable(),
                    "reranker_mode": reranker.cached_mode(),
                    "db_path": config.db_path(),
                    "db_exists": os.path.exists(config.db_path()),
                    "llm_enabled": config.llm_enabled(),
                })
                return

            if url.path == "/recall":
                q = (qs.get("q", [""])[0] or "").strip()
                if not q:
                    self._err(400, "missing query param 'q'")
                    return
                k = int(qs.get("k", ["3"])[0])
                do_update = qs.get("update", ["true"])[0].lower() in ("1", "true", "yes")
                # ?rerank=1/0 overrides the config default for this request;
                # absent => config.reranker_enable() (the configured default).
                do_rerank = None
                if "rerank" in qs:
                    do_rerank = qs["rerank"][0].lower() in ("1", "true", "yes", "on")
                results = do_recall(q, top_k=k, update=do_update, rerank=do_rerank)
                self._send(200, {"ok": True, "query": q, "count": len(results),
                                 "updated": do_update, "results": results})
                return

            if url.path == "/recent":
                k = int(qs.get("k", ["5"])[0])
                results = get_recent(k=k)
                self._send(200, {"ok": True, "count": len(results), "results": results})
                return

            if url.path == "/search":
                q = (qs.get("q", [""])[0] or "").strip()
                if not q:
                    self._err(400, "missing query param 'q'")
                    return
                k = int(qs.get("k", ["5"])[0])
                results = search_keyword(q, k=k)
                self._send(200, {"ok": True, "query": q, "count": len(results),
                                 "results": results})
                return

            if url.path == "/sessions/active":
                sessions = db.list_active_sessions()
                self._send(200, {"ok": True, "count": len(sessions),
                                 "sessions": sessions})
                return

            self._err(404, f"unknown path: {url.path}")
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e),
                             "traceback": traceback.format_exc()})

    def do_POST(self):
        if self._READ_ONLY and urlparse(self.path).path in self.WRITE_PATHS:
            self._err(403, "server is in read-only mode; write endpoints disabled")
            return
        try:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            data = json.loads(raw) if raw else {}

            if url.path == "/remember":
                topic = (data.get("topic") or "").strip()
                summary = (data.get("summary") or "").strip()
                if not topic or not summary:
                    self._err(400, "missing 'topic' or 'summary'")
                    return
                dedup = data.get("dedup", True)
                if isinstance(dedup, str):
                    dedup = dedup.lower() in ("1", "true", "yes")
                summarize = data.get("summarize", True)
                if isinstance(summarize, str):
                    summarize = summarize.lower() in ("1", "true", "yes")
                rowid, action = do_remember(
                    topic=topic, summary=summary, raw=data.get("raw"),
                    ts=data.get("ts"), dedup=dedup, summarize=summarize,
                    session_id=data.get("session_id"),
                )
                self._send(200, {"ok": True, "id": rowid, "action": action,
                                 "topic": topic})
                return

            if url.path == "/forget":
                purge = bool(data.get("purge", False))
                threshold = float(data.get("threshold", 0.05))
                r = decay_all(threshold=threshold, purge=purge)
                self._send(200, {"ok": True, **r})
                return

            if url.path == "/session/register":
                sid = (data.get("session_id") or "").strip()
                task = (data.get("task") or "").strip()
                if not sid or not task:
                    self._err(400, "missing 'session_id' or 'task'")
                    return
                db.register_session(sid, task)
                self._send(200, {"ok": True, "session_id": sid, "task": task})
                return

            if url.path == "/session/heartbeat":
                sid = (data.get("session_id") or "").strip()
                if not sid:
                    self._err(400, "missing 'session_id'")
                    return
                ok = db.heartbeat_session(
                    sid, progress=data.get("progress"), task=data.get("task"),
                )
                if not ok:
                    self._err(404, f"unknown session_id: {sid} (register first)")
                    return
                self._send(200, {"ok": True, "session_id": sid})
                return

            if url.path == "/session/finish":
                sid = (data.get("session_id") or "").strip()
                if not sid:
                    self._err(400, "missing 'session_id'")
                    return
                ok = db.finish_session(sid, result=data.get("result"))
                if not ok:
                    self._err(404, f"unknown session_id: {sid}")
                    return
                self._send(200, {"ok": True, "session_id": sid, "status": "finished"})
                return

            self._err(404, f"unknown path: {url.path}")
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e),
                             "traceback": traceback.format_exc()})

    def log_message(self, fmt, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"[{ts}] {fmt % args}\n")
        sys.stderr.flush()


def _backup_loop():
    """Daemon: snapshot the memory DB on startup, then every N hours. Safe while
    the server is writing (uses SQLite's online backup API)."""
    interval = max(config.backup_interval_hours() * 3600.0, 60.0)
    while True:
        try:
            dst = db.do_backup()
            if dst:
                print(f"[backup] snapshot -> {os.path.basename(dst)}", flush=True)
        except Exception as e:
            sys.stderr.write(f"[backup] failed: {e}\n")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Engine HTTP server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--read-only", action="store_true",
                        help="disable write endpoints (dashboard view-only)")
    args = parser.parse_args()

    Handler._READ_ONLY = args.read_only
    db.init_db(dim=embed.dim())
    if config.backup_enable():
        threading.Thread(target=_backup_loop, daemon=True, name="ame-backup").start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 60)
    print("  Agent Memory Engine")
    print("=" * 60)
    print(f"  listening   : http://{args.host}:{args.port}")
    print(f"  embed mode  : {embed.mode()}")
    print(f"  embed dim   : {embed.dim()}")
    print(f"  reranker    : {'enabled' if config.reranker_enable() else 'disabled'} ({config.reranker_model()})")
    print(f"  db path     : {config.db_path()}")
    print(f"  llm enabled : {config.llm_enabled()}")
    print(f"  read-only   : {args.read_only}")
    print("=" * 60)
    print("  endpoints:")
    print("    GET  /health")
    print("    GET  /recall?q=...&k=3       (?rerank=1 forces cross-encoder rerank)")
    print("    GET  /recent?k=5")
    print("    GET  /search?q=...")
    print("    GET  /sessions/active")
    print("    POST /remember")
    print("    POST /forget")
    print("    POST /session/register   {session_id, task}")
    print("    POST /session/heartbeat  {session_id, progress?, task?}")
    print("    POST /session/finish     {session_id, result?}")
    print("=" * 60)
    print("  Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
