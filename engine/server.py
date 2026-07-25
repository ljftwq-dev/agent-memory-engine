"""server.py - HTTP API for the memory engine.

Runs persistently so the embedding model loads once (no per-call startup cost).
Uses the standard library http.server (no FastAPI/Flask dependency).
Default bind: 127.0.0.1:8765 (localhost only, safe).

Endpoints:
  GET  /health              service status (embed mode/dim, db path, llm on/off)
  GET  /recall?q=&k=3       two-stage semantic recall, top-k
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
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import config, db, embed
from .recall import recall as do_recall
from .remember import remember as do_remember
from .forget import decay_all


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

            if url.path == "/health":
                self._send(200, {
                    "ok": True,
                    "service": "agent-memory-engine",
                    "embed_mode": embed.mode(),
                    "embed_dim": embed.dim(),
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
                results = do_recall(q, top_k=k, update=do_update)
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


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Engine HTTP server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    db.init_db(dim=embed.dim())
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 60)
    print("  Agent Memory Engine")
    print("=" * 60)
    print(f"  listening   : http://{args.host}:{args.port}")
    print(f"  embed mode  : {embed.mode()}")
    print(f"  embed dim   : {embed.dim()}")
    print(f"  db path     : {config.db_path()}")
    print(f"  llm enabled : {config.llm_enabled()}")
    print("=" * 60)
    print("  endpoints:")
    print("    GET  /health")
    print("    GET  /recall?q=...&k=3")
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
