"""`voxguard serve`: voxguard as a small HTTP service, for hosted agents (Vapi, Retell, any webhook).

Your tool endpoint (the server the platform calls when the LLM uses a tool) asks voxguard first:

    POST /check   {"call_id": "...", "tool": "record_details", "args": {...},
                   "transcript": "caller words this turn",            (optional but recommended)
                   "roster": [...], "known": {"order_id": [...]}}      (optional, per call)
    ->            {"action": "pass|correct|confirm", "args": {...}, "say": "...", "question": "..."}

    POST /end_turn {"call_id": "..."}      after the agent replies
    POST /end_call {"call_id": "..."}      when the call ends (frees memory)
    GET  /health

If action is "confirm", return `say` to the platform as the tool result instead of running the tool.
Standard library only; no audio over HTTP (the transcript is the evidence).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sdk import Guard, decision_dict


def make_handler(guard: Guard, token: str | None):
    calls: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):   # keep caller data out of the console
            pass

        def do_GET(self):
            self._send(200, {"ok": True, "calls": len(calls)}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                return self._send(401, {"error": "unauthorized"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            cid = str(body.get("call_id") or "default")
            if self.path == "/check":
                call = calls.get(cid) or calls.setdefault(cid, guard.new_call(cid))
                if body.get("roster") is not None or body.get("known") is not None:
                    # per-call lists (today's bookings, this caller's orders) override the defaults
                    g = Guard(guard.fields, body.get("roster", guard.roster), body.get("known", guard.known),
                              None, guard.transcript_source, str(guard.log_path) if guard.log_path else None,
                              guard.redact)
                    call.guard = g
                if body.get("transcript"):
                    call.add_transcript(body["transcript"])
                d = call.check(str(body.get("tool", "")), dict(body.get("args") or {}))
                return self._send(200, decision_dict(d))
            if self.path == "/end_turn":
                if cid in calls:
                    calls[cid].end_turn()
                return self._send(200, {"ok": True})
            if self.path == "/end_call":
                calls.pop(cid, None)
                return self._send(200, {"ok": True})
            self._send(404, {"error": "unknown path"})

    return Handler


def serve(guard: Guard, host: str = "127.0.0.1", port: int = 8765, token: str | None = None) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(guard, token))
    print(f"voxguard listening on http://{host}:{port}  (POST /check, /end_turn, /end_call)", flush=True)
    srv.serve_forever()
