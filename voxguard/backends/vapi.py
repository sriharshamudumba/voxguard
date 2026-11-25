"""Vapi backend over the WebSocket transport.

Needs an assistant created in the Vapi dashboard with a function tool named
`record_details` (no server URL needed). See docs/PROVIDERS.md.

Flow (docs.vapi.ai/calls/websocket-transport):
  1. POST /call {assistantId, transport: {provider: "vapi.websocket", audioFormat: pcm_s16le raw 16 kHz}}
     returns transport.websocketCallUrl
  2. stream binary PCM to that socket, then silence
  3. send {"type": "end-call"}, close
  4. GET /call/{id} and read the tool calls the assistant made from the call
     record (artifact.messages / messages entries with role "tool_calls")

Tool calls are read from the call record, not the socket, because the
transport socket is documented for audio and control messages only. The
record can take a few seconds to finalize, so it is polled.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import httpx

from ..core.types import ToolCall
from .base import AgentBackend, AgentResult, TurnInput
from .streaming import CallCollector, stream_turn

API = "https://api.vapi.ai"


def parse_call_record(call: dict) -> tuple[list[ToolCall], list[float]]:
    """Extract tool calls in order, and the seconds-from-start of each tool_calls message."""
    msgs = (call.get("artifact") or {}).get("messages") or call.get("messages") or []
    out: list[ToolCall] = []
    times: list[float] = []
    for m in msgs:
        if m.get("role") != "tool_calls":
            continue
        for tc in m.get("toolCalls") or m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            out.append(ToolCall(fn.get("name", ""), args))
        if m.get("secondsFromStart") is not None:
            times.append(float(m["secondsFromStart"]))
    return out, times


class VapiBackend(AgentBackend):
    kind = "platform"

    def __init__(self, assistant_id: str, api_key: str | None = None, api_key_env: str = "VAPI_API_KEY",
                 timeout: float = 30, pace: bool = True, wait_s: float = 6.0, name: str | None = None):
        self.assistant_id = assistant_id
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.pace = pace
        self.wait_s = wait_s   # silence after speech before ending the call
        self.name = name or f"vapi:{assistant_id[:8]}"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        hdr = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        speech_s = len(turn.audio) / turn.sr
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{API}/call", headers=hdr, json={
                    "assistantId": self.assistant_id,
                    "transport": {"provider": "vapi.websocket",
                                  "audioFormat": {"format": "pcm_s16le", "container": "raw", "sampleRate": 16000}},
                })
                r.raise_for_status()
                call = r.json()
                call_id = call["id"]
                ws_url = call["transport"]["websocketCallUrl"]

                async with websockets.connect(ws_url, max_size=None) as ws:
                    async def drain():
                        async for _ in ws:
                            pass  # agent audio and control messages; not needed
                    dtask = asyncio.create_task(drain())
                    # tool calls are read from the call record afterwards, so just stream and wait
                    await stream_turn(ws.send, turn.audio, turn.sr, 16000, CallCollector(), pace=self.pace,
                                      max_silence_s=self.wait_s)
                    await ws.send(json.dumps({"type": "end-call"}))
                    dtask.cancel()

                calls: list[ToolCall] = []
                times: list[float] = []
                for _ in range(10):   # call record finalizes asynchronously
                    await asyncio.sleep(1.5)
                    rec = (await c.get(f"{API}/call/{call_id}", headers=hdr)).json()
                    calls, times = parse_call_record(rec)
                    if calls or rec.get("status") == "ended":
                        break
        except Exception as e:
            return AgentResult([], latency_ms=(time.perf_counter() - t0) * 1000, error=repr(e))
        col = CallCollector()
        for call in calls:
            col.calls.append((0.0, call))
        merged = col.merged()
        # secondsFromStart is relative to call start; compare with speech duration
        cut_in = bool(times) and times[0] < speech_s
        lat = (times[-1] - speech_s) * 1000 if times and times[-1] >= speech_s else float("nan")
        return AgentResult(merged, "", lat, {"call_id": call_id, "cut_in": cut_in, "n_tool_calls": len(calls),
                                             "latency_source": "call_record"})
