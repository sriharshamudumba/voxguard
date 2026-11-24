"""OpenAI Realtime API (speech-to-speech) backend.

Server VAD is disabled: we append the whole caller turn, commit it manually,
and request a text-only response so the benchmark measures what the model
heard and put into tool calls, not audio synthesis.

schema="ga" targets the GA Realtime API (gpt-realtime, gpt-realtime-2).
schema="beta" targets the older preview event schema.
Event names follow OpenAI's published docs; verify against the current docs
if a run fails, since the API has changed between versions.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

from ..core.types import ToolCall
from .audio import resample, to_pcm16
from .base import AgentBackend, AgentResult, TurnInput


class OpenAIRealtimeBackend(AgentBackend):
    kind = "sts"

    def __init__(self, model: str = "gpt-realtime", api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", url: str = "wss://api.openai.com/v1/realtime",
                 schema: str = "ga", timeout: float = 60, name: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.url = url
        self.schema = schema
        self.timeout = timeout
        self.name = name or f"openai_realtime:{model}"

    def _session(self, turn: TurnInput) -> dict:
        tools = [{"type": "function", **t} for t in turn.tools]
        if self.schema == "beta":
            return {"modalities": ["text"], "instructions": turn.system_prompt,
                    "input_audio_format": "pcm16", "turn_detection": None,
                    "tools": tools, "tool_choice": "auto", "temperature": 0.6}
        return {"type": "realtime", "instructions": turn.system_prompt, "output_modalities": ["text"],
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}},
                "tools": tools, "tool_choice": "auto"}

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.schema == "beta":
            headers["OpenAI-Beta"] = "realtime=v1"
        pcm = to_pcm16(resample(turn.audio, turn.sr, 24000))
        calls: list[ToolCall] = []
        text: list[str] = []
        t0 = time.perf_counter()
        try:
            async with websockets.connect(f"{self.url}?model={self.model}", additional_headers=headers,
                                          max_size=None) as ws:
                await ws.send(json.dumps({"type": "session.update", "session": self._session(turn)}))
                chunk = 24000 * 2 // 10  # 100 ms
                for i in range(0, len(pcm), chunk):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                              "audio": base64.b64encode(pcm[i:i + chunk]).decode()}))
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                t_commit = time.perf_counter()
                await ws.send(json.dumps({"type": "response.create"}))
                first_event_ms = None
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.timeout))
                    et = msg.get("type", "")
                    if et == "error":
                        return AgentResult(calls, "".join(text), (time.perf_counter() - t0) * 1000,
                                           error=json.dumps(msg.get("error")))
                    if et in ("response.output_text.delta", "response.text.delta"):
                        first_event_ms = first_event_ms or (time.perf_counter() - t_commit) * 1000
                        text.append(msg.get("delta", ""))
                    elif et == "response.function_call_arguments.done":
                        first_event_ms = first_event_ms or (time.perf_counter() - t_commit) * 1000
                        try:
                            args = json.loads(msg.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {"_unparsed": msg.get("arguments")}
                        calls.append(ToolCall(msg.get("name", ""), args))
                    elif et == "response.done":
                        break
        except Exception as e:
            return AgentResult(calls, "".join(text), (time.perf_counter() - t0) * 1000, error=repr(e))
        total = (time.perf_counter() - t0) * 1000
        return AgentResult(calls, "".join(text), first_event_ms or total, {"total_ms": total})
