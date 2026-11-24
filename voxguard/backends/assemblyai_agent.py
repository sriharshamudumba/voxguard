"""AssemblyAI Voice Agent API backend.

Protocol (assemblyai.com/docs/voice-agents/voice-agent-api/events-reference):
  connect wss://agents.assemblyai.com/v1/ws with "Authorization: Bearer <key>"
  send    session.update {session: {system_prompt, tools, input.format, output.format}}
  wait    session.ready, then stream input.audio {audio: base64 PCM16 mono 24 kHz}
  receive tool.call {call_id, name, arguments (object)}
Turn detection is server-side, so audio is followed by silence.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

from ..core.types import ToolCall
from .base import AgentBackend, AgentResult, TurnInput
from .streaming import stream_turn

URL = "wss://agents.assemblyai.com/v1/ws"


def parse_tool_call(msg: dict) -> ToolCall | None:
    if msg.get("type") != "tool.call":
        return None
    args = msg.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_unparsed": args}
    return ToolCall(msg.get("name", ""), args)


class AssemblyAIAgentBackend(AgentBackend):
    kind = "voice_agent_api"

    def __init__(self, api_key: str | None = None, api_key_env: str = "ASSEMBLYAI_API_KEY", url: str = URL,
                 voice: str | None = None, timeout: float = 30, pace: bool = True, name: str | None = None):
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.url = url
        self.voice = voice
        self.timeout = timeout
        self.pace = pace
        self.name = name or "assemblyai_agent"

    def session(self, turn: TurnInput) -> dict:
        s = {
            "system_prompt": turn.system_prompt,
            "input": {"format": {"encoding": "audio/pcm"}},
            "output": {"format": {"encoding": "audio/pcm"}},
            "tools": [{"type": "function", **t} for t in turn.tools],
        }
        if self.voice:
            s["output"]["voice"] = self.voice
        return {"type": "session.update", "session": s}

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        calls: list = []
        text: list[str] = []
        done = asyncio.Event()
        t0 = time.perf_counter()
        t_call = None
        try:
            async with websockets.connect(self.url, additional_headers={"Authorization": f"Bearer {self.api_key}"},
                                          max_size=None) as ws:
                await ws.send(json.dumps(self.session(turn)))
                while True:
                    j = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.timeout))
                    if j.get("type") == "session.ready":
                        break
                    if j.get("type") == "session.error":
                        return AgentResult([], error=json.dumps(j))

                async def reader():
                    nonlocal t_call
                    async for m in ws:
                        j = json.loads(m)
                        t = j.get("type")
                        if t == "transcript.agent":
                            text.append(j.get("text", ""))
                        elif t == "session.error":
                            text.append("ERROR " + json.dumps(j))
                            done.set()
                            return
                        call = parse_tool_call(j)
                        if call:
                            calls.append(call)
                            t_call = time.perf_counter()
                            done.set()
                            return

                async def send(chunk: bytes):
                    await ws.send(json.dumps({"type": "input.audio", "audio": base64.b64encode(chunk).decode()}))

                rtask = asyncio.create_task(reader())
                t_end = await stream_turn(send, turn.audio, turn.sr, 24000, done, pace=self.pace,
                                          max_silence_s=self.timeout / 3)
                try:
                    await asyncio.wait_for(rtask, timeout=5)
                except TimeoutError:
                    rtask.cancel()
                await ws.send(json.dumps({"type": "session.end"}))
        except Exception as e:
            return AgentResult(calls, "".join(text), (time.perf_counter() - t0) * 1000, error=repr(e))
        lat = (t_call - t_end) * 1000 if t_call else (time.perf_counter() - t0) * 1000
        return AgentResult(calls, " ".join(text), lat, {"turn_latency": True})
