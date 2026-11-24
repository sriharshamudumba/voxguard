"""Deepgram Voice Agent API backend (STT + LLM + TTS behind one WebSocket).

Protocol (developers.deepgram.com/docs/voice-agent):
  1. send Settings (audio formats, agent.listen / agent.think / agent.speak,
     agent.think.prompt, agent.think.functions)
  2. wait for SettingsApplied
  3. stream binary linear16 audio
  4. receive FunctionCallRequest {functions: [{id, name, arguments (JSON string), client_side}]}

Functions without an `endpoint` are client-side, so the request comes back to us.
The think provider is configurable: the result reflects Deepgram's STT plus
whichever LLM you choose here.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from ..core.types import ToolCall
from .base import AgentBackend, AgentResult, TurnInput
from .streaming import stream_turn

URL = "wss://agent.deepgram.com/v1/agent/converse"


def parse_function_calls(msg: dict) -> list[ToolCall]:
    if msg.get("type") != "FunctionCallRequest":
        return []
    calls = []
    for f in msg.get("functions", []):
        raw = f.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            args = {"_unparsed": raw}
        calls.append(ToolCall(f.get("name", ""), args))
    return calls


class DeepgramAgentBackend(AgentBackend):
    kind = "voice_agent_api"

    def __init__(self, listen_model: str = "nova-3", think_provider: str = "open_ai",
                 think_model: str = "gpt-4o-mini", speak_model: str = "aura-2-thalia-en",
                 api_key: str | None = None, api_key_env: str = "DEEPGRAM_API_KEY",
                 url: str = URL, timeout: float = 30, pace: bool = True, name: str | None = None):
        self.listen_model = listen_model
        self.think_provider = think_provider
        self.think_model = think_model
        self.speak_model = speak_model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.url = url
        self.timeout = timeout
        self.pace = pace
        self.name = name or f"deepgram_agent:{listen_model}+{think_model}"

    def settings(self, turn: TurnInput) -> dict:
        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": 16000},
                "output": {"encoding": "linear16", "sample_rate": 16000, "container": "none"},
            },
            "agent": {
                "language": "en",
                "listen": {"provider": {"type": "deepgram", "model": self.listen_model, "smart_format": False}},
                "think": {
                    "provider": {"type": self.think_provider, "model": self.think_model, "temperature": 0},
                    "prompt": turn.system_prompt,
                    "functions": [{"name": t["name"], "description": t.get("description", ""),
                                   "parameters": t["parameters"]} for t in turn.tools],
                },
                "speak": {"provider": {"type": "deepgram", "model": self.speak_model}},
            },
        }

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        calls: list = []
        text: list[str] = []
        done = asyncio.Event()
        t0 = time.perf_counter()
        t_call = None
        try:
            async with websockets.connect(self.url, additional_headers={"Authorization": f"Token {self.api_key}"},
                                          max_size=None) as ws:
                await ws.send(json.dumps(self.settings(turn)))
                while True:
                    m = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                    if isinstance(m, str):
                        j = json.loads(m)
                        if j.get("type") == "SettingsApplied":
                            break
                        if j.get("type") == "Error":
                            return AgentResult([], error=json.dumps(j))

                async def reader():
                    nonlocal t_call
                    async for m in ws:
                        if not isinstance(m, str):
                            continue  # agent audio
                        j = json.loads(m)
                        if j.get("type") == "ConversationText" and j.get("role") == "assistant":
                            text.append(j.get("content", ""))
                        elif j.get("type") == "Error":
                            text.append("ERROR " + json.dumps(j))
                            done.set()
                            return
                        found = parse_function_calls(j)
                        if found:
                            calls.extend(found)
                            t_call = time.perf_counter()
                            done.set()
                            return

                rtask = asyncio.create_task(reader())
                t_end = await stream_turn(ws.send, turn.audio, turn.sr, 16000, done, pace=self.pace,
                                          max_silence_s=self.timeout / 3)
                try:
                    await asyncio.wait_for(rtask, timeout=5)
                except TimeoutError:
                    rtask.cancel()
        except Exception as e:
            return AgentResult(calls, "".join(text), (time.perf_counter() - t0) * 1000, error=repr(e))
        lat = (t_call - t_end) * 1000 if t_call else (time.perf_counter() - t0) * 1000
        return AgentResult(calls, " ".join(text), lat, {"turn_latency": True})
