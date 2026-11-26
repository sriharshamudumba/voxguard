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
from .streaming import CallCollector, stream_turn

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
                 url: str = URL, timeout: float = 30, pace: bool = True, name: str | None = None,
                 listen_version: str | None = None, listen_options: dict | None = None):
        self.listen_model = listen_model
        # Flux models use the v2 listen API, which owns end-of-turn detection
        # (eot_threshold, eot_timeout_ms, eager_eot_threshold).
        self.listen_version = listen_version or ("v2" if listen_model.startswith("flux") else None)
        self.listen_options = dict(listen_options or {})
        self.think_provider = think_provider
        self.think_model = think_model
        self.speak_model = speak_model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.url = url
        self.timeout = timeout
        self.pace = pace
        self.name = name or f"deepgram_agent:{listen_model}+{think_model}"

    def listen_provider(self) -> dict:
        provider: dict = {"type": "deepgram", "model": self.listen_model}
        if self.listen_version:
            provider["version"] = self.listen_version
        else:
            provider["smart_format"] = False
        provider.update(self.listen_options)
        return provider

    def settings(self, turn: TurnInput) -> dict:
        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": 16000},
                "output": {"encoding": "linear16", "sample_rate": 16000, "container": "none"},
            },
            "agent": {
                "language": "en",
                "listen": {"provider": self.listen_provider()},
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
        col = CallCollector()
        text: list[str] = []
        t0 = time.perf_counter()
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
                    async for m in ws:
                        if not isinstance(m, str):
                            continue  # agent audio
                        j = json.loads(m)
                        if j.get("type") == "ConversationText" and j.get("role") == "assistant":
                            text.append(j.get("content", ""))
                        elif j.get("type") == "Error":
                            text.append("ERROR " + json.dumps(j))
                        if j.get("type") == "FunctionCallRequest":
                            for f, call in zip(j.get("functions", []), parse_function_calls(j)):
                                col.add(call)
                                # answer so the agent can keep listening and call again if needed
                                await ws.send(json.dumps({"type": "FunctionCallResponse", "id": f.get("id"),
                                                          "name": f.get("name"), "content": "recorded"}))

                rtask = asyncio.create_task(reader())
                await stream_turn(ws.send, turn.audio, turn.sr, 16000, col, pace=self.pace,
                                  max_silence_s=self.timeout / 3)
                rtask.cancel()
        except Exception as e:
            return AgentResult(col.merged(), " ".join(text), (time.perf_counter() - t0) * 1000, error=repr(e))
        st = col.stats()
        return AgentResult(col.merged(), " ".join(text), st["latency_ms"], st)
