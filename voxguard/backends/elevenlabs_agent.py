"""ElevenLabs Agents backend.

Needs an agent created in the ElevenLabs dashboard with a CLIENT tool named
`record_details` (parameters: name, phone, order_id, all strings). See
docs/PROVIDERS.md. The system prompt can be overridden per session if
"prompt" overrides are enabled in the agent's security settings.

Protocol (elevenlabs.io/docs/eleven-agents/libraries/web-sockets):
  private agents: GET /v1/convai/conversation/get-signed-url?agent_id=... (xi-api-key)
  connect, send conversation_initiation_client_data
  receive conversation_initiation_metadata (gives user_input_audio_format, e.g. pcm_16000)
  send {"user_audio_chunk": base64 PCM16}; answer {"type":"ping"} with {"type":"pong","event_id":...}
  receive {"type":"client_tool_call","client_tool_call":{tool_name, tool_call_id, parameters}}
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import httpx

from ..core.types import ToolCall
from .base import AgentBackend, AgentResult, TurnInput
from .streaming import CallCollector, stream_turn

WS = "wss://api.elevenlabs.io/v1/convai/conversation"


def parse_client_tool_call(msg: dict) -> ToolCall | None:
    if msg.get("type") != "client_tool_call":
        return None
    c = msg.get("client_tool_call") or {}
    return ToolCall(c.get("tool_name", ""), c.get("parameters") or {})


def input_rate(meta: dict) -> int:
    fmt = (meta.get("conversation_initiation_metadata_event") or {}).get("user_input_audio_format", "pcm_16000")
    try:
        return int(str(fmt).split("_")[-1])
    except ValueError:
        return 16000


class ElevenLabsAgentBackend(AgentBackend):
    kind = "platform"
    heard_source = "pipeline"

    def __init__(self, agent_id: str, api_key: str | None = None, api_key_env: str = "ELEVENLABS_API_KEY",
                 override_prompt: bool = False, timeout: float = 30, pace: bool = True, name: str | None = None):
        self.agent_id = agent_id
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.override_prompt = override_prompt
        self.timeout = timeout
        self.pace = pace
        self.name = name or f"elevenlabs_agent:{agent_id[:8]}"

    async def _url(self) -> str:
        if not self.api_key:
            return f"{WS}?agent_id={self.agent_id}"  # public agent
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get("https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
                            params={"agent_id": self.agent_id}, headers={"xi-api-key": self.api_key})
            r.raise_for_status()
            return r.json()["signed_url"]

    def init_message(self, turn: TurnInput) -> dict:
        msg: dict = {"type": "conversation_initiation_client_data"}
        if self.override_prompt:
            msg["conversation_config_override"] = {"agent": {"prompt": {"prompt": turn.system_prompt},
                                                             "first_message": ""}}
        return msg

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        col = CallCollector()
        text: list[str] = []
        heard: list[str] = []
        t0 = time.perf_counter()
        try:
            url = await self._url()
            async with websockets.connect(url, max_size=None) as ws:
                await ws.send(json.dumps(self.init_message(turn)))
                rate = 16000
                while True:
                    j = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.timeout))
                    if j.get("type") == "conversation_initiation_metadata":
                        rate = input_rate(j)
                        break
                    if j.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong", "event_id": j.get("ping_event", {}).get("event_id")}))

                async def reader():
                    async for m in ws:
                        j = json.loads(m)
                        t = j.get("type")
                        if t == "ping":
                            await ws.send(json.dumps({"type": "pong",
                                                      "event_id": j.get("ping_event", {}).get("event_id")}))
                        elif t == "user_transcript":
                            heard.append((j.get("user_transcription_event") or {}).get("user_transcript", ""))
                        elif t == "agent_response":
                            text.append((j.get("agent_response_event") or {}).get("agent_response", ""))
                        call = parse_client_tool_call(j)
                        if call:
                            col.add(call)
                            tid = (j.get("client_tool_call") or {}).get("tool_call_id")
                            await ws.send(json.dumps({"type": "client_tool_result", "tool_call_id": tid,
                                                      "result": "recorded", "is_error": False}))

                async def send(chunk: bytes):
                    await ws.send(json.dumps({"user_audio_chunk": base64.b64encode(chunk).decode()}))

                rtask = asyncio.create_task(reader())
                await stream_turn(send, turn.audio, turn.sr, rate, col, pace=self.pace,
                                  max_silence_s=self.timeout / 3)
                rtask.cancel()
        except Exception as e:
            return AgentResult(col.merged(), " ".join(text), (time.perf_counter() - t0) * 1000,
                               {"heard": heard}, error=repr(e))
        st = dict(col.stats(), heard=heard)
        return AgentResult(col.merged(), " ".join(text), st["latency_ms"], st)
