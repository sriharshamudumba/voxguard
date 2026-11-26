"""Google Gemini Live API (speech-to-speech) backend.

Automatic activity detection is disabled; we mark the turn with
activityStart / activityEnd ourselves. Field names follow the public
BidiGenerateContent docs; verify against the current docs if a run fails.
Native-audio models only answer in AUDIO; set response_modality="AUDIO" for
those. Tool calls are reported the same way in both modes.
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

URL = ("wss://generativelanguage.googleapis.com/ws/"
       "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")


class GeminiLiveBackend(AgentBackend):
    kind = "sts"

    def __init__(self, model: str = "gemini-3.8-live", api_key: str | None = None,
                 api_key_env: str = "GEMINI_API_KEY", response_modality: str = "AUDIO",
                 timeout: float = 60, name: str | None = None):
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.response_modality = response_modality
        self.timeout = timeout
        self.name = name or f"gemini_live:{model}"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        import websockets
        pcm = to_pcm16(resample(turn.audio, turn.sr, 16000))
        calls: list[ToolCall] = []
        text: list[str] = []
        t0 = time.perf_counter()
        setup = {"setup": {
            "model": self.model,
            "generationConfig": {"responseModalities": [self.response_modality]},
            "systemInstruction": {"parts": [{"text": turn.system_prompt}]},
            "tools": [{"functionDeclarations": turn.tools}],
            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
            # text of what the model says, so a spoken question ("could you spell that?")
            # is scored as the agent asking, not as a missing value
            "outputAudioTranscription": {},
        }}
        try:
            async with websockets.connect(f"{URL}?key={self.api_key}", max_size=None) as ws:
                await ws.send(json.dumps(setup))
                await asyncio.wait_for(ws.recv(), timeout=self.timeout)  # setupComplete
                await ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
                chunk = 16000 * 2 // 10
                for i in range(0, len(pcm), chunk):
                    await ws.send(json.dumps({"realtimeInput": {"audio": {
                        "data": base64.b64encode(pcm[i:i + chunk]).decode(),
                        "mimeType": "audio/pcm;rate=16000"}}}))
                await ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
                t_end = time.perf_counter()
                first = None
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.timeout))
                    if "toolCall" in msg:
                        first = first or (time.perf_counter() - t_end) * 1000
                        for fc in msg["toolCall"].get("functionCalls", []):
                            calls.append(ToolCall(fc.get("name", ""), fc.get("args") or {}))
                        break  # the model waits for a toolResponse; one call is all we score
                    sc = msg.get("serverContent") or {}
                    spoken = (sc.get("outputTranscription") or {}).get("text")
                    if spoken:
                        first = first or (time.perf_counter() - t_end) * 1000
                        text.append(spoken)
                    for part in (sc.get("modelTurn") or {}).get("parts", []):
                        if "text" in part:
                            first = first or (time.perf_counter() - t_end) * 1000
                            text.append(part["text"])
                    if sc.get("turnComplete"):
                        break
        except Exception as e:
            return AgentResult(calls, "".join(text), (time.perf_counter() - t0) * 1000, error=repr(e))
        total = (time.perf_counter() - t0) * 1000
        return AgentResult(calls, "".join(text), first or total, {"total_ms": total, "n_tool_calls": len(calls)})
