"""Backends for any OpenAI-compatible HTTP API.

That covers OpenAI, Groq, Together, Fireworks, OpenRouter, and self-hosted
open models served by vLLM / vLLM-Omni / SGLang / Ollama.

  CascadeBackend    STT (any /audio/transcriptions) -> LLM (any /chat/completions with tools)
  AudioLLMBackend   audio goes straight into /chat/completions as input_audio
                    (gpt-4o-audio, or open speech LLMs such as Ultravox and
                    Qwen-Omni served by vLLM)
  OpenAICompatSTT   also usable as the shadow recognizer
"""
from __future__ import annotations

import base64
import json
import math
import os
import time

import httpx

from ..core.types import ShadowTranscript, ToolCall
from .audio import resample, wav_bytes
from .base import AgentBackend, AgentResult, ShadowASR, TurnInput


def _key(api_key: str | None, env: str) -> str:
    return api_key or os.environ.get(env, "")


def _parse_tool_calls(msg: dict) -> list[ToolCall]:
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": fn.get("arguments")}
        calls.append(ToolCall(fn.get("name", ""), args))
    return calls


class OpenAICompatSTT(ShadowASR):
    def __init__(self, base_url: str = "https://api.openai.com/v1", model: str = "whisper-1",
                 api_key: str | None = None, api_key_env: str = "OPENAI_API_KEY", timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = _key(api_key, api_key_env)
        self.timeout = timeout
        self.name = f"stt:{model}"

    async def transcribe(self, turn: TurnInput, keyterms: list[str] | None = None) -> ShadowTranscript:
        audio = resample(turn.audio, turn.sr, 16000)
        data = {"model": self.model, "response_format": "verbose_json", "language": "en"}
        if keyterms:
            data["prompt"] = ", ".join(keyterms)[:800]
        files = {"file": ("turn.wav", wav_bytes(audio, 16000), "audio/wav")}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/audio/transcriptions",
                             headers={"Authorization": f"Bearer {self.api_key}"}, data=data, files=files)
            if r.status_code == 400 and "verbose_json" in r.text:
                data["response_format"] = "json"
                r = await c.post(f"{self.base_url}/audio/transcriptions",
                                 headers={"Authorization": f"Bearer {self.api_key}"}, data=data, files=files)
            r.raise_for_status()
            j = r.json()
        conf = None
        segs = j.get("segments") or []
        if segs and "avg_logprob" in segs[0]:
            conf = math.exp(sum(s["avg_logprob"] for s in segs) / len(segs))
        return ShadowTranscript(j.get("text", ""), conf)


class _ChatLLM:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, temperature: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]


class CascadeBackend(AgentBackend):
    kind = "cascade"

    def __init__(self, stt: ShadowASR, llm_base_url: str, llm_model: str, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 60, temperature: float = 0.0,
                 name: str | None = None):
        self.stt = stt
        self.llm = _ChatLLM(llm_base_url, llm_model, _key(api_key, api_key_env), timeout, temperature)
        self.name = name or f"cascade:{stt.name}+{llm_model}"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        t0 = time.perf_counter()
        try:
            tr = await self.stt.transcribe(turn)
            msg = await self.llm.complete(
                [{"role": "system", "content": turn.system_prompt}, {"role": "user", "content": tr.text}],
                turn.tools)
        except Exception as e:  # report, do not crash the benchmark
            return AgentResult([], latency_ms=(time.perf_counter() - t0) * 1000, error=repr(e))
        return AgentResult(_parse_tool_calls(msg), msg.get("content") or "",
                           (time.perf_counter() - t0) * 1000, {"transcript": tr.text})


class AudioLLMBackend(AgentBackend):
    kind = "audio_llm"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 90, temperature: float = 0.0,
                 sample_rate: int = 16000, name: str | None = None):
        self.llm = _ChatLLM(base_url, model, _key(api_key, api_key_env), timeout, temperature)
        self.sample_rate = sample_rate
        self.name = name or f"audio_llm:{model}"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        t0 = time.perf_counter()
        audio = resample(turn.audio, turn.sr, self.sample_rate)
        b64 = base64.b64encode(wav_bytes(audio, self.sample_rate)).decode()
        messages = [
            {"role": "system", "content": turn.system_prompt},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]},
        ]
        try:
            msg = await self.llm.complete(messages, turn.tools)
        except Exception as e:
            return AgentResult([], latency_ms=(time.perf_counter() - t0) * 1000, error=repr(e))
        return AgentResult(_parse_tool_calls(msg), msg.get("content") or "", (time.perf_counter() - t0) * 1000)
