"""Deepgram pre-recorded STT, usable as a shadow recognizer or as the STT stage of a cascade.

Keyterm prompting (Nova-3) is passed through when the plugin supplies keyterms,
so per-call names and menu items bias recognition.
"""
from __future__ import annotations

import os

import httpx

from ..core.types import ShadowTranscript
from .audio import resample, wav_bytes
from .base import ShadowASR, TurnInput


class DeepgramSTT(ShadowASR):
    def __init__(self, model: str = "nova-3", api_key: str | None = None,
                 api_key_env: str = "DEEPGRAM_API_KEY", timeout: float = 60):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.name = f"deepgram:{model}"

    async def transcribe(self, turn: TurnInput, keyterms: list[str] | None = None) -> ShadowTranscript:
        audio = resample(turn.audio, turn.sr, 16000)
        params: list[tuple[str, str]] = [("model", self.model), ("smart_format", "false"), ("numerals", "false")]
        for k in (keyterms or [])[:100]:
            params.append(("keyterm", k))
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post("https://api.deepgram.com/v1/listen", params=params,
                             headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
                             content=wav_bytes(audio, 16000))
            r.raise_for_status()
            alt = r.json()["results"]["channels"][0]["alternatives"][0]
        words = alt.get("words") or []
        return ShadowTranscript(alt.get("transcript", ""), alt.get("confidence"),
                                [w.get("confidence", 0.0) for w in words] or None)
