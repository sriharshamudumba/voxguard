"""Local shadow recognizer with faster-whisper (free, runs on your GPU or CPU).

pip install 'voxguard[local]'
"""
from __future__ import annotations

import asyncio

from ..core.types import ShadowTranscript
from .audio import resample
from .base import ShadowASR, TurnInput


class FasterWhisperSTT(ShadowASR):
    def __init__(self, model: str = "small.en", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"faster_whisper:{model}"

    def _run(self, audio, keyterms):
        prompt = ", ".join(keyterms) if keyterms else None
        segs, _ = self.model.transcribe(audio, language="en", beam_size=5, word_timestamps=True,
                                        initial_prompt=prompt, vad_filter=False)
        text, probs = [], []
        for s in segs:
            text.append(s.text)
            probs.extend(w.probability for w in (s.words or []))
        conf = sum(probs) / len(probs) if probs else None
        return ShadowTranscript("".join(text).strip(), conf, probs or None)

    async def transcribe(self, turn: TurnInput, keyterms: list[str] | None = None) -> ShadowTranscript:
        audio = resample(turn.audio, turn.sr, 16000)
        return await asyncio.to_thread(self._run, audio, keyterms)

