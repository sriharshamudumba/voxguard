"""Shared helpers for platforms that detect the end of the caller's turn themselves.

Deepgram Voice Agent, AssemblyAI Voice Agent, ElevenLabs Agents, and Vapi run
server-side turn detection, so we cannot commit the turn manually. We stream the
caller audio at real-time pace, then silence, and collect EVERY tool call the
agent makes along the way.

Why every call: these platforms often end the turn early, during the pauses
between digit groups, and call the tool with a partial value ("888272"), then
call it again once the caller finishes. Scoring only the first call would
measure the platform's endpointing, not its final answer. We record both:

  cut_in      the first tool call arrived before the caller finished speaking
  final args  merged across calls, later non-empty values overwrite earlier ones
  latency     end of caller speech to the LAST tool call (NaN if every call was a cut-in)
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from ..core.types import ToolCall
from .audio import resample, to_pcm16


def pcm_chunks(audio: np.ndarray, sr_in: int, sr_out: int, chunk_ms: int = 100) -> list[bytes]:
    pcm = to_pcm16(resample(audio, sr_in, sr_out))
    step = sr_out * 2 * chunk_ms // 1000
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


def silence_chunk(sr: int, chunk_ms: int = 100) -> bytes:
    return b"\x00\x00" * (sr * chunk_ms // 1000)


@dataclass
class CallCollector:
    """Collects tool calls with arrival times relative to the end of caller speech."""
    calls: list[tuple[float, ToolCall]] = field(default_factory=list)
    t_speech_end: float | None = None
    got_post_speech_call: asyncio.Event = field(default_factory=asyncio.Event)

    def add(self, call: ToolCall) -> None:
        t = time.perf_counter()
        self.calls.append((t, call))
        if self.t_speech_end is not None:
            self.got_post_speech_call.set()

    def speech_ended(self) -> None:
        self.t_speech_end = time.perf_counter()

    def merged(self) -> list[ToolCall]:
        if not self.calls:
            return []
        name = self.calls[-1][1].name
        args: dict = {}
        for _, c in self.calls:
            for k, v in c.args.items():
                if v not in (None, "", [], {}):
                    args[k] = v
        return [ToolCall(name, args)]

    def stats(self) -> dict:
        if not self.calls or self.t_speech_end is None:
            return {"cut_in": False, "n_tool_calls": len(self.calls), "latency_ms": math.nan}
        first, last = self.calls[0][0], self.calls[-1][0]
        cut_in = first < self.t_speech_end
        lat = (last - self.t_speech_end) * 1000 if last >= self.t_speech_end else math.nan
        return {"cut_in": cut_in, "n_tool_calls": len(self.calls), "latency_ms": lat}


async def stream_turn(
    send: Callable[[bytes], Awaitable[None]],
    audio: np.ndarray,
    sr_in: int,
    sr_out: int,
    collector: CallCollector,
    chunk_ms: int = 100,
    pace: bool = True,
    max_silence_s: float = 8.0,
    grace_s: float = 3.0,
) -> None:
    """Send all speech, then silence until the agent settles.

    Stops when a tool call arrives after speech ended, or when the agent already
    called during speech and `grace_s` of silence passed with no further call,
    or after `max_silence_s`.
    """
    for chunk in pcm_chunks(audio, sr_in, sr_out, chunk_ms):
        await send(chunk)
        if pace:
            await asyncio.sleep(chunk_ms / 1000)
    collector.speech_ended()
    silence = silence_chunk(sr_out, chunk_ms)
    waited = 0.0
    while waited < max_silence_s and not collector.got_post_speech_call.is_set():
        if collector.calls and waited >= grace_s:
            break
        await send(silence)
        await asyncio.sleep(chunk_ms / 1000)
        waited += chunk_ms / 1000
    if collector.got_post_speech_call.is_set():
        # short settle window in case the agent splits fields across calls
        for _ in range(int(1.0 * 1000 / chunk_ms)):
            await send(silence)
            await asyncio.sleep(chunk_ms / 1000)
