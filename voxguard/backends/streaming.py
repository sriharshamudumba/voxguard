"""Shared helpers for platforms that detect the end of the caller's turn themselves.

Deepgram Voice Agent, AssemblyAI Voice Agent, ElevenLabs Agents, and Vapi all run
server-side turn detection, so we cannot commit the turn manually. Instead we
stream the caller audio at real-time pace, then keep streaming silence until
the agent calls a tool or a timeout passes. Latency is measured from the end
of the caller's speech to the tool call, which is the turn latency a caller
actually experiences (it includes the platform's endpointing wait).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import numpy as np

from .audio import resample, to_pcm16


def pcm_chunks(audio: np.ndarray, sr_in: int, sr_out: int, chunk_ms: int = 100) -> list[bytes]:
    pcm = to_pcm16(resample(audio, sr_in, sr_out))
    step = sr_out * 2 * chunk_ms // 1000
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


def silence_chunk(sr: int, chunk_ms: int = 100) -> bytes:
    return b"\x00\x00" * (sr * chunk_ms // 1000)


async def stream_turn(
    send: Callable[[bytes], Awaitable[None]],
    audio: np.ndarray,
    sr_in: int,
    sr_out: int,
    done: asyncio.Event,
    chunk_ms: int = 100,
    pace: bool = True,
    max_silence_s: float = 8.0,
) -> float:
    """Send speech then silence until `done` is set. Returns perf_counter at end of speech."""
    for chunk in pcm_chunks(audio, sr_in, sr_out, chunk_ms):
        await send(chunk)
        if pace:
            await asyncio.sleep(chunk_ms / 1000)
    t_end = time.perf_counter()
    silence = silence_chunk(sr_out, chunk_ms)
    waited = 0.0
    while not done.is_set() and waited < max_silence_s:
        await send(silence)
        await asyncio.sleep(chunk_ms / 1000)
        waited += chunk_ms / 1000
    return t_end
