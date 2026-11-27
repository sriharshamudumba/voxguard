"""Generate caller audio for benchmark cases, then pass it through a phone line.

TTS providers:
  openai   gpt-4o-mini-tts, with accent instructions per voice (API key needed)
  piper    local Piper TTS (pip install piper-tts, plus a voice model)
  espeak   espeak-ng CLI (robotic, but free and offline)

Synthetic accents are an approximation. Add real recordings (with consent and
a license that allows it) before drawing conclusions about specific accents.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from ..backends.audio import read_wav, telephony_channel, write_wav
from .dataset import Case

# accent key: (OpenAI voice, speaking instruction)
ACCENTS: dict[str, tuple[str, str]] = {
    "us": ("alloy", "Speak naturally at a normal phone pace with a General American accent."),
    "indian": ("coral", "Speak with a noticeable Indian English accent, at a natural pace."),
    "irish": ("sage", "Speak with an Irish accent, at a natural pace."),
    "scottish": ("ash", "Speak with a Scottish accent, at a natural pace."),
    "nigerian": ("verse", "Speak with a Nigerian English accent, at a natural pace."),
    "mexican": ("shimmer", "Speak with a Mexican Spanish accent in English, at a natural pace."),
    "chinese": ("nova", "Speak with a Mandarin Chinese accent in English, at a natural pace."),
    "arabic": ("onyx", "Speak with an Arabic accent in English, at a natural pace."),
    "british": ("fable", "Speak with a London British accent, at a natural pace."),
    "fast_casual": ("echo", "Speak quickly and casually, like on a busy street, slightly mumbling."),
}


def tts_openai(text: str, voice: str, instructions: str, api_key: str | None = None,
               model: str = "gpt-4o-mini-tts") -> bytes:
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    r = httpx.post("https://api.openai.com/v1/audio/speech", timeout=120,
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": model, "voice": voice, "input": text,
                         "instructions": instructions, "response_format": "wav"})
    r.raise_for_status()
    return r.content


def tts_piper(text: str, model_path: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["piper", "--model", model_path, "--output_file", f.name],
                       input=text.encode(), check=True, capture_output=True)
        return Path(f.name).read_bytes()


def tts_espeak(text: str, voice: str = "en-us") -> bytes:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak-ng not installed")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run([exe, "-v", voice, "-s", "150", "-w", f.name, text], check=True, capture_output=True)
        return Path(f.name).read_bytes()


def synthesize_cases(cases: list[Case], out_dir: str | Path, provider: str = "openai",
                     snr_db: float = 20.0, packet_loss: float = 0.01, piper_model: str | None = None,
                     api_key: str | None = None, overwrite: bool = False, workers: int = 8) -> list[Case]:
    """Synthesize every case that has no audio yet; resumable, and parallel for API providers."""
    from concurrent.futures import ThreadPoolExecutor

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cases):
        if provider == "openai" and not c.accent:
            c.accent = list(ACCENTS)[i % len(ACCENTS)]

    def one(c: Case) -> None:
        path = out / f"{c.id}.wav"
        if path.exists() and not overwrite:
            c.audio_path = str(path)
            return
        text = c.tts_text or c.caller_text
        if provider == "openai":
            voice, instr = ACCENTS[c.accent]
            raw = tts_openai(text, voice, instr, api_key)
        elif provider == "piper":
            raw = tts_piper(text, piper_model or os.environ.get("PIPER_MODEL", ""))
        elif provider == "espeak":
            raw = tts_espeak(text)
        else:
            raise ValueError(provider)
        x, sr = read_wav(raw)
        y, sr8 = telephony_channel(x, sr, snr_db=snr_db, packet_loss=packet_loss, seed=c.seed)
        write_wav(path, y, sr8)
        c.audio_path = str(path)

    n = workers if provider == "openai" else 1
    done = 0
    with ThreadPoolExecutor(n) as pool:
        for _ in pool.map(one, cases):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(cases)}", flush=True)
    return cases
