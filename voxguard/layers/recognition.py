"""Layer 1: what did the platform's own recognizer hear?

Cascade platforms (Deepgram, AssemblyAI, ElevenLabs) pass a transcript to their LLM,
so that transcript is exactly what the LLM saw. Speech-to-speech models (OpenAI
Realtime, Gemini Live) hear audio directly; their transcript is a side channel from a
separate recognizer, so Layer 1 for them is an estimate, and rows say so.
"""
from __future__ import annotations

from ..names.phonetic import name_similarity
from ..numbers.spoken import extract_digit_runs, only_digits

NUMERIC = {"phone", "order_id", "card", "number"}
SOUND_ALIKE = 0.5   # name_similarity at or above this = same sound, different spelling


def heard_digits(heard: str, expected_len: int | None = None) -> str | None:
    """Digits in a transcript, joining runs split by pauses or by turn fragments ('|')."""
    runs = extract_digit_runs(heard.replace("|", " "))
    if not runs:
        return None
    cands = {"".join(runs[i:j]) for i in range(len(runs)) for j in range(i + 1, len(runs) + 1)}
    if expected_len:
        exact = [c for c in cands if len(c) == expected_len]
        if exact:
            return max(exact, key=lambda c: -abs(len(c) - expected_len))
    return max(cands, key=len)


def recognize_entity(field: str, expected: str, heard: str | None) -> str | None:
    """Layer 1 outcome for one field.

    numbers: 'correct' | 'wrong'
    names:   'correct' (spelled right) | 'sound_alike' (same sound, other spelling; no
             recognizer can resolve this from audio) | 'partial' (part of the name missing) | 'wrong'
    None when there is no transcript to judge.
    """
    if not heard or not heard.strip():
        return None
    if field in NUMERIC:
        d = only_digits(str(expected))
        return "correct" if d in (heard_digits(heard, len(d)) or "") or d in only_digits(heard) else "wrong"
    exp_tokens = str(expected).split()
    words = [w.strip(".,!?|").lower() for w in heard.split()]
    words = [w for w in words if w]
    if all(t.lower() in words for t in exp_tokens):
        return "correct"
    found = [max((name_similarity(w, t) for w in words), default=0.0) for t in exp_tokens]
    if all(f >= SOUND_ALIKE for f in found):
        return "sound_alike"
    if any(f >= SOUND_ALIKE for f in found):
        return "partial"
    return "wrong"
