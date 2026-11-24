"""Simulated agent and shadow recognizer.

Used for tests, CI, and a no-API-key demo. They read the ground truth and inject
errors at configurable rates that mimic what is reported for real systems:
digit substitutions on confusable pairs, dropped digits, and speech-to-speech
padding (a dropped digit replaced by an invented one). Results from these
backends are SIMULATED and must never be reported as real model results.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from ..bench.names_data import FIRST_TIME_NAMES, NAMES
from ..core.types import ShadowTranscript, ToolCall
from ..numbers.channel import _CONFUSABLE
from .base import AgentBackend, AgentResult, ShadowASR, TurnInput

_PAIR = {}
for a, b in _CONFUSABLE:
    _PAIR.setdefault(str(a), []).append(str(b))
    _PAIR.setdefault(str(b), []).append(str(a))

_ALL_NAMES = {**NAMES, **FIRST_TIME_NAMES}

# Fraction of errors driven by the audio itself, shared by every recognizer that hears it.
# Real recognizers make correlated mistakes on the same bad audio, which limits what a
# second opinion can catch. Independent errors would overstate the plugin's benefit.
ACOUSTIC_CORRELATION = 0.4

PROFILES = {
    # rates are per entity
    "sts_typical": dict(digit_sub=0.07, digit_drop=0.02, digit_pad=0.05, name_sub=0.18, latency=(900, 250)),
    "cascade_typical": dict(digit_sub=0.05, digit_drop=0.03, digit_pad=0.0, name_sub=0.14, latency=(1300, 300)),
    "shadow_typical": dict(digit_sub=0.035, digit_drop=0.015, digit_pad=0.0, name_sub=0.10, latency=(250, 60)),
}


def _rng(seed: int, field: str, own: str) -> random.Random:
    """With prob ACOUSTIC_CORRELATION use the shared acoustic stream, else the recognizer's own."""
    gate = random.Random(f"gate-{seed}-{field}-{own}")
    if gate.random() < ACOUSTIC_CORRELATION:
        return random.Random(f"acoustic-{seed}-{field}")
    return random.Random(f"{own}-{seed}-{field}")


def _corrupt_digits(d: str, rng: random.Random, p: dict) -> str:
    s = list(d)
    r = rng.random()
    if r < p["digit_sub"] and s:
        i = rng.randrange(len(s))
        s[i] = rng.choice(_PAIR.get(s[i], [str(rng.randrange(10))]))
    elif r < p["digit_sub"] + p["digit_drop"] and len(s) > 1:
        s.pop(rng.randrange(len(s)))
    elif r < p["digit_sub"] + p["digit_drop"] + p["digit_pad"] and len(s) > 1:
        i = rng.randrange(len(s))
        s.pop(i)
        s.insert(rng.randrange(len(s) + 1), rng.choice(s))  # invents a digit it already "saw"
    return "".join(s)


def _corrupt_name(name: str, rng: random.Random, p: dict) -> str:
    if rng.random() < p["name_sub"]:
        variants = _ALL_NAMES.get(name, ([], None))[0]
        if variants:
            return rng.choice(variants)
    return name


def _spoken_digits(d: str) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return " ".join(words[int(c)] for c in d)


@dataclass
class MockAgent(AgentBackend):
    profile: str = "sts_typical"
    name: str = "mock:sts_typical"
    kind: str = "mock"
    seed_offset: int = 0

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        p = PROFILES[self.profile]
        args = {}
        for k, v in turn.truth.items():
            rng = _rng(turn.seed, k, f"agent-{self.profile}-{self.seed_offset}")
            args[k] = _corrupt_name(v, rng, p) if k == "name" else _corrupt_digits(str(v), rng, p)
        lat = max(150.0, random.Random(f"lat-{turn.seed}-{self.profile}").gauss(*p["latency"]))
        await asyncio.sleep(0)
        return AgentResult([ToolCall("record_details", args)], "", lat, {"simulated": True})


@dataclass
class MockShadow(ShadowASR):
    profile: str = "shadow_typical"
    name: str = "mock:shadow"

    async def transcribe(self, turn: TurnInput, keyterms: list[str] | None = None) -> ShadowTranscript:
        p = dict(PROFILES[self.profile])
        parts = []
        for k, v in turn.truth.items():
            rng = _rng(turn.seed, k, "shadow")
            if k == "name":
                pk = dict(p)
                if keyterms and v in keyterms:
                    pk["name_sub"] = p["name_sub"] * 0.3  # biasing toward known names helps
                parts.append(f"my name is {_corrupt_name(v, rng, pk)}")
            else:
                parts.append(f"my number is {_spoken_digits(_corrupt_digits(str(v), rng, p))}")
        return ShadowTranscript(". ".join(parts), 0.9)
