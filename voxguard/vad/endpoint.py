"""Entity-aware endpointing as an optimal stopping problem.

At silence duration t with partial transcript x, model the probability the
caller has finished with an exponential hazard whose time constant depends on
the conversational state s(x):

    P(done | t, x) = 1 - exp(-t / tau[s(x)])

Commit the turn when the expected cost of waiting exceeds the cost of cutting in:

    P(done) * C_deadair  >  (1 - P(done)) * C_cutin[s]
    <=>  P(done) > p* = C_cutin / (C_cutin + C_deadair)

which gives a closed-form silence threshold per state:

    t*(s) = -tau[s] * ln(1 - p*(s))

Cutting in during a phone number corrupts an entity, so C_cutin is higher in
that state and the optimal wait gets longer automatically. Normal turns get a
shorter wait than a fixed threshold, so average latency also drops.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..numbers.spoken import extract_digit_runs

STATES = ("complete", "incomplete", "filler", "mid_number", "spelling")

_PHONE_CUES = re.compile(r"\b(phone|cell|mobile|reach|callback|call back)\b|\bnumber is\b", re.I)
# Words after which a turn almost never ends: the caller is about to say the entity.
_FILLERS = {"um", "uh", "er", "and", "so", "like", "the", "a", "my", "is", "to", "for", "at", "of",
            "under", "it's", "its", "name", "called", "am", "i'm", "this", "number", "order", "me"}
_NUM_WORDS = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
              "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
              "eighty", "ninety", "hundred", "double", "triple"}


def classify_state(partial: str, expected_digits: int = 10) -> str:
    """Map a partial transcript to a conversational state.

    Rule-based so it is transparent and fast. Swap in a semantic end-of-turn
    model (LiveKit turn detector, Pipecat smart-turn) by returning its states.
    """
    text = partial.strip()
    if not text:
        return "incomplete"
    toks = re.findall(r"[a-zA-Z']+|\d+", text.lower())
    if re.search(r"[.?!]$", text):
        return "complete"
    if len(toks) >= 2 and all(len(t) == 1 and t.isalpha() for t in toks[-2:]):
        return "spelling"
    if toks and (toks[-1] in _NUM_WORDS or toks[-1].isdigit()):
        if _PHONE_CUES.search(text):
            n = len("".join(extract_digit_runs(text)))
            return "mid_number" if n < expected_digits else "complete"
        return "mid_number"   # any number still being read out
    if toks and toks[-1] in _FILLERS:
        return "filler"
    return "complete" if len(toks) >= 3 else "incomplete"


@dataclass
class EntityAwareEndpointer:
    tau_ms: dict[str, float] = field(default_factory=lambda: {
        "complete": 280.0, "incomplete": 300.0, "filler": 450.0,
        "mid_number": 550.0, "spelling": 500.0,
    })
    c_deadair: float = 1.0
    c_cutin: dict[str, float] = field(default_factory=lambda: {
        "complete": 1.5, "incomplete": 3.0, "filler": 3.0, "mid_number": 8.0, "spelling": 8.0,
    })
    min_ms: float = 150.0
    max_ms: float = 2000.0

    def p_done(self, silence_ms: float, state: str) -> float:
        return 1 - math.exp(-silence_ms / self.tau_ms[state])

    def threshold_ms(self, state: str) -> float:
        c = self.c_cutin[state]
        p_star = c / (c + self.c_deadair)
        t = -self.tau_ms[state] * math.log(1 - p_star)
        return min(max(t, self.min_ms), self.max_ms)

    def should_commit(self, silence_ms: float, partial: str) -> bool:
        return silence_ms >= self.threshold_ms(classify_state(partial))


@dataclass
class FixedEndpointer:
    """Baseline used by most stacks: a single silence threshold."""
    ms: float = 600.0

    def threshold_ms(self, state: str) -> float:
        return self.ms

    def should_commit(self, silence_ms: float, partial: str) -> bool:
        return silence_ms >= self.ms


class EnergyVAD:
    """Minimal frame VAD for offline use. Swap in Silero VAD for production."""

    def __init__(self, frame_ms: int = 20, threshold_db: float = -40.0):
        self.frame_ms = frame_ms
        self.threshold_db = threshold_db

    def speech_frames(self, pcm, sr: int):
        import numpy as np
        n = int(sr * self.frame_ms / 1000)
        x = np.asarray(pcm, dtype=float)
        if x.size == 0:
            return np.zeros(0, bool)
        frames = x[: len(x) // n * n].reshape(-1, n)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms / (np.abs(x).max() + 1e-12) + 1e-12)
        return db > self.threshold_db
