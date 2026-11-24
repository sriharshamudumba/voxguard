"""Offline endpointing benchmark.

Builds word timelines from benchmark cases with realistic pauses (short gaps
between words, long pauses between digit groups), then replays them against an
endpointer. At every pause the endpointer sees the partial transcript so far.

  cut_in   the endpointer fired during a pause inside the caller's turn
  latency  silence the caller waits after actually finishing (if no cut-in)

No audio or API keys needed, so the math can be checked and tuned quickly.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ..vad.endpoint import _NUM_WORDS as _NUMS
from .dataset import Case


def timeline(case: Case, rng: random.Random) -> list[tuple[str, float]]:
    """[(word, pause_after_ms)] where the last pause is the true end of turn."""
    out = []
    words = case.caller_text.replace(",", " ,").replace(".", " .").split()
    clean = [w for w in words if w not in {",", "."}]
    idx = 0
    for i, w in enumerate(words):
        if w in {",", "."}:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt == "," and (w.lower() in _NUMS):
            pause = max(250.0, rng.gauss(700, 220))       # pause between digit groups
        elif nxt == ",":
            pause = max(80.0, rng.gauss(250, 80))         # ordinary comma pause
        elif w.lower().strip(",") in {"is", "it's", "at", "under"}:
            pause = max(120.0, rng.gauss(420, 150))       # hesitation before the entity
        else:
            pause = max(40.0, rng.gauss(140, 50))
        idx += 1
        out.append((w, pause if idx < len(clean) else 0.0))
    return out


def simulate(cases: list[Case], endpointers: dict, seed: int = 0) -> pd.DataFrame:
    rows = []
    for c in cases:
        rng = random.Random(f"{seed}-{c.id}")
        tl = timeline(c, rng)
        for name, ep in endpointers.items():
            partial, cut = [], False
            for w, pause in tl[:-1]:
                partial.append(w)
                if pause >= ep.threshold_ms(_state(ep, " ".join(partial))):
                    cut = True
                    break
            full = " ".join(w for w, _ in tl)
            lat = np.nan if cut else ep.threshold_ms(_state(ep, full))
            rows.append({"case_id": c.id, "kind": c.kind, "endpointer": name, "cut_in": cut, "latency_ms": lat})
    return pd.DataFrame(rows)


def _state(ep, partial: str) -> str:
    from ..vad.endpoint import classify_state
    return classify_state(partial)


def summarize_vad(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["endpointer", "kind"])
    s = g.agg(n=("cut_in", "size"), cut_in_rate=("cut_in", "mean"),
              latency_p50=("latency_ms", "median"),
              latency_p95=("latency_ms", lambda x: x.quantile(0.95))).reset_index()
    a = df.groupby("endpointer").agg(n=("cut_in", "size"), cut_in_rate=("cut_in", "mean"),
                                     latency_p50=("latency_ms", "median"),
                                     latency_p95=("latency_ms", lambda x: x.quantile(0.95))).reset_index()
    a["kind"] = "all"
    return pd.concat([s, a], ignore_index=True)
