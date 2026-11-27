"""Layer 3 benchmark: replay every confirmation in a saved run as a real follow-up turn.

For each row where voxguard asked the caller (action == confirm), and for both repair
strategies (targeted, full):
  1. plan the question from the tool value and the evidence only (never the truth)
  2. simulate the caller's answer: the asked digits or letters, or "yes" if the
     read-back part was right
  3. speak that answer with the same accent voice, pass it through the phone-line
     simulation, and recognize it with the second recognizer
  4. merge it into the value and score the final value

Modes:
  tts   OpenAI TTS + phone line + the configured recognizer (realistic; costs API calls)
  text  the recognizer is skipped and the answer is taken as said (an upper bound)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from ..backends.audio import read_wav, telephony_channel
from ..backends.base import ShadowASR, TurnInput
from ..layers.recognition import heard_digits
from ..layers.recovery import WORDS_PER_SECOND, apply_reply, plan_repair
from ..numbers.validate import expected_length
from .metrics import field_kind, norm

STRATEGIES = ("targeted", "full")


def _evidence(row: dict, kind: str) -> str | None:
    """What voxguard compares the tool value against: the agent's own transcript when it is
    the LLM's input, else the second recognizer."""
    texts = []
    if row.get("heard_source") == "pipeline" and row.get("agent_heard"):
        texts.append(row["agent_heard"])
    if row.get("shadow_text"):
        texts.append(row["shadow_text"])
    if kind in {"phone", "order_id", "card", "number"}:
        for t in texts:
            d = heard_digits(t, expected_length(kind))
            if d:
                return d
        return None
    model = str(row.get("before_value") or "")
    if not texts or not model:
        return texts[0] if texts else None
    from ..names.plugin import NameGuard
    return NameGuard()._best_span(texts[0], model, []) or None


class Recognizer:
    """Speak a reply in the caller's accent and recognize it. Cached by (accent, text)."""

    def __init__(self, shadow: ShadowASR | None, mode: str = "tts"):
        self.shadow, self.mode, self.cache = shadow, mode, {}

    async def __call__(self, text: str, accent: str, seed: int) -> tuple[str, float]:
        key = (accent, text)
        if key in self.cache:
            return self.cache[key]
        if self.mode == "text" or self.shadow is None or text.lower().startswith("yes"):
            out = (text, len(text.split()) / WORDS_PER_SECOND)
        elif type(self.shadow).__name__ == "MockShadow":   # simulated recognizer: no audio needed
            from ..layers.recovery import parse_spelled
            from ..numbers.spoken import only_digits, words_to_digits
            digits = only_digits(words_to_digits(text))
            truth = {"number": digits} if digits else {"name": parse_spelled(text)}
            st = await self.shadow.transcribe(TurnInput(None, 8000, "", [], text, truth, seed, accent))
            out = (st.text or "", len(text.split()) / WORDS_PER_SECOND)
        else:
            from .synth import ACCENTS, tts_openai
            voice, instr = ACCENTS.get(accent, ACCENTS["us"])
            instr += " Say it clearly, as an answer on a phone call."
            raw = await asyncio.to_thread(tts_openai, text, voice, instr)
            x, sr = read_wav(raw)
            y, sr8 = telephony_channel(x, sr, seed=seed)
            st = await self.shadow.transcribe(TurnInput(y, sr8, "", [], text, {}, seed, accent))
            out = (st.text or "", len(y) / sr8)
        self.cache[key] = out
        return out


async def recover_rows(rows: pd.DataFrame, recognizer: Recognizer, progress=None) -> pd.DataFrame:
    todo = rows[rows["action"] == "confirm"]
    out = []
    for n, (_, r) in enumerate(todo.iterrows()):
        row = r.to_dict()
        kind = field_kind(row["field"])
        ev = _evidence(row, kind)
        rec = {"case_id": row["case_id"], "field": row["field"], "trial": row.get("trial", 0),
               "model": row["model"], "evidence": ev}
        for strat in STRATEGIES:
            rep = plan_repair(row["field"], kind, row.get("before_value"), ev, str(row["expected"]), strat)
            heard, secs = [], rep.question_seconds()
            for i, a in enumerate(rep.asks):
                text, dur = await recognizer(a.reply, row.get("accent") or "us", 1000 + n * 7 + i)
                heard.append(text)
                secs += dur
            final = apply_reply(rep, row.get("before_value"), heard, kind)
            rec.update({
                f"{strat}_question": " / ".join(a.question for a in rep.asks),
                f"{strat}_heard": " / ".join(heard),
                f"{strat}_value": final,
                f"{strat}_ok": norm(kind, final) == norm(kind, row["expected"]),
                f"{strat}_units": rep.units, f"{strat}_of": rep.total_units,
                f"{strat}_seconds": round(secs, 2),
            })
        out.append(rec)
        if progress:
            progress(n + 1, len(todo))
    return pd.DataFrame(out)


def end_to_end(rows: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Per model: share of values saved correctly, and the caller's extra time.

    without voxguard      the tool value as the model wrote it
    voxguard, no repair   pass/correct as before; a confirmation saves nothing yet
    voxguard + full       confirmations repaired by asking for the whole value again
    voxguard + targeted   confirmations repaired by asking only for the part in doubt
    """
    keys = ["case_id", "field", "trial", "model"]
    df = rows.copy()
    df["trial"] = df.get("trial", 0)
    if not rec.empty:
        df = df.merge(rec, on=keys, how="left")
    out = []
    for model, g in df.groupby("model"):
        n = len(g)
        base = (g["before"] == "correct")
        guard = g["after"].isin(["correct", "corrected"])
        line = {"model": model, "n": n, "without_voxguard": base.mean(), "voxguard_no_repair": guard.mean()}
        conf = g["action"] == "confirm"
        for s in STRATEGIES:
            col = f"{s}_ok"
            ok = g[col].fillna(False).astype(bool) if col in g else pd.Series(False, index=g.index)
            line[f"voxguard_{s}"] = (guard | (conf & ok)).mean()
            sec = g[f"{s}_seconds"] if f"{s}_seconds" in g else pd.Series(dtype=float)
            line[f"{s}_seconds_per_call"] = sec.fillna(0).sum() / n
            units, of = g.get(f"{s}_units"), g.get(f"{s}_of")
            line[f"{s}_share_repeated"] = (units[conf].sum() / of[conf].sum()) if (units is not None and
                                                                                    of[conf].sum()) else 0.0
        line["asked_rate"] = conf.mean()
        out.append(line)
    return pd.DataFrame(out)


def save_recovery(run_dir: Path, rec: pd.DataFrame, mode: str) -> Path:
    p = Path(run_dir) / "recovery.jsonl"
    rec.assign(mode=mode).to_json(p, orient="records", lines=True)
    return p


def load_recovery(run_dir: Path) -> pd.DataFrame:
    p = Path(run_dir) / "recovery.jsonl"
    return pd.read_json(p, lines=True, dtype=False) if p.exists() else pd.DataFrame()


def recovery_meta(run_dir: Path) -> dict:
    p = Path(run_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}
