"""Run every case through an agent, with and without the plugin chain.

"Before" and "after" use the same agent output for each case, so the
comparison isolates the plugin's effect: the agent is called once per case,
then the chain is applied to that exact tool call. The shadow recognizer runs
concurrently with the agent, as it would in production.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..backends.audio import read_wav
from ..backends.base import AgentBackend, ShadowASR, TurnInput
from ..core.plugin import PluginChain
from ..core.types import ShadowTranscript, ToolCall, TurnContext
from ..names import NameGuard
from ..numbers import NumberGuard
from .dataset import SYSTEM_PROMPT, TOOLS, Case
from .metrics import overall, score_field, summarize


def default_chain() -> PluginChain:
    return PluginChain([NumberGuard(fields={"phone": "phone", "order_id": "order_id"}), NameGuard()])


@dataclass
class RunConfig:
    concurrency: int = 4
    repeats: int = 1               # run the whole set this many times (models are not deterministic)
    use_keyterms: bool = True      # pass roster names to the shadow recognizer
    out_dir: str = "runs"
    tag: str = ""
    extra: dict = field(default_factory=dict)


async def _one(case: Case, agent: AgentBackend, shadow: ShadowASR | None, chain: PluginChain,
               cfg: RunConfig, trial: int = 0) -> list[dict]:
    audio, sr = (read_wav(case.audio_path) if case.audio_path else (np.zeros(1600, np.float32), 8000))
    turn = TurnInput(audio, sr, SYSTEM_PROMPT, TOOLS, case.caller_text, case.expected,
                     case.seed + 7919 * trial, case.accent)
    keyterms = case.roster if (cfg.use_keyterms and case.roster) else None

    async def run_shadow():
        if shadow is None:
            return None, 0.0
        t = time.perf_counter()
        try:
            st = await shadow.transcribe(turn, keyterms)
        except Exception as e:  # shadow failure must not break the run
            st = ShadowTranscript("", None)
            st.error = repr(e)  # type: ignore[attr-defined]
        return st, (time.perf_counter() - t) * 1000

    res, (st, shadow_ms) = await asyncio.gather(agent.run_turn(turn), run_shadow())
    call = next((c for c in res.tool_calls if c.name == "record_details"), None) or \
        (res.tool_calls[0] if res.tool_calls else ToolCall("record_details", {}))
    ctx = TurnContext(shadow=st, roster=case.roster, known_values=case.known_values)
    t = time.perf_counter()
    guarded = chain.guard_tool_call(call, ctx)
    guard_ms = (time.perf_counter() - t) * 1000
    by_field = {v.field: v for v in guarded.verdicts}

    rows = []
    for f, exp in case.expected.items():
        v = by_field.get(f)
        b, a = call.args.get(f), guarded.tool_call.args.get(f)
        before_o, after_o = score_field(f, exp, b, a, v.action.value if v else None)
        rows.append({
            "case_id": case.id, "kind": case.kind, "accent": case.accent or "unknown", "trial": trial,
            "field": f, "model": agent.name, "agent_kind": agent.kind,
            "expected": exp, "before_value": b, "after_value": a,
            "action": v.action.value if v else None, "p_error": v.p_error if v else None,
            "reason": v.reason if v else "", "before": before_o, "after": after_o,
            "shadow_text": st.text if st else None, "agent_ms": res.latency_ms, "shadow_ms": shadow_ms,
            "guard_ms": guard_ms, "agent_error": res.error, "caller_text": case.caller_text,
            "cut_in": bool(res.raw.get("cut_in", False)), "n_tool_calls": res.raw.get("n_tool_calls"),
            "agent_text": res.text,
        })
    return rows


async def run_benchmark(cases: list[Case], agent: AgentBackend, shadow: ShadowASR | None,
                        chain: PluginChain | None = None, cfg: RunConfig | None = None,
                        progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or RunConfig()
    chain = chain or default_chain()
    if agent.kind != "mock" and any(not c.audio_path for c in cases):
        raise ValueError("real backends need audio: run `voxguard synth` first (or use the UI's Data tab)")
    sem = asyncio.Semaphore(cfg.concurrency)
    done = 0
    all_rows: list[dict] = []

    jobs = [(c, t) for t in range(max(1, cfg.repeats)) for c in cases]

    async def guarded(c: Case, t: int):
        nonlocal done
        async with sem:
            rows = await _one(c, agent, shadow, chain, cfg, t)
        done += 1
        if progress:
            progress(done, len(jobs))
        return rows

    for rows in await asyncio.gather(*(guarded(c, t) for c, t in jobs)):
        all_rows.extend(rows)
    await agent.close()
    df = pd.DataFrame(all_rows)
    summ = pd.concat([summarize(df), overall(df)], ignore_index=True) if not df.empty else pd.DataFrame()
    return df, summ


def save_run(rows: pd.DataFrame, summary: pd.DataFrame, out_dir: str = "runs", tag: str = "",
             meta: dict | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = rows["model"].iloc[0] if not rows.empty else "empty"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
    d = Path(out_dir) / f"{stamp}_{safe}{('_' + tag) if tag else ''}"
    d.mkdir(parents=True, exist_ok=True)
    rows.to_json(d / "rows.jsonl", orient="records", lines=True)
    summary.to_csv(d / "summary.csv", index=False)
    (d / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
    return d


def load_runs(out_dir: str = "runs") -> pd.DataFrame:
    frames = []
    for d in sorted(Path(out_dir).glob("*/rows.jsonl")):
        df = pd.read_json(d, lines=True)
        df["run"] = d.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
