"""Command line.

  voxguard gen    --n 50 --out data/cases.jsonl
  voxguard synth  --cases data/cases.jsonl --provider openai --accents all
  voxguard bench  --config configs/example.toml
  voxguard vad    --cases data/cases.jsonl
  voxguard report --runs runs --out report
"""
from __future__ import annotations

import argparse
import asyncio
import tomllib

import pandas as pd

from .backends.registry import build_agent, build_shadow
from .bench.dataset import generate_cases, load_cases, save_cases
from .bench.runner import RunConfig, run_benchmark, save_run


def _fmt(df: pd.DataFrame) -> str:
    cols = ["model", "kind", "n", "before_error", "after_silent_error", "relative_reduction",
            "caught", "corrected", "false_confirm", "agent_p50_ms", "guard_p95_ms"]
    d = df[[c for c in cols if c in df]].copy()
    for c in ["before_error", "after_silent_error", "relative_reduction", "caught", "corrected", "false_confirm"]:
        if c in d:
            d[c] = (d[c] * 100).round(1).astype(str) + "%"
    for c in ["agent_p50_ms", "guard_p95_ms"]:
        if c in d:
            d[c] = d[c].round(1)
    return d.to_string(index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="voxguard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate benchmark cases")
    g.add_argument("--n", type=int, default=50, help="cases per kind")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/cases.jsonl")
    g.add_argument("--accents", default="",
                   help="expand each case into one per accent now ('all' or comma list); synth can also do this")

    s = sub.add_parser("synth", help="synthesize caller audio through a phone channel")
    s.add_argument("--cases", default="data/cases.jsonl")
    s.add_argument("--provider", default="openai", choices=["openai", "piper", "espeak"])
    s.add_argument("--out", default="data/audio")
    s.add_argument("--snr", type=float, default=20.0)
    s.add_argument("--loss", type=float, default=0.01)
    s.add_argument("--accents", default="all",
                   help="'all', 'rotate' (one accent per case), or a comma list, e.g. us,indian,scottish")

    b = sub.add_parser("bench", help="run before/after benchmark from a TOML config")
    b.add_argument("--config", required=True)

    r = sub.add_parser("report", help="charts and markdown summary from saved runs")
    r.add_argument("--runs", default="runs")
    r.add_argument("--out", default="report")

    v = sub.add_parser("vad", help="offline endpointing benchmark")
    v.add_argument("--cases", default=None)
    v.add_argument("--n", type=int, default=100)

    a = ap.parse_args(argv)
    if a.cmd == "gen":
        cases = generate_cases(a.n, seed=a.seed)
        if a.accents:
            from .bench.dataset import expand_accents
            from .bench.synth import ACCENTS
            cases = expand_accents(cases, list(ACCENTS) if a.accents == "all" else a.accents.split(","))
        save_cases(cases, a.out)
        print(f"wrote {len(cases)} cases to {a.out}")
    elif a.cmd == "synth":
        from .bench.dataset import expand_accents
        from .bench.synth import ACCENTS, synthesize_cases
        cases = load_cases(a.cases)
        if a.accents != "rotate":
            chosen = list(ACCENTS) if a.accents == "all" else [x.strip() for x in a.accents.split(",")]
            unknown = [x for x in chosen if x not in ACCENTS]
            if unknown:
                raise SystemExit(f"unknown accents {unknown}; choose from {list(ACCENTS)}")
            cases = expand_accents(cases, chosen)
        cases = synthesize_cases(cases, a.out, a.provider, a.snr, a.loss)
        save_cases(cases, a.cases)
        print(f"synthesized {len(cases)} files into {a.out}; updated {a.cases}")
    elif a.cmd == "bench":
        with open(a.config, "rb") as f:
            cfg = tomllib.load(f)
        cases = load_cases(cfg["data"]["cases"])
        shadow = build_shadow(cfg.get("shadow", {"type": "mock"}))
        rc = RunConfig(concurrency=cfg.get("run", {}).get("concurrency", 4),
                       repeats=cfg.get("run", {}).get("repeats", 1),
                       use_keyterms=cfg.get("run", {}).get("use_keyterms", True))
        for acfg in cfg["agents"]:
            agent = build_agent(acfg)
            rows, summ = asyncio.run(run_benchmark(cases, agent, shadow, cfg=rc))
            d = save_run(rows, summ, cfg.get("run", {}).get("out_dir", "runs"),
                         meta={"agent": {k: v for k, v in acfg.items() if "key" not in k},
                               "shadow": {k: v for k, v in cfg.get("shadow", {}).items() if "key" not in k}})
            print(f"\n{agent.name}  ->  {d}")
            print(_fmt(summ))
    elif a.cmd == "report":
        from .bench.report import build_report
        print(f"report written to {build_report(a.runs, a.out)}")
    elif a.cmd == "vad":
        from .bench.vad_sim import simulate, summarize_vad
        from .vad import EntityAwareEndpointer, FixedEndpointer
        cases = load_cases(a.cases) if a.cases else generate_cases(a.n)
        eps = {"fixed_500": FixedEndpointer(500), "fixed_800": FixedEndpointer(800),
               "entity_aware": EntityAwareEndpointer()}
        print(summarize_vad(simulate(cases, eps)).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
