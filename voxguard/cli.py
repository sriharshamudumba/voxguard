"""Command line.

  voxguard gen    --n 50 --out data/cases.jsonl
  voxguard synth  --cases data/cases.jsonl --provider openai --accents all
  voxguard bench  --config configs/example.toml
  voxguard vad    --cases data/cases.jsonl
  voxguard report --runs runs --out report
  voxguard samples   (pick demo calls for the Live tab)
  voxguard publish   (copy the latest runs into published/ for the hosted demo)
  voxguard probe --config configs/deepgram_endpointing.toml --expect 8882722028
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tomllib
from pathlib import Path

import pandas as pd

from .backends.registry import build_agent, build_shadow
from .bench.dataset import generate_cases, load_cases, save_cases
from .bench.runner import RunConfig, run_benchmark, save_run


def _fmt(df: pd.DataFrame) -> str:
    cols = ["model", "kind", "n", "errors", "missing", "agent_asked", "cut_in", "before_error", "after_silent_error",
            "relative_reduction", "caught", "corrected", "false_confirm", "agent_p50_ms", "guard_p95_ms"]
    d = df[[c for c in cols if c in df]].copy()
    for c in ["missing", "agent_asked", "cut_in", "before_error", "after_silent_error", "relative_reduction", "caught",
              "corrected", "false_confirm"]:
        if c in d:
            d[c] = (d[c] * 100).round(1).astype(str) + "%"
    for c in ["agent_p50_ms", "guard_p95_ms"]:
        if c in d:
            d[c] = d[c].round(1)
    return d.to_string(index=False)


def recover(runs_dir: str, config: str, mode: str, model: str | None) -> str:
    """Layer 3 on the latest run per model; writes recovery.jsonl next to each run's rows."""
    from .bench.recover import Recognizer, end_to_end, recover_rows, save_recovery
    from .bench.report import latest_per_model, read_excluded
    from .bench.runner import load_runs

    runs = load_runs(runs_dir)
    if runs.empty:
        return f"no runs in {runs_dir}"
    rows = latest_per_model(runs)
    rows = rows[~rows["case_id"].isin(read_excluded())]
    if "agent_error" in rows:
        rows = rows[rows["agent_error"].isna()]   # failed calls are not scored
    if model:
        rows = rows[rows["model"].str.contains(model, case=False)]
    shadow = None
    if mode == "tts":
        with open(config, "rb") as f:
            shadow = build_shadow(tomllib.load(f).get("shadow", {"type": "mock"}))
    recog = Recognizer(shadow, mode)
    recs = []
    for run, g in rows.groupby("run"):
        n = int((g["action"] == "confirm").sum())
        print(f"{g['model'].iloc[0]}: {n} confirmations to replay", flush=True)
        rec = asyncio.run(recover_rows(g, recog))
        save_recovery(Path(runs_dir) / run, rec, mode)
        recs.append(rec)
    rec = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    e2e = end_to_end(rows, rec)
    pct = [c for c in e2e.columns if c.startswith(("without", "voxguard", "asked")) or c.endswith("repeated")]
    show = e2e.copy()
    for c in pct:
        show[c] = (show[c] * 100).map(lambda v: f"{v:.1f}%")
    for c in [c for c in e2e.columns if c.endswith("seconds_per_call")]:
        show[c] = show[c].map(lambda v: f"{v:.2f}s")
    note = "" if mode == "tts" else "\n(text mode: answers taken as said, so recovery is an upper bound)"
    return "Values saved correctly, per call:\n" + show.to_string(index=False) + note


def probe(config: str, expect: str, accent: str) -> str:
    """Run one matching case through every agent in a config; show the agent's own transcript."""
    from .backends.audio import read_wav
    from .backends.base import TurnInput
    from .bench.dataset import SYSTEM_PROMPT, TOOLS

    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    cases = [c for c in load_cases(cfg["data"]["cases"])
             if expect in [str(v) for v in c.expected.values()] and (c.accent or "") == accent]
    if not cases:
        return f"no case with expected value {expect} and accent {accent}"
    c = cases[0]
    audio, sr = read_wav(c.audio_path)
    out = [f"case {c.id}  accent {c.accent}  audio {c.audio_path}",
           f"caller text: {c.caller_text}", f"expected:    {c.expected}"]
    for acfg in cfg["agents"]:
        agent = build_agent(acfg)
        res = asyncio.run(agent.run_turn(TurnInput(audio, sr, SYSTEM_PROMPT, TOOLS, c.caller_text,
                                                   c.expected, c.seed, c.accent)))
        out += ["", f"== {agent.name}",
                f"agent heard:  {res.raw.get('heard')}",
                f"tool calls:   {[t.args for t in res.tool_calls]}",
                f"cut in:       {res.raw.get('cut_in')}   calls: {res.raw.get('n_tool_calls')}",
                f"warnings:     {res.raw.get('warnings') or 'none'}"]
        if res.error:
            out.append(f"error:        {res.error[:300]}")
    return "\n".join(out)


KIND_LABEL = {"phone": "Phone number", "order_id": "Order ID", "name": "Name on the booking list",
              "name_new": "First-time caller name"}


def make_samples(cases_path: str, out: str, per_kind: int, runs_dir: str | None) -> str:
    """Copy a few benchmark calls (audio + ground truth) into demo/samples with a manifest."""
    import json
    import shutil
    from pathlib import Path

    from .bench.report import read_excluded
    from .bench.runner import load_runs
    excluded = set(read_excluded())   # mislabeled cases found by `voxguard audit`
    cases = [c for c in load_cases(cases_path)
             if c.audio_path and Path(c.audio_path).exists() and c.id not in excluded]
    if not cases:
        return "no cases with audio; run `voxguard synth` first"
    hard: dict[str, int] = {}
    if runs_dir and Path(runs_dir).exists():
        runs = load_runs(runs_dir)
        if not runs.empty:
            bad = runs[runs["before"].isin(["wrong", "missing"])]
            hard = bad.groupby("case_id").size().to_dict()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for kind in ["phone", "order_id", "name", "name_new"]:
        pool = sorted([c for c in cases if c.kind == kind], key=lambda c: -hard.get(c.id, 0))
        seen_accents: set[str] = set()
        picked = []
        for c in pool:          # prefer different accents among the hardest calls
            if len(picked) >= per_kind:
                break
            if c.accent in seen_accents and len(pool) > per_kind:
                continue
            picked.append(c)
            seen_accents.add(c.accent)
        for c in picked:
            fname = f"{c.id}.wav"
            shutil.copy(c.audio_path, out_dir / fname)
            manifest.append({
                "file": fname, "label": f"{KIND_LABEL[kind]}, {c.accent or 'mixed'} accent",
                "text": c.caller_text, "accent": c.accent, "truth": c.expected,
                "roster": c.roster, "order_ids": c.known_values.get("order_id", []),
                "models_wrong_in_benchmark": int(hard.get(c.id, 0)),
            })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return f"wrote {len(manifest)} samples to {out_dir}"


def publish_runs(runs_dir: str, out: str) -> str:
    """Latest run per model, copied so it can be committed for the hosted demo."""
    import shutil
    from pathlib import Path

    from .bench.runner import load_runs

    runs = load_runs(runs_dir)
    if runs.empty:
        return f"no runs in {runs_dir}"
    latest = runs.groupby("model")["run"].max().tolist()
    out_dir = Path(out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    from .bench.report import read_excluded
    excluded = set(read_excluded())
    for r in latest:
        shutil.copytree(Path(runs_dir) / r, out_dir / r, ignore=shutil.ignore_patterns("*.wav"))
        rows_path = out_dir / r / "rows.jsonl"
        if excluded and rows_path.exists():
            keep = [ln for ln in rows_path.read_text().splitlines()
                    if ln.strip() and json.loads(ln).get("case_id") not in excluded]
            rows_path.write_text("\n".join(keep) + "\n")
    return f"published {len(latest)} runs to {out_dir}: " + ", ".join(latest)


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

    sp = sub.add_parser("samples", help="pick benchmark calls as demo samples for the Live tab")
    sp.add_argument("--cases", default="data/cases.jsonl")
    sp.add_argument("--out", default="demo/samples")
    sp.add_argument("--per-kind", type=int, default=2)
    sp.add_argument("--from-runs", default="runs",
                    help="prefer calls that some model got wrong in these runs")

    pb = sub.add_parser("publish", help="copy the latest run per model into published/ for the hosted demo")
    pb.add_argument("--runs", default="runs")
    pb.add_argument("--out", default="published")

    sv = sub.add_parser("serve", help="run voxguard as an HTTP service for hosted agents (Vapi, Retell, webhooks)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--fields", default="phone:phone,order_id:order_id,name:name",
                    help="tool argument:kind pairs; kinds: phone, order_id, card, number, name")
    sv.add_argument("--roster", default=None, help="text file, one expected name per line")
    sv.add_argument("--log", default="voxguard_calls.jsonl", help="decision log for `voxguard calls`")
    sv.add_argument("--transcript-source", choices=["pipeline", "side_channel"], default="pipeline")
    sv.add_argument("--no-redact", action="store_true", help="log values unmasked (test calls only)")

    cl = sub.add_parser("calls", help="summarize what voxguard did on real calls, from the decision log")
    cl.add_argument("--log", default="voxguard_calls.jsonl")

    rc = sub.add_parser("recover", help="Layer 3: replay each confirmation as a follow-up turn and score it")
    rc.add_argument("--runs", default="runs")
    rc.add_argument("--config", default="configs/deepgram_endpointing.toml",
                    help="config whose [shadow] recognizes the caller's answers")
    rc.add_argument("--mode", choices=["tts", "text"], default="tts",
                    help="tts: speak and recognize each answer (realistic); text: take answers as said")
    rc.add_argument("--model", default=None, help="only runs whose model name contains this")

    au = sub.add_parser("audit", help="find test cases whose scripted words do not say the labeled number")
    au.add_argument("--cases", default="data/cases.jsonl")

    pr = sub.add_parser("probe", help="run one call per agent and print what the agent heard and recorded")
    pr.add_argument("--config", required=True)
    pr.add_argument("--expect", required=True, help="an expected value to match, e.g. 8882722028")
    pr.add_argument("--accent", default="us")

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
            n_err = int(rows["agent_error"].notna().sum()) if not rows.empty else 0
            if n_err:
                first = rows["agent_error"].dropna().iloc[0]
                print(f"WARNING: {n_err} of {len(rows)} rows failed. These are not model errors. First error:\n  "
                      f"{str(first)[:300]}")
    elif a.cmd == "report":
        from .bench.report import build_report
        print(f"report written to {build_report(a.runs, a.out)}")
    elif a.cmd == "samples":
        print(make_samples(a.cases, a.out, a.per_kind, a.from_runs))
    elif a.cmd == "publish":
        print(publish_runs(a.runs, a.out))
    elif a.cmd == "serve":
        import os

        from .sdk import Guard
        from .server import serve
        fields = dict(x.split(":") for x in a.fields.split(",") if x)
        roster = Path(a.roster).read_text().splitlines() if a.roster else []
        serve(Guard(fields, [r for r in roster if r.strip()], log_path=a.log, transcript_source=a.transcript_source,
                    redact=not a.no_redact), a.host, a.port, os.environ.get("VOXGUARD_TOKEN"))
    elif a.cmd == "calls":
        from .sdk import summarize_log
        print(summarize_log(a.log))
    elif a.cmd == "recover":
        print(recover(a.runs, a.config, a.mode, a.model))
    elif a.cmd == "audit":
        from .bench.dataset import audit_cases
        from .bench.report import EXCLUDE_FILE
        bad = audit_cases(load_cases(a.cases))
        ids = sorted({b[0] for b in bad})
        for cid, f, exp, heard in bad:
            print(f"{cid}  {f}: labeled {exp}, but the words say {heard}")
        Path(EXCLUDE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(EXCLUDE_FILE).write_text("\n".join(ids) + ("\n" if ids else ""))
        print(f"{len(ids)} cases written to {EXCLUDE_FILE}; report and publish leave them out")
    elif a.cmd == "probe":
        print(probe(a.config, a.expect, a.accent))
    elif a.cmd == "vad":
        from .bench.vad_sim import simulate, summarize_vad
        from .vad import EntityAwareEndpointer, FixedEndpointer
        cases = load_cases(a.cases) if a.cases else generate_cases(a.n)
        eps = {"fixed_500": FixedEndpointer(500), "fixed_800": FixedEndpointer(800),
               "entity_aware": EntityAwareEndpointer()}
        print(summarize_vad(simulate(cases, eps)).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
