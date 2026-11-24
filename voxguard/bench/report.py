"""Static report: PNG charts plus a markdown summary, built from saved runs.

    voxguard report --runs runs --out report

Uses the most recent run for each model. Colors: before = orange, after = blue
(a validated pair, CVD-safe); text stays in ink colors; the neutral latency chart
uses ink only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import overall, summarize
from .runner import load_runs

BEFORE = "#eb6834"
AFTER = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"
KIND_LABEL = {"all": "All entities", "phone": "Phone numbers", "order_id": "Order IDs",
              "name": "Names in roster", "name_new": "First-time names"}


def latest_per_model(runs: pd.DataFrame) -> pd.DataFrame:
    last = runs.groupby("model")["run"].max()
    return runs[runs.apply(lambda r: r["run"] == last[r["model"]], axis=1)]


def _style(ax, xlabel: str):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)


def _before_after(ax, d: pd.DataFrame, title: str, label_after: bool = True):
    import numpy as np
    d = d.sort_values("before_error")
    y = np.arange(len(d))
    h = 0.36
    ax.barh(y + h / 2 + 0.01, d["before_error"] * 100, height=h, color=BEFORE, label="Before")
    ax.barh(y - h / 2 - 0.01, d["after_silent_error"] * 100, height=h, color=AFTER, label="After")
    if label_after:
        for yi, v in zip(y, d["after_silent_error"] * 100):
            ax.text(v + 0.3, yi - h / 2 - 0.01, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d["model"], color=INK)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    _style(ax, "Silent error rate (%)")


def build_report(runs_dir: str = "runs", out_dir: str = "report") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(runs_dir)
    if runs.empty:
        raise SystemExit(f"no runs in {runs_dir}")
    rows = latest_per_model(runs)
    summ = pd.concat([summarize(rows), overall(rows)], ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "summary.csv", index=False)
    n_models = rows["model"].nunique()
    height = max(2.4, 0.55 * n_models + 1.2)

    # 1. headline: silent error before vs after, all entities
    fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
    _before_after(ax, summ[summ["kind"] == "all"], "Wrong values reaching the backend without the caller being asked")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "silent_error.png", dpi=160)
    plt.close(fig)

    # 2. small multiples by entity type
    kinds = [k for k in ["phone", "order_id", "name", "name_new"] if k in set(summ["kind"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, height * 1.7), facecolor=SURFACE, sharex=True)
    for ax, k in zip(axes.flat, kinds):
        _before_after(ax, summ[summ["kind"] == k], KIND_LABEL[k])
    for ax in list(axes.flat)[len(kinds):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "silent_error_by_entity.png", dpi=160)
    plt.close(fig)

    # 3. the cost of the fix: false confirmations
    d = summ[summ["kind"] == "all"].sort_values("false_confirm")
    fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
    ax.barh(d["model"], d["false_confirm"] * 100, height=0.5, color=AFTER)
    for i, v in enumerate(d["false_confirm"] * 100):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_title("Correct values the plugin asked about anyway (one extra turn each)", loc="left",
                 fontsize=10, color=INK)
    ax.tick_params(axis="y", colors=INK)
    _style(ax, "False confirmation rate (%)")
    fig.tight_layout()
    fig.savefig(out / "false_confirm.png", dpi=160)
    plt.close(fig)

    # 4. latency range p50 to p95 (ink only; one measure, one axis)
    d = summ[summ["kind"] == "all"].sort_values("agent_p50_ms")
    fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["agent_p50_ms"], r["agent_p95_ms"]], [i, i], color=INK_2, linewidth=2)
        ax.plot(r["agent_p50_ms"], i, "o", color=INK, markersize=8, label="p50" if i == 0 else None)
        ax.plot(r["agent_p95_ms"], i, "o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=8,
                label="p95" if i == 0 else None)
    ax.set_yticks(range(len(d)), d["model"], color=INK)
    ax.set_title("Agent response latency per turn", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    _style(ax, "Milliseconds (end of caller speech to tool call)")
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=160)
    plt.close(fig)

    # markdown summary
    t = summ[summ["kind"] == "all"].copy()
    lines = ["# voxguard results", "",
             f"Runs: {', '.join(sorted(rows['run'].unique()))}", "",
             "| Model | n | Before error | After silent error | Reduction | Caught | Corrected | "
             "False confirm | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values("after_silent_error").iterrows():
        red = "" if pd.isna(r["relative_reduction"]) else f"{r['relative_reduction'] * 100:.0f}%"
        lines.append(f"| {r['model']} | {r['n']} | {r['before_error'] * 100:.1f}% | "
                     f"{r['after_silent_error'] * 100:.1f}% | {red} | {r['caught'] * 100:.1f}% | "
                     f"{r['corrected'] * 100:.1f}% | {r['false_confirm'] * 100:.1f}% | "
                     f"{r['agent_p50_ms']:.0f} | {r['agent_p95_ms']:.0f} |")
    lines += ["", "![Silent error](silent_error.png)", "", "![By entity](silent_error_by_entity.png)", "",
              "![False confirmations](false_confirm.png)", "", "![Latency](latency.png)", ""]
    if rows["model"].str.startswith("mock:").any():
        lines.insert(2, "**Includes simulated mock models. Do not report those rows as real results.**\n")
    (out / "REPORT.md").write_text("\n".join(lines))
    return out
