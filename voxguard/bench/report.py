"""Static report: PNG charts plus a markdown summary, built from saved runs.

    voxguard report --runs runs --out report

Uses the most recent run for each model. Colors: before = orange, after = blue
(a validated pair, CVD-safe); text stays in ink colors; the neutral latency chart
uses ink only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import bootstrap_ci, overall, summarize
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


def _before_after(ax, d: pd.DataFrame, title: str, label_after: bool = True, ci: pd.DataFrame | None = None):
    import numpy as np
    d = d.sort_values("before_error")
    y = np.arange(len(d))
    h = 0.36
    berr = aerr = None
    if ci is not None and not ci.empty:
        c = d[["model"]].merge(ci, on="model", how="left")
        berr = np.vstack([(d["before_error"].to_numpy() - c["before_lo"].to_numpy()),
                          (c["before_hi"].to_numpy() - d["before_error"].to_numpy())]) * 100
        aerr = np.vstack([(d["after_silent_error"].to_numpy() - c["after_lo"].to_numpy()),
                          (c["after_hi"].to_numpy() - d["after_silent_error"].to_numpy())]) * 100
    ekw = {"ecolor": INK_2, "elinewidth": 1, "capsize": 2}
    ax.barh(y + h / 2 + 0.01, d["before_error"] * 100, height=h, color=BEFORE, label="Before",
            xerr=berr, error_kw=ekw)
    ax.barh(y - h / 2 - 0.01, d["after_silent_error"] * 100, height=h, color=AFTER, label="After",
            xerr=aerr, error_kw=ekw)
    if label_after:
        right = d["after_silent_error"].to_numpy() * 100
        if aerr is not None:
            right = right + np.nan_to_num(aerr[1])
        for yi, v, r in zip(y, d["after_silent_error"] * 100, right):
            ax.text(r + 0.4, yi - h / 2 - 0.01, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d["model"], color=INK)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    _style(ax, "Silent error rate (%)")


EXCLUDE_FILE = "data/excluded_cases.txt"


def read_excluded(path: str = EXCLUDE_FILE) -> list[str]:
    p = Path(path)
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def build_report(runs_dir: str = "runs", out_dir: str = "report") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(runs_dir)
    if runs.empty:
        raise SystemExit(f"no runs in {runs_dir}")
    rows = latest_per_model(runs)
    excluded = read_excluded()
    if excluded:
        rows = rows[~rows["case_id"].isin(excluded)]
    summ = pd.concat([summarize(rows), overall(rows)], ignore_index=True)
    ci = bootstrap_ci(rows)
    has_accents = rows["accent"].nunique() > 1 if "accent" in rows else False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "summary.csv", index=False)
    n_models = rows["model"].nunique()
    height = max(2.4, 0.55 * n_models + 1.2)

    # 1. headline: silent error before vs after, all entities
    fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
    _before_after(ax, summ[summ["kind"] == "all"],
                  "Wrong values reaching the backend without the caller being asked (95% CI)", ci=ci)
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

    # 5. heatmaps: before and after on one shared sequential scale, by entity and by accent
    def heatmap(by: str, fname: str, title: str):
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        tab = summarize(rows, by=by)
        before = tab.pivot(index="model", columns=by, values="before_error") * 100
        after = tab.pivot(index="model", columns=by, values="after_silent_error") * 100
        order = before.mean(axis=0).sort_values().index
        before, after = before[order], after[order]
        labels = [KIND_LABEL.get(c, c) for c in order]
        vmax = max(np.nanmax(before.to_numpy()), 1.0)
        cmap = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#9cc3ef", AFTER, "#123f73"])
        fig, axes = plt.subplots(1, 2, figsize=(min(16, max(10, 0.75 * len(order) * 2)), height * 1.1),
                                 facecolor=SURFACE, sharey=True)
        im = None
        for ax, mat, sub in [(axes[0], before, f"Before: error rate by {title} (%)"),
                             (axes[1], after, f"After: silent error rate by {title} (%)")]:
            im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(order)), labels, rotation=35, ha="right", color=INK, fontsize=9)
            ax.set_yticks(range(len(mat.index)), mat.index, color=INK, fontsize=9)
            ax.set_title(sub, loc="left", fontsize=10, color=INK)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.tick_params(length=0)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.iat[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                                color="white" if v > vmax * 0.55 else INK)
        fig.colorbar(im, ax=axes, shrink=0.8, label="%")
        fig.savefig(out / fname, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        tab.to_csv(out / f"summary_by_{by}.csv", index=False)
        return before.round(1), after.round(1)

    entity_table = heatmap("kind", "entities.png", "entity")
    acc_table = heatmap("accent", "accents.png", "accent") if has_accents else None

    # 6. turn-taking: how often the agent called the tool before the caller finished
    d = summ[summ["kind"] == "all"].sort_values("cut_in")
    if "cut_in" in d and d["cut_in"].fillna(0).max() > 0:
        fig, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE)
        ax.barh(d["model"], d["cut_in"].fillna(0) * 100, height=0.5, color=INK_2)
        for i, v in enumerate(d["cut_in"].fillna(0) * 100):
            ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
        ax.set_title("Calls where the agent recorded a value before the caller finished", loc="left",
                     fontsize=10, color=INK)
        ax.tick_params(axis="y", colors=INK)
        _style(ax, "Cut-in rate (%)")
        fig.tight_layout()
        fig.savefig(out / "cut_in.png", dpi=160)
        plt.close(fig)

    # markdown summary
    t = summ[summ["kind"] == "all"].copy()
    t = t.merge(ci, on="model", how="left")
    lines = ["# voxguard results", "",
             f"Runs: {', '.join(sorted(rows['run'].unique()))}", "",
             "Intervals are 95% bootstrap confidence intervals, resampling whole test cases.", "",
             *([f"Excluded {len(excluded)} test cases whose scripted words do not say the labeled number "
                f"(found by `voxguard audit`): {', '.join(excluded)}.", ""] if excluded else []),
             *([f"The second recognizer took {rows['shadow_ms'].median():.0f} ms per turn (median). It runs "
                "alongside the agent, so it adds cost but not latency unless it is slower than the agent.", ""]
               if "shadow_ms" in rows and rows["shadow_ms"].notna().any() else []),
             "| Model | n | Before error | After silent error | Reduction | Caught | Corrected | "
             "False confirm | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values("after_silent_error").iterrows():
        red = "" if pd.isna(r["relative_reduction"]) else f"{r['relative_reduction'] * 100:.0f}%"
        lines.append(f"| {r['model']} | {r['n']} | {r['before_error'] * 100:.1f}% "
                     f"({r['before_lo'] * 100:.1f} to {r['before_hi'] * 100:.1f}) | "
                     f"{r['after_silent_error'] * 100:.1f}% "
                     f"({r['after_lo'] * 100:.1f} to {r['after_hi'] * 100:.1f}) | {red} | "
                     f"{r['caught'] * 100:.1f}% | {r['corrected'] * 100:.1f}% | {r['false_confirm'] * 100:.1f}% | "
                     f"{r['agent_p50_ms']:.0f} | {r['agent_p95_ms']:.0f} |")
    if acc_table is not None:
        b, a = acc_table
        lines += ["", "## By accent", "", "Before error (%):", "", b.to_markdown(), "",
                  "After silent error (%):", "", a.to_markdown(), "", "![Accents](accents.png)"]
    b, a = entity_table
    lines += ["", "## By entity", "", "Before error (%):", "", b.to_markdown(), "",
              "After silent error (%):", "", a.to_markdown(), "",
              "![Silent error](silent_error.png)", "", "![Entities](entities.png)", "",
              "![By entity](silent_error_by_entity.png)", "", "![False confirmations](false_confirm.png)", ""]
    if (out / "cut_in.png").exists():
        lines += ["![Cut-ins](cut_in.png)", ""]
    lines += ["![Latency](latency.png)", "",
              "Latency is measured from the end of caller audio to the tool call. Models with manual turn "
              "commits (OpenAI, Gemini) and platforms with their own end-of-turn detection (Deepgram, "
              "AssemblyAI, ElevenLabs, Vapi) are measured differently; compare within each group.", ""]
    if rows["model"].str.startswith("mock:").any():
        lines.insert(2, "**Includes simulated mock models. Do not report those rows as real results.**\n")
    (out / "REPORT.md").write_text("\n".join(lines))
    return out
