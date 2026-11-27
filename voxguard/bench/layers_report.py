"""Report section for the three layers: recognition, understanding, recovery."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .recover import end_to_end, load_recovery

# categorical hues, distinct from the before (orange) and after (blue) colours used elsewhere
LAYER_COLORS = {"turn_taking": "#1baf7a", "recognition": "#eda100", "spelling": "#e87ba4",
                "understanding": "#6250d6", "unknown": "#b5b4ae"}
LAYER_LABEL = {"turn_taking": "Turn-taking", "recognition": "Recognition", "spelling": "Name spelling",
               "understanding": "Understanding", "unknown": "No transcript"}
NUMERIC = {"phone", "order_id"}


def _pct(v: float) -> str:
    return "" if pd.isna(v) else f"{v * 100:.0f}%"


def layer1_table(rows: pd.DataFrame) -> pd.DataFrame:
    if "layer1" not in rows:
        return pd.DataFrame()
    out = []
    for model, g in rows.groupby("model"):
        seen = g[g["layer1"].notna()]
        if seen.empty:
            continue
        num, names = seen[seen["field"].isin(NUMERIC)], seen[~seen["field"].isin(NUMERIC)]
        src = seen["heard_source"].iloc[0] if "heard_source" in seen else ""
        out.append({
            "Model": model,
            "Transcript": "what the LLM received" if src == "pipeline" else "side transcript (model hears audio)",
            "Turns with a transcript": _pct(len(seen) / len(g)),
            "Numbers heard right": _pct((num["layer1"] == "correct").mean()) if len(num) else "",
            "Names heard right": _pct((names["layer1"] == "correct").mean()) if len(names) else "",
            "Names: same sound, other spelling": _pct((names["layer1"] == "sound_alike").mean()) if len(names) else "",
        })
    return pd.DataFrame(out)


def layer2_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Share of each model's wrong or missing values, by the layer that failed."""
    if "layer2" not in rows:
        return pd.DataFrame()
    bad = rows[rows["layer2"].notna() & (rows["layer2"] != "ok")]
    if bad.empty:
        return pd.DataFrame()
    t = pd.crosstab(bad["model"], bad["layer2"], normalize="index")
    n = bad.groupby("model").size().rename("errors")
    cols = [c for c in LAYER_COLORS if c in t.columns]
    return t[cols].join(n)


def layer2_chart(shares: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    from .report import GRID, INK, INK_2, SURFACE
    models = list(shares.index)
    cols = [c for c in LAYER_COLORS if c in shares.columns]
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.55 * len(models) + 1.3)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, m in enumerate(models):
        left = 0.0
        for c in cols:
            w = float(shares.loc[m, c])
            if w <= 0:
                continue
            ax.barh(i, w, left=left, height=0.55, color=LAYER_COLORS[c], edgecolor=SURFACE, linewidth=2)
            if w >= 0.08:
                ink = "#ffffff" if c == "understanding" else INK   # dark segment takes light text
                ax.text(left + w / 2, i, f"{w * 100:.0f}%", ha="center", va="center", fontsize=8, color=ink)
            left += w
        ax.text(1.01, i, f"n={int(shares.loc[m, 'errors'])}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(models)), models, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.set_title("Where wrong values came from, without voxguard", loc="left", fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS[c]) for c in cols]
    ax.legend(handles, [LAYER_LABEL[c] for c in cols], frameon=False, fontsize=8, ncol=len(cols),
              loc="upper left", bbox_to_anchor=(0, -0.12), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def layers_markdown(rows: pd.DataFrame, runs_dir: str, out: Path) -> list[str]:
    lines = ["", "## Three layers", "",
             "Each turn is scored on where it broke: **1. Recognition** (did the system hear the right words), "
             "**2. Understanding** (did those words become the right value), **3. Recovery** (when the value "
             "was wrong, did the agent get it right without the caller repeating everything).", ""]

    l1 = layer1_table(rows)
    lines += ["### Layer 1: recognition", ""]
    if l1.empty:
        lines += ["No transcripts in these runs. Rerun the benchmark to record each platform's transcript.", ""]
    else:
        lines += [l1.to_markdown(index=False), "",
                  "For speech-to-speech models the transcript comes from a separate recognizer; the model itself "
                  "hears audio, so their Layer 1 is an estimate.", ""]

    sh = layer2_shares(rows)
    lines += ["### Layer 2: where wrong values came from", ""]
    if sh.empty:
        lines += ["No layer attribution in these runs yet.", ""]
    else:
        layer2_chart(sh, out / "layers.png")
        tbl = sh.drop(columns="errors").map(_pct)
        tbl.columns = [LAYER_LABEL[c] for c in tbl.columns]
        tbl.insert(0, "Wrong or missing values", sh["errors"].astype(int))
        lines += ["![Where wrong values came from](layers.png)", "", tbl.to_markdown(), "",
                  "Understanding errors are fixed without asking the caller: voxguard re-reads the number from "
                  "the transcript with a grammar instead of trusting the LLM's conversion.", ""]

    recs = [load_recovery(Path(runs_dir) / r) for r in rows["run"].unique()]
    recs = [r for r in recs if not r.empty]
    lines += ["### Layer 3: recovery", ""]
    if not recs:
        lines += ["Run `voxguard recover` to replay each confirmation as a follow-up turn.", ""]
        return lines
    rec = pd.concat(recs, ignore_index=True)
    mode = rec["mode"].iloc[0] if "mode" in rec else "tts"
    e = end_to_end(rows, rec)
    t = pd.DataFrame({
        "Model": e["model"],
        "Saved correctly, without voxguard": e["without_voxguard"].map(_pct),
        "With voxguard + repeat everything": e["voxguard_full"].map(_pct),
        "With voxguard + targeted repair": e["voxguard_targeted"].map(_pct),
        "Asked the caller": e["asked_rate"].map(_pct),
        "Share re-said (targeted vs full)": [f"{_pct(a)} vs {_pct(b)}" for a, b in
                                             zip(e["targeted_share_repeated"], e["full_share_repeated"], strict=False)],
        "Extra seconds per call (targeted vs full)": [f"{a:.1f} vs {b:.1f}" for a, b in
                                                      zip(e["targeted_seconds_per_call"],
                                                          e["full_seconds_per_call"], strict=False)],
    })
    lines += [t.to_markdown(index=False), "",
              "Each confirmation is replayed: voxguard plans the question from the tool value and the evidence "
              "(never the true value); the caller's answer is spoken in the same accent, passed through the "
              "phone line, and recognized. A wrong read-back of the part in doubt is answered with the digits "
              "or letters; a right one with \"yes\"." +
              (" These answers were taken as said (text mode), so this is an upper bound." if mode == "text"
               else ""), ""]
    return lines
