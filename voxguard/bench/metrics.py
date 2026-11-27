"""Scoring. The headline metric is the SILENT ERROR RATE: a wrong value that
reaches the backend without anyone asking the caller. That is the failure that
books the table under the wrong name. A plugin that asks too often is penalized
by the false-confirmation rate, so both must be reported together.

Per field outcome:
  before: correct | wrong | missing | asked
          (asked = no value, but the agent asked the caller a question instead;
           an extra turn, not an error, the same as the plugin confirming)
  after:  correct              right value, passed through
          corrected            was wrong, plugin fixed it silently
          caught               was wrong, plugin asked the caller to confirm
          false_confirm        was right, plugin asked anyway (costs one turn)
          silent_error         was wrong, plugin did not catch it
          broken               was right, plugin changed it to something wrong
          missing              model produced no value; nothing to check
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def norm(kind: str, v) -> str:
    if v is None:
        return ""
    s = str(v)
    if kind in ("phone", "order_id"):
        d = re.sub(r"\D", "", s)
        return d[1:] if kind == "phone" and len(d) == 11 and d.startswith("1") else d
    return re.sub(r"\s+", " ", s).strip().casefold()


def field_kind(field: str) -> str:
    return {"phone": "phone", "order_id": "order_id"}.get(field, "name")


_ASK = re.compile(r"\?|\b(spell|repeat|say that again|confirm|one more time|didn'?t catch|could you)\b", re.I)


def agent_asked(text: str | None) -> bool:
    return bool(text) and bool(_ASK.search(str(text)))


def score_field(field: str, expected, before, after, action: str | None,
                agent_text: str | None = None) -> tuple[str, str]:
    k = field_kind(field)
    e, b, a = norm(k, expected), norm(k, before), norm(k, after)
    if not b:
        before_o = "asked" if agent_asked(agent_text) else "missing"
        if action == "confirm":          # voxguard noticed the caller said it and the tool left it out
            return before_o, "caught"
        return before_o, before_o
    before_o = "correct" if b == e else "wrong"
    if action == "confirm":
        return before_o, ("caught" if before_o == "wrong" else "false_confirm")
    if before_o == "correct":
        return before_o, ("correct" if a == e else "broken")
    return before_o, ("corrected" if a == e else "silent_error")


def summarize(rows: pd.DataFrame, by: str = "kind") -> pd.DataFrame:
    """One row per (model, group) with before/after rates. `by` is "kind" or "accent"."""
    if rows.empty:
        return pd.DataFrame()
    out = []
    for (model, kind), g in rows.groupby(["model", by]):
        n = len(g)
        bo, ao = g["before"], g["after"]
        before_err = ((bo == "wrong") | (bo == "missing")).mean()
        silent = ao.isin(["silent_error", "broken", "missing"]).mean()
        out.append({
            "model": model, by: kind, "n": n,
            "before_error": before_err,
            "after_silent_error": silent,
            "relative_reduction": (1 - silent / before_err) if before_err > 0 else np.nan,
            "corrected": (ao == "corrected").mean(),
            "caught": (ao == "caught").mean(),
            "false_confirm": (ao == "false_confirm").mean(),
            "broken": (ao == "broken").mean(),
            "missing": (bo == "missing").mean(),
            "agent_asked": (bo == "asked").mean(),
            "cut_in": g["cut_in"].mean() if "cut_in" in g else 0.0,
            "agent_p50_ms": g["agent_ms"].median(),
            "agent_p95_ms": g["agent_ms"].quantile(0.95),
            "guard_p95_ms": g["guard_ms"].quantile(0.95),
            "errors": int(g["agent_error"].notna().sum()),
        })
    return pd.DataFrame(out)


def overall(rows: pd.DataFrame, by: str = "kind") -> pd.DataFrame:
    r = rows.copy()
    r[by] = "all"
    return summarize(r, by)


def bootstrap_ci(rows: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """95% intervals for before error and after silent error per model.

    Resamples whole test cases (all accents and trials of a case move together),
    because rows from the same case are not independent.
    """
    rng = np.random.default_rng(seed)
    out = []
    for model, g in rows.groupby("model"):
        base = g["case_id"].str.split("__").str[0]
        per_case = g.assign(base=base,
                            b=((g["before"] == "wrong") | (g["before"] == "missing")).astype(float),
                            a=g["after"].isin(["silent_error", "broken", "missing"]).astype(float))
        agg = per_case.groupby("base")[["b", "a"]].agg(["sum", "count"])
        bs, bc = agg[("b", "sum")].to_numpy(), agg[("b", "count")].to_numpy()
        as_ = agg[("a", "sum")].to_numpy()
        idx = rng.integers(0, len(bs), size=(n_boot, len(bs)))
        b_rates = bs[idx].sum(1) / bc[idx].sum(1)
        a_rates = as_[idx].sum(1) / bc[idx].sum(1)
        out.append({"model": model,
                    "before_lo": np.quantile(b_rates, 0.025), "before_hi": np.quantile(b_rates, 0.975),
                    "after_lo": np.quantile(a_rates, 0.025), "after_hi": np.quantile(a_rates, 0.975)})
    return pd.DataFrame(out)
