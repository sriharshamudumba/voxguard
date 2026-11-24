"""Scoring. The headline metric is the SILENT ERROR RATE: a wrong value that
reaches the backend without anyone asking the caller. That is the failure that
books the table under the wrong name. A plugin that asks too often is penalized
by the false-confirmation rate, so both must be reported together.

Per field outcome:
  before: correct | wrong | missing
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


def score_field(field: str, expected, before, after, action: str | None) -> tuple[str, str]:
    k = field_kind(field)
    e, b, a = norm(k, expected), norm(k, before), norm(k, after)
    if not b:
        return "missing", "missing"
    before_o = "correct" if b == e else "wrong"
    if action == "confirm":
        return before_o, ("caught" if before_o == "wrong" else "false_confirm")
    if before_o == "correct":
        return before_o, ("correct" if a == e else "broken")
    return before_o, ("corrected" if a == e else "silent_error")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, kind) with before/after rates."""
    if rows.empty:
        return pd.DataFrame()
    out = []
    for (model, kind), g in rows.groupby(["model", "kind"]):
        n = len(g)
        bo, ao = g["before"], g["after"]
        before_err = ((bo == "wrong") | (bo == "missing")).mean()
        silent = ao.isin(["silent_error", "broken", "missing"]).mean()
        out.append({
            "model": model, "kind": kind, "n": n,
            "before_error": before_err,
            "after_silent_error": silent,
            "relative_reduction": (1 - silent / before_err) if before_err > 0 else np.nan,
            "corrected": (ao == "corrected").mean(),
            "caught": (ao == "caught").mean(),
            "false_confirm": (ao == "false_confirm").mean(),
            "broken": (ao == "broken").mean(),
            "agent_p50_ms": g["agent_ms"].median(),
            "agent_p95_ms": g["agent_ms"].quantile(0.95),
            "guard_p95_ms": g["guard_ms"].quantile(0.95),
            "errors": int(g["agent_error"].notna().sum()),
        })
    return pd.DataFrame(out)


def overall(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows.copy()
    r["kind"] = "all"
    return summarize(r)
