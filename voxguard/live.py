"""Live comparison: one recorded caller turn, several models, before and after the plugins.

Used by the UI's Live tab. Kept separate from the UI so it can be tested and
reused from scripts.

There is no ground truth for a live recording, so the output shows what each
model recorded, what the plugins did with it, and why. If the speaker types
what they actually said, each value is also scored.
"""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field

import numpy as np

from .backends.audio import resample, telephony_channel
from .backends.base import TurnInput
from .backends.registry import build_agent, build_shadow
from .bench.dataset import SYSTEM_PROMPT, TOOLS
from .bench.metrics import agent_asked, norm
from .core.plugin import PluginChain
from .core.types import ShadowTranscript, ToolCall, TurnContext
from .layers import Repair, apply_reply, attribute, heard_digits, plan_repair, recognize_entity
from .names import NameGuard
from .numbers import NumberGuard
from .numbers.validate import expected_length

FIELDS = ("name", "phone", "order_id")


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Any PCM WAV (int16, int32, float) to float32 mono."""
    from scipy.io import wavfile
    sr, x = wavfile.read(io.BytesIO(data))
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float32), int(sr)


@dataclass
class LiveResult:
    model: str
    before: dict
    after: dict
    actions: dict = field(default_factory=dict)      # field -> (action, reason)
    agent_text: str = ""
    latency_ms: float = float("nan")
    cut_in: bool = False
    error: str | None = None
    scores: dict = field(default_factory=dict)       # field -> (before_ok, after_ok) when truth given
    heard: str | None = None                         # Layer 1: the caller transcript the platform sent
    heard_source: str = "none"                       # pipeline | side_channel | none
    layer1: dict = field(default_factory=dict)       # field -> recognition outcome (when truth given)
    layer2: dict = field(default_factory=dict)       # field -> which layer failed (when truth given)
    repairs: dict = field(default_factory=dict)      # field -> targeted Repair, for confirmations


async def _run_one(cfg: dict, turn: TurnInput, shadow_task: asyncio.Task, chain: PluginChain,
                   roster: list[str], known: dict, truth: dict) -> LiveResult:
    t0 = time.perf_counter()
    try:
        agent = build_agent(cfg)
        res = await agent.run_turn(turn)
    except Exception as e:  # a failing provider must not break the comparison
        return LiveResult(cfg.get("name") or cfg["type"], {}, {}, error=repr(e),
                          latency_ms=(time.perf_counter() - t0) * 1000)
    shadow = await shadow_task
    call = next((c for c in res.tool_calls if c.name == "record_details"), None) or \
        (res.tool_calls[0] if res.tool_calls else ToolCall("record_details", {}))
    heard = " | ".join(h for h in (res.raw.get("heard") or []) if h) or None
    source = getattr(agent, "heard_source", "none") if heard else "none"
    ctx = TurnContext(shadow=shadow, roster=roster, known_values=known,
                      extra={"heard": heard, "heard_source": source})
    guarded = chain.guard_tool_call(call, ctx)
    out = LiveResult(agent.name, dict(call.args), dict(guarded.tool_call.args),
                     {v.field: (v.action.value, v.reason) for v in guarded.verdicts},
                     res.text, res.latency_ms, bool(res.raw.get("cut_in", False)), res.error,
                     heard=heard, heard_source=source)
    for v in guarded.verdicts:
        if v.action.value == "confirm":
            kind = {"phone": "phone", "order_id": "order_id"}.get(v.field, "name")
            ev = _evidence_for(v.field, kind, out.before.get(v.field), heard, source, shadow)
            out.repairs[v.field] = plan_repair(v.field, kind, out.before.get(v.field), ev, "", "targeted")
    for f, exp in truth.items():
        if not exp:
            continue
        kind = {"phone": "phone", "order_id": "order_id"}.get(f, "name")
        b_ok = norm(kind, out.before.get(f)) == norm(kind, exp)
        act = out.actions.get(f, ("pass", ""))[0]
        a_ok = act == "confirm" or norm(kind, out.after.get(f)) == norm(kind, exp)
        out.scores[f] = (b_ok, a_ok, act)
        out.layer1[f] = recognize_entity(f, exp, heard)
        out.layer2[f] = attribute("correct" if b_ok else ("wrong" if out.before.get(f) else "missing"),
                                  out.layer1[f], out.cut_in)
    return out


def _evidence_for(field: str, kind: str, before, heard: str | None, source: str,
                  shadow: ShadowTranscript | None) -> str | None:
    texts = ([heard] if heard and source == "pipeline" else []) + ([shadow.text] if shadow and shadow.text else [])
    if kind in {"phone", "order_id"}:
        for t in texts:
            d = heard_digits(t, expected_length(kind))
            if d:
                return d
        return None
    if not texts or not before:
        return None
    from .names.plugin import NameGuard
    return NameGuard()._best_span(texts[0], str(before), []) or None


async def answer_repair(repair: Repair, before, audio: np.ndarray, sr: int, shadow_cfg: dict,
                        phone_line: bool = True) -> tuple[str, str]:
    """Layer 3, live: recognize the caller's spoken answer and merge it. Returns (heard, final value)."""
    shadow = build_shadow(shadow_cfg)
    if phone_line:
        audio, sr = telephony_channel(audio, sr, seed=7)
    st = await shadow.transcribe(TurnInput(audio, sr, "", [], "", {}, 0, ""))
    kind = {"phone": "phone", "order_id": "order_id"}.get(repair.field, "name")
    heard = st.text or ""
    parts = [p for p in heard.replace(" / ", "|").split("|")] if len(repair.asks) > 1 else [heard]
    if len(repair.asks) > 1 and len(parts) < len(repair.asks):
        words = heard.split()                     # one answer for several parts: split evenly
        k = max(1, len(words) // len(repair.asks))
        parts = [" ".join(words[i * k:(i + 1) * k if i < len(repair.asks) - 1 else None])
                 for i in range(len(repair.asks))]
    return heard, apply_reply(repair, before, parts, kind)


async def compare(audio: np.ndarray, sr: int, agent_cfgs: list[dict], shadow_cfg: dict,
                  roster: list[str] | None = None, known_order_ids: list[str] | None = None,
                  phone_line: bool = True, truth: dict | None = None
                  ) -> tuple[list[LiveResult], ShadowTranscript, tuple[np.ndarray, int]]:
    """Run every model on the same turn, concurrently, and apply the plugins to each.

    Returns per-model results, the shadow transcript, and the audio the models heard.
    """
    if phone_line:
        audio, sr = telephony_channel(audio, sr, snr_db=None, packet_loss=0.0)
    else:
        audio, sr = resample(audio, sr, 16000), 16000
    # truth is only read by the simulated mock backends; real models never see it
    turn = TurnInput(audio, sr, SYSTEM_PROMPT, TOOLS, truth={k: v for k, v in (truth or {}).items() if v})
    roster = [r.strip() for r in (roster or []) if r.strip()]
    known = {"order_id": [k.strip() for k in (known_order_ids or []) if k.strip()]}
    chain = PluginChain([NumberGuard(fields={"phone": "phone", "order_id": "order_id"}), NameGuard()])
    shadow = build_shadow(shadow_cfg)

    async def run_shadow() -> ShadowTranscript:
        try:
            return await shadow.transcribe(turn, roster or None)
        except Exception:
            return ShadowTranscript("", None)

    shadow_task = asyncio.create_task(run_shadow())
    results = await asyncio.gather(*(_run_one(c, turn, shadow_task, chain, roster, known, truth or {})
                                     for c in agent_cfgs))
    return list(results), await shadow_task, (audio, sr)


def asked(result: LiveResult) -> bool:
    return not any(result.before.get(f) for f in FIELDS) and agent_asked(result.agent_text)
