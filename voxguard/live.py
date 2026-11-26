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
from .names import NameGuard
from .numbers import NumberGuard

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
    guarded = chain.guard_tool_call(call, TurnContext(shadow=shadow, roster=roster, known_values=known))
    out = LiveResult(agent.name, dict(call.args), dict(guarded.tool_call.args),
                     {v.field: (v.action.value, v.reason) for v in guarded.verdicts},
                     res.text, res.latency_ms, bool(res.raw.get("cut_in", False)), res.error)
    for f, exp in truth.items():
        if not exp:
            continue
        kind = {"phone": "phone", "order_id": "order_id"}.get(f, "name")
        b_ok = norm(kind, out.before.get(f)) == norm(kind, exp)
        act = out.actions.get(f, ("pass", ""))[0]
        a_ok = act == "confirm" or norm(kind, out.after.get(f)) == norm(kind, exp)
        out.scores[f] = (b_ok, a_ok, act)
    return out


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
