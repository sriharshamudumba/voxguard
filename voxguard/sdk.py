"""voxguard in a live agent: one Guard per deployment, one Call per phone call.

    from voxguard.sdk import Guard

    guard = Guard(roster=["Siobhan Murphy", "Fahad Rahman"], log_path="voxguard_calls.jsonl")
    call = guard.new_call("call-123")

    # as the caller speaks (any of these, whatever your stack gives you):
    call.add_transcript("my number is eight eight eight two seven two two zero twenty eight")
    call.add_audio(pcm16_bytes, 8000)            # only needed for the second recognizer

    # when the LLM calls a tool, before you run it:
    d = call.check("record_details", {"phone": "88827222028"})
    if d.action == "confirm":
        return d.say                             # give this back to the LLM as the tool result
    run_tool(d.args)                             # d.args may hold corrected values
    call.end_turn()

Everything is framework-agnostic: it needs the tool name, its arguments, and the caller's words
(and optionally the caller's audio). See docs/INTEGRATE.md for Pipecat, LiveKit, OpenAI Realtime,
and hosted platforms (Vapi, Retell) through `voxguard serve`.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .core.plugin import PluginChain
from .core.types import Action, ShadowTranscript, ToolCall, TurnContext
from .layers.recovery import plan_repair
from .names import NameGuard
from .numbers import NumberGuard
from .numbers.spoken import only_digits

NUMERIC_KINDS = {"phone", "order_id", "card", "number"}
DEFAULT_FIELDS = {"phone": "phone", "order_id": "order_id", "name": "name"}


@dataclass
class Decision:
    action: str                       # "pass" | "correct" | "confirm"
    args: dict                        # arguments to run the tool with (corrected where voxguard fixed them)
    say: str | None = None            # tool result to return to the LLM when action == "confirm"
    question: str | None = None       # the targeted question alone, for agents that speak it directly
    fields: dict = field(default_factory=dict)   # field -> {"action", "reason"}
    guard_ms: float = 0.0


def _redact(value) -> str | None:
    """Keep the shape, hide the content: phone 7192098799 -> ********99, name -> F**** R*****."""
    if value is None or value == "":
        return None
    s = str(value)
    if only_digits(s) and len(only_digits(s)) >= 4:
        d = only_digits(s)
        return "*" * (len(d) - 2) + d[-2:]
    return " ".join(w[:1] + "*" * (len(w) - 1) for w in s.split())


def _redact_text(text: str) -> str:
    return re.sub(r"\d", "*", text or "")


class Guard:
    """Configuration shared by every call: which tool fields to check, and what is known."""

    def __init__(self, fields: dict[str, str] | None = None, roster: list[str] | None = None,
                 known: dict[str, list[str]] | None = None, second_recognizer: dict | None = None,
                 transcript_source: str = "pipeline", log_path: str | None = None, redact: bool = True,
                 ask_unlisted_names: bool = True):
        """
        fields             tool argument -> kind ("phone", "order_id", "card", "number", "name")
        roster             names you expect (today's bookings, customer list); fixes misspelled names
        known              known values per field, e.g. {"order_id": [...open orders...]}
        second_recognizer  e.g. {"type": "deepgram", "model": "nova-3"}; needs add_audio(); adds ~0.3-0.6 s
                           at tool time. None = use the transcript only.
        transcript_source  "pipeline" if the transcript you pass is what your LLM reads (cascade);
                           "side_channel" if your model hears audio (OpenAI Realtime, Gemini Live)
        log_path           JSONL decision log, for measuring voxguard on real calls (`voxguard calls`)
        redact             mask digits and names in the log (recommended for real callers)
        ask_unlisted_names with a roster, ask callers whose name is not on it to spell it (True), or accept
                           the name as heard unless the audio contradicts it (False)
        """
        self.fields = dict(fields or DEFAULT_FIELDS)
        self.roster = list(roster or [])
        self.known = {k: list(v) for k, v in (known or {}).items()}
        self.second_recognizer = second_recognizer
        self.transcript_source = transcript_source
        self.log_path = Path(log_path) if log_path else None
        self.redact = redact
        self.ask_unlisted_names = ask_unlisted_names
        self._lock = threading.Lock()
        nums = {f: k for f, k in self.fields.items() if k in NUMERIC_KINDS}
        names = [f for f, k in self.fields.items() if k == "name"]
        plugins = []
        if nums:
            plugins.append(NumberGuard(fields=nums))
        if names:
            plugins.append(NameGuard(fields=names))
        self.chain = PluginChain(plugins)
        self._shadow = None

    def shadow(self):
        if self.second_recognizer and self._shadow is None:
            from .backends.registry import build_shadow
            self._shadow = build_shadow(self.second_recognizer)
        return self._shadow

    def new_call(self, call_id: str | None = None) -> Call:
        return Call(self, call_id or uuid.uuid4().hex[:12])

    def log(self, record: dict) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


class Call:
    """State for one phone call: the current turn's words and audio, and earlier decisions."""

    def __init__(self, guard: Guard, call_id: str):
        self.guard, self.id = guard, call_id
        self.turn = 0
        self._text: list[str] = []
        self._audio: list[np.ndarray] = []
        self._sr: int | None = None
        self._asked: dict[str, str] = {}      # field -> value we asked about (normalized)

    # ---- caller input
    def add_transcript(self, text: str) -> None:
        """Caller words for the current turn. Call it per final transcript segment."""
        if text and text.strip():
            self._text.append(text.strip())

    def add_audio(self, pcm, sr: int) -> None:
        """Caller audio for the current turn: int16 bytes or a float array. Only for the second recognizer."""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768 if isinstance(pcm, (bytes, bytearray)) \
            else np.asarray(pcm, dtype=np.float32)
        if self._sr and sr != self._sr:
            from .backends.audio import resample
            x = resample(x, sr, self._sr)
        else:
            self._sr = sr
        self._audio.append(x)

    def end_turn(self) -> None:
        """Call after the agent replies, so the next turn starts clean."""
        self.turn += 1
        self._text, self._audio = [], []

    # ---- the check
    def check(self, tool: str, args: dict) -> Decision:
        """Synchronous check. Inside an event loop, use `await call.acheck(...)` instead."""
        return asyncio.run(self.acheck(tool, args))

    async def acheck(self, tool: str, args: dict) -> Decision:
        t0 = time.perf_counter()
        g = self.guard
        heard = " | ".join(self._text) or None
        shadow = None
        if g.shadow() is not None and self._audio:
            from .backends.base import TurnInput
            audio = np.concatenate(self._audio)
            shadow = await g.shadow().transcribe(TurnInput(audio, self._sr or 8000, "", [], "", {}, 0, ""),
                                                 keyterms=g.roster or None)
        ctx = TurnContext(shadow=shadow if isinstance(shadow, ShadowTranscript) else None, roster=g.roster,
                          known_values=g.known, extra={"heard": heard, "heard_source": g.transcript_source})
        res = g.chain.guard_tool_call(ToolCall(tool, dict(args)), ctx)
        for v in res.verdicts:
            if v.action != Action.CONFIRM:
                continue
            if v.field in self._asked:
                # ask once per field per call: the caller has answered, so accept it (no question loops)
                v.action, v.reason = Action.PASS, "the caller was already asked about this; accepting their answer"
            elif (not g.ask_unlisted_names and g.fields.get(v.field) == "name"
                  and v.reason.startswith("uncertain name") and "audio sounds like" not in v.reason):
                v.action, v.reason = Action.PASS, "name not on the list; accepted as heard (ask_unlisted_names off)"
        fields = {v.field: {"action": v.action.value, "reason": v.reason} for v in res.verdicts}
        confirms = [v for v in res.verdicts if v.action == Action.CONFIRM]
        question = None
        if confirms:
            v = confirms[0]
            kind = g.fields.get(v.field, "name")
            evidence = None
            if kind in NUMERIC_KINDS:
                from .layers.recognition import heard_digits
                from .numbers.validate import expected_length
                for t in ([heard] if g.transcript_source == "pipeline" else []) + \
                         ([shadow.text] if shadow is not None and shadow.text else []):
                    evidence = heard_digits(t, expected_length(kind)) if t else None
                    if evidence:
                        break
            rep = plan_repair(v.field, kind, args.get(v.field), evidence, "", "targeted")
            question = " ".join(a.question for a in rep.asks) if rep.asks else \
                "Could you repeat that for me?"
            action = "confirm"
            say = (f"VALIDATION_FAILED: do not save yet. Ask the caller exactly this, then call {tool} "
                   f"again with what they say: \"{question}\"")
        else:
            action = "correct" if any(v.action == Action.CORRECT for v in res.verdicts) else "pass"
            say = None
        d = Decision(action, dict(res.tool_call.args), say, question, fields, (time.perf_counter() - t0) * 1000)
        self._log(tool, args, d, heard, shadow)
        return d

    def _log(self, tool: str, args: dict, d: Decision, heard: str | None, shadow) -> None:
        g = self.guard
        followups = {}
        for f, info in d.fields.items():
            value = only_digits(str(args.get(f, ""))) if g.fields.get(f) in NUMERIC_KINDS else \
                str(args.get(f, "")).strip().lower()
            if f in self._asked and info["action"] != "confirm":
                # the agent asked, the caller answered, the tool was called again
                followups[f] = "changed" if value != self._asked.pop(f) else "unchanged"
            if info["action"] == "confirm":
                self._asked[f] = value
        r = lambda v: _redact(v) if g.redact else v  # noqa: E731
        g.log({
            "ts": time.time(), "call": self.id, "turn": self.turn, "tool": tool, "action": d.action,
            "fields": d.fields, "followup": followups,
            "args_in": {k: r(v) for k, v in args.items()}, "args_out": {k: r(v) for k, v in d.args.items()},
            "heard": _redact_text(heard) if (g.redact and heard) else heard,
            "second": (_redact_text(shadow.text) if g.redact else shadow.text) if shadow is not None else None,
            "question": d.question, "guard_ms": round(d.guard_ms, 1),
        })


def summarize_log(path: str) -> str:
    """What voxguard did on real calls, from the decision log."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not rows:
        return f"no decisions in {path}"
    calls = {r["call"] for r in rows}
    per_field: dict[str, dict] = {}
    for r in rows:
        for f, info in r["fields"].items():
            s = per_field.setdefault(f, {"checked": 0, "pass": 0, "correct": 0, "confirm": 0,
                                         "asked_then_changed": 0, "asked_then_same": 0})
            s["checked"] += 1
            s[info["action"]] += 1
        for f, how in r.get("followup", {}).items():
            s = per_field.setdefault(f, {"checked": 0, "pass": 0, "correct": 0, "confirm": 0,
                                         "asked_then_changed": 0, "asked_then_same": 0})
            s["asked_then_changed" if how == "changed" else "asked_then_same"] += 1
    ms = sorted(r["guard_ms"] for r in rows)
    lines = [f"{len(calls)} calls, {len(rows)} tool calls checked, guard p95 {ms[int(0.95 * (len(ms) - 1))]:.0f} ms",
             "", f"{'field':<12}{'checked':>8}{'passed':>8}{'fixed':>7}{'asked':>7}{'asked: caller changed it':>26}"
             f"{'asked: same value':>19}"]
    for f, s in per_field.items():
        lines.append(f"{f:<12}{s['checked']:>8}{s['pass']:>8}{s['correct']:>7}{s['confirm']:>7}"
                     f"{s['asked_then_changed']:>26}{s['asked_then_same']:>19}")
    lines += ["", "'asked: caller changed it' = voxguard asked and the value changed after the caller answered: "
              "an error caught.", "'asked: same value' = the caller confirmed what the model had: an unnecessary "
              "question.", "'fixed' cannot be verified without listening to the call; spot-check a few."]
    return "\n".join(lines)


def decision_dict(d: Decision) -> dict:
    return asdict(d)
