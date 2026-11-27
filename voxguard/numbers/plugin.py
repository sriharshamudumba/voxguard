"""NumberGuard: catches wrong, invented, or malformed digits in tool-call arguments.

For each configured numeric field it compares three sources of evidence:
  1. the agent's tool argument (from any model, cascade or speech-to-speech),
  2. an independent shadow transcript of the caller's audio,
  3. hard constraints (format validators, known database values).

It never fills in digits the caller did not say. Length padding by the model
(9 heard digits becoming 10) is treated as evidence of error, not something to
repair, because the posterior over an unheard digit is uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.decision import DEFAULT_COSTS, CostModel, combine_agreement
from ..core.plugin import VoicePlugin
from ..core.text import is_subsequence, levenshtein
from ..core.types import Action, ToolCall, TurnContext, Verdict
from .channel import DigitChannelDecoder
from .spoken import extract_digit_runs, only_digits, words_to_digits
from .validate import VALIDATORS, expected_length


@dataclass
class NumberGuard(VoicePlugin):
    fields: dict[str, str] = field(default_factory=lambda: {"phone": "phone"})
    # Prior error rates. Measure these on your own traffic with the benchmark.
    p_model_wrong: float = 0.12
    p_shadow_wrong_default: float = 0.08
    costs: dict[str, CostModel] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    decoder: DigitChannelDecoder = field(default_factory=DigitChannelDecoder)
    name: str = "numbers"

    # ---- helpers -------------------------------------------------------------
    def _shadow_digits(self, text: str, kind: str, model_digits: str) -> str | None:
        runs = extract_digit_runs(text)
        if not runs:
            return None
        n = expected_length(kind)
        joined = "".join(runs)
        candidates = set(runs) | {joined}
        # also consider contiguous concatenations (people pause between groups)
        for i in range(len(runs)):
            for j in range(i + 1, len(runs) + 1):
                candidates.add("".join(runs[i:j]))
        if n:
            exact = [c for c in candidates if len(c) == n]
            if exact:
                candidates = set(exact)
        return min(candidates, key=lambda c: (levenshtein(c, model_digits), -len(c)))

    # ---- main ----------------------------------------------------------------
    def check_tool_call(self, call: ToolCall, ctx: TurnContext) -> list[Verdict]:
        out: list[Verdict] = []
        for fname, kind in self.fields.items():
            if fname not in call.args or call.args[fname] in (None, ""):
                missing = self._check_missing(fname, kind, ctx, call)
                if missing:
                    out.append(missing)
                continue
            out.append(self._check_field(fname, kind, call.args[fname], ctx))
        return out

    def _evidence(self, ctx: TurnContext, kind: str) -> tuple[str | None, str | None]:
        """(digits in the agent's own transcript, digits the second recognizer heard)."""
        from ..layers.recognition import heard_digits
        n = expected_length(kind)
        heard = ctx.extra.get("heard")
        h = heard_digits(heard, n) if heard else None
        s = heard_digits(ctx.shadow.text, n) if ctx.shadow and ctx.shadow.text else None
        return h, s

    def _check_missing(self, fname: str, kind: str, ctx: TurnContext, call: ToolCall) -> Verdict | None:
        """Layer 2: the caller said a number but the tool call left the field out.

        Only for kinds with a fixed length (phone, card): a free-length ID cannot be told apart
        from another number in the same sentence."""
        n = expected_length(kind)
        if not n:
            return None
        valid = VALIDATORS.get(kind, VALIDATORS["number"])
        h, s = self._evidence(ctx, kind)
        used = {only_digits(str(v)) for v in call.args.values() if v}
        heard = next((d for d in (h, s) if d and len(d) == n and valid(d) and d not in used), None)
        if not heard:
            return None
        return Verdict(fname, Action.CONFIRM, None, None, 0.9,
                       f"the caller said a {kind.replace('_', ' ')} ({heard}) but the tool call left it out")

    def _check_field(self, fname: str, kind: str, raw, ctx: TurnContext) -> Verdict:
        """Models sometimes write digits as words ("eight eight eight"). Convert first;
        if the converted value then passes, return it as a correction."""
        spelled = any(ch.isalpha() for ch in str(raw))
        v = self._check_digits(fname, kind, words_to_digits(str(raw)) if spelled else raw, ctx)
        if spelled and v.action == Action.PASS:
            return Verdict(fname, Action.CORRECT, raw, v.value, v.p_error,
                           "number words converted to digits; " + v.reason)
        if spelled:
            v.original = raw
        return v

    def _check_digits(self, fname: str, kind: str, raw, ctx: TurnContext) -> Verdict:
        cost = self.costs.get(kind, self.costs["number"])
        valid = VALIDATORS.get(kind, VALIDATORS["number"])
        m = only_digits(raw)
        known = ctx.known_values.get(fname) or ctx.known_values.get(kind)

        shadow = ctx.shadow
        s = self._shadow_digits(shadow.text, kind, m) if shadow and shadow.text else None
        p_s = (1 - shadow.confidence) if (shadow and shadow.confidence is not None) else self.p_shadow_wrong_default

        if known and m in known:
            return Verdict(fname, Action.PASS, raw, raw, 0.01, "matches a known record")

        # Layer 2: the LLM had a transcript and turned it into digits. Converting spoken
        # numbers is a grammar problem, so redo it with code and compare.
        heard_text = ctx.extra.get("heard")
        if heard_text:
            h, s2 = self._evidence(ctx, kind)
            pipeline = ctx.extra.get("heard_source") == "pipeline"
            if h and h != m and valid(h) and (s2 == h or (pipeline and s2 is None)):
                why = "the second recognizer agrees" if s2 == h else "no other evidence"
                return Verdict(fname, Action.CORRECT, raw, h, 0.05,
                               f"the agent's transcript says {h} ({why}); the LLM wrote {m}")

        # No independent evidence: only format checks are possible.
        if s is None:
            if not valid(m):
                return Verdict(fname, Action.CONFIRM, raw, raw, 0.95,
                               f"'{m}' fails {kind} format check")
            p = self.p_model_wrong
            act = Action.CONFIRM if cost.should_confirm(p) and kind in {"card"} else Action.PASS
            return Verdict(fname, act, raw, raw, p, "no shadow transcript; format ok")

        # Model padded or invented digits relative to what was heard.
        if len(m) > len(s) and is_subsequence(s, m):
            return Verdict(fname, Action.CONFIRM, raw, raw, 0.9,
                           f"caller said {len(s)} digits ({s}) but tool got {len(m)} ({m}); "
                           "digits were added that the caller did not say")

        if m == s:
            p = combine_agreement(self.p_model_wrong, p_s, agree=True)
            if not valid(m):
                return Verdict(fname, Action.CONFIRM, raw, raw, 0.8,
                               f"both recognizers heard {m} but it fails {kind} format check")
            return Verdict(fname, Action.PASS, raw, raw, p, "model and shadow agree")

        # Disagreement. Use known records first, then the channel decoder, then the cost rule.
        if known:
            ranked = self.decoder.decode(s, valid=valid, known_values=known)
            if ranked and ranked[0][1] > 0.9 and len(ranked[0][0]) == len(s):
                return Verdict(fname, Action.CORRECT, raw, ranked[0][0], 1 - ranked[0][1],
                               f"shadow heard {s}; closest known record {ranked[0][0]}")

        model_ok, shadow_ok = valid(m), valid(s)
        p_err_model = combine_agreement(self.p_model_wrong, p_s, agree=False)
        if not model_ok and shadow_ok and p_s < 0.15:
            return Verdict(fname, Action.CORRECT, raw, s, p_s,
                           f"tool value {m} invalid; shadow heard valid {s}")
        if cost.should_confirm(p_err_model):
            return Verdict(fname, Action.CONFIRM, raw, raw, p_err_model,
                           f"caller audio sounds like {s} but tool got {m}")
        return Verdict(fname, Action.PASS, raw, raw, p_err_model, "minor disagreement below cost threshold")
