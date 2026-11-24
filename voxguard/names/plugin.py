"""NameGuard: resolves caller names against known context and flags uncertain ones."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.decision import DEFAULT_COSTS, CostModel
from ..core.plugin import VoicePlugin
from ..core.types import Action, ToolCall, TurnContext, Verdict
from .phonetic import name_similarity
from .resolver import NameResolver

_STOP = {
    "my", "name", "is", "it's", "its", "this", "the", "a", "reservation", "for", "under",
    "booking", "i", "am", "i'm", "yes", "hi", "hello", "it", "table", "please", "and", "last",
    "first", "spelled", "so", "uh", "um", "called", "call", "me",
}


def candidate_spans(text: str, max_len: int = 3) -> list[str]:
    toks = [t for t in re.findall(r"[A-Za-z'][A-Za-z'\-]*", text)]
    spans = []
    for n in range(1, max_len + 1):
        for i in range(len(toks) - n + 1):
            span = toks[i:i + n]
            if any(t.lower() in _STOP for t in span):
                continue
            spans.append(" ".join(span))
    return spans


@dataclass
class NameGuard(VoicePlugin):
    fields: list[str] = field(default_factory=lambda: ["name"])
    resolver: NameResolver = field(default_factory=NameResolver)
    cost: CostModel = field(default_factory=lambda: DEFAULT_COSTS["name"])
    name: str = "names"

    def _best_span(self, text: str, target: str, roster: list[str]) -> str | None:
        spans = candidate_spans(text)
        if not spans:
            return None
        refs = [target] + roster

        def score(s: str) -> float:
            return max(name_similarity(s, r) for r in refs)

        return max(spans, key=score)

    def check_tool_call(self, call: ToolCall, ctx: TurnContext) -> list[Verdict]:
        out = []
        for f in self.fields:
            raw = call.args.get(f)
            if not raw:
                continue
            m = str(raw).strip()
            shadow_span = None
            if ctx.shadow and ctx.shadow.text:
                shadow_span = self._best_span(ctx.shadow.text, m, ctx.roster)
            evidence = [m] + ([shadow_span] if shadow_span else [])
            hyps = self.resolver.resolve(evidence, ctx.roster)
            if not hyps:
                continue
            best = hyps[0]
            p_err = 1 - best.posterior
            same = best.name.lower() == m.lower()
            if best.in_roster and not self.cost.should_confirm(p_err):
                if same:
                    out.append(Verdict(f, Action.PASS, raw, raw, p_err, "matches roster"))
                else:
                    out.append(Verdict(f, Action.CORRECT, raw, best.name, p_err,
                                       f"'{m}' resolved to roster name '{best.name}'"))
                continue
            if self.cost.should_confirm(p_err):
                heard = f" (audio sounds like '{shadow_span}')" if shadow_span else ""
                out.append(Verdict(f, Action.CONFIRM, raw, raw, p_err,
                                   f"uncertain name '{m}'{heard}; ask the caller to spell it"))
            else:
                out.append(Verdict(f, Action.PASS, raw, raw, p_err, "confident"))
        return out
