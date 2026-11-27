"""Layer 3: recovery. When a value is wrong, fix it without making the caller repeat everything.

Two strategies are compared on the same rows:

  targeted  ask only for the part in doubt. For a number, voxguard compares the tool value
            with the independent evidence (second recognizer, or the agent's own transcript)
            and asks for the digit groups where they disagree: "I have 719 209 3799. Could you
            say the last four digits again?" For a name, it asks the caller to spell only the
            part that disagrees, or the part that is missing.
  full      the usual fallback: "Could you repeat your number?" The caller says all of it.

The question is planned WITHOUT the true value. The true value is used only to simulate
what the caller answers: the digits or letters they were asked for, or "yes, that's right"
when the read-back part was already correct. The answer is then synthesized and run through
a real recognizer (see `voxguard recover`), so recovery can fail the same way the first
attempt did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..numbers.spoken import only_digits, words_to_digits
from ..numbers.validate import expected_length

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
WORDS_PER_SECOND = 2.5   # speaking rate used to estimate the agent's question time


@dataclass
class Ask:
    question: str
    reply: str               # what the simulated caller says
    start: int = 0           # numbers: span of the true value the caller re-says
    end: int = 0
    token: int = -1          # names: which name token is spelled (-1 = append a missing one)
    units: int = 0           # digits or letters the caller says
    confirms: bool = False   # caller only says "yes"


@dataclass
class Repair:
    strategy: str            # 'targeted' | 'full'
    field: str
    asks: list[Ask] = field(default_factory=list)
    total_units: int = 0

    @property
    def units(self) -> int:
        return sum(a.units for a in self.asks)

    def question_seconds(self) -> float:
        return sum(len(a.question.split()) for a in self.asks) / WORDS_PER_SECOND


def groups(n: int) -> list[tuple[int, int]]:
    """How people say an n-digit value: 3-3-4 for a phone number, otherwise threes."""
    if n == 10:
        return [(0, 3), (3, 6), (6, 10)]
    out, i = [], 0
    while i < n:
        j = min(n, i + 3)
        if n - j == 1:
            j = n
        out.append((i, j))
        i = j
    return out


def _common_prefix(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def _common_suffix(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def _say_digits(d: str) -> str:
    return " ".join(DIGIT_WORDS[int(c)] for c in d)


def _group_phrase(gs: list[tuple[int, int]], all_g: list[tuple[int, int]]) -> str:
    if len(gs) == len(all_g):
        return "the whole number"
    names = {0: "first", len(all_g) - 1: "last"}
    if len(all_g) == 3:
        names[1] = "middle"
    parts = []
    for g in gs:
        idx = all_g.index(g)
        parts.append(f"the {names.get(idx, f'group {idx + 1}')} {g[1] - g[0]} digits")
    return " and ".join(parts)


def suspect_groups(model: str, evidence: str | None, n: int) -> list[tuple[int, int]]:
    """Digit groups of an n-digit value where the tool value and the evidence disagree."""
    all_g = groups(n)
    if not evidence or not model:
        return all_g
    if len(model) == len(evidence) == n:
        diff = [i for i in range(n) if model[i] != evidence[i]]
    else:
        p, q = _common_prefix(model, evidence), _common_suffix(model, evidence)
        lo, hi = sorted((p, n - q))
        diff = list(range(max(0, lo), min(n, max(hi, lo + 1))))
    if not diff:
        return all_g
    return [g for g in all_g if any(g[0] <= i < g[1] for i in diff)]


def plan_repair(field: str, kind: str, model_value: str | None, evidence: str | None,
                expected: str, strategy: str = "targeted") -> Repair:
    """Plan the follow-up questions. `expected` only shapes the simulated caller's replies."""
    if kind in {"phone", "order_id", "card", "number"}:
        return _plan_number(field, kind, only_digits(str(model_value or "")),
                            only_digits(evidence) if evidence else None, only_digits(str(expected)), strategy)
    return _plan_name(field, str(model_value or "").strip(), (evidence or "").strip(), str(expected), strategy)


def _plan_number(field, kind, m, ev, truth, strategy) -> Repair:
    n = expected_length(kind) or len(truth) or len(ev or "") or len(m) or 1
    all_g = groups(n)
    gs = all_g if (strategy == "full" or not m) else suspect_groups(m, ev, n)
    start, end = gs[0][0], gs[-1][1]
    rep = Repair(strategy, field, total_units=n)
    if gs == all_g:
        q = "Sorry, could you say the whole number again, one digit at a time?"
        rep.asks.append(Ask(q, _say_digits(truth), 0, n, units=n))
        return rep
    readback = " ".join(m[a:b] for a, b in groups(len(m))) if m else ""
    q = f"I have {readback}. Could you say {_group_phrase(gs, all_g)} again?"
    # the caller compares the read-back part with what they said
    prefix_len, suffix_len = start, n - end
    asked_part_in_model = m[prefix_len:len(m) - suffix_len] if suffix_len else m[prefix_len:]
    if asked_part_in_model == truth[start:end] and len(m) == n:
        rep.asks.append(Ask(q, "Yes, that's right.", start, end, units=0, confirms=True))
    else:
        rep.asks.append(Ask(q, _say_digits(truth[start:end]), start, end, units=end - start))
    return rep


def _plan_name(field, m, ev, truth, strategy) -> Repair:
    toks, ev_toks, true_toks = m.split(), ev.split(), truth.split()
    rep = Repair(strategy, field, total_units=sum(len(t) for t in true_toks))
    label = ["first name", "last name"]

    def spell(i: int, append: bool = False) -> Ask:
        t = true_toks[i] if i < len(true_toks) else ""
        q = f"Could you spell your {label[min(i, 1)]} for me?"
        return Ask(q, ", ".join(t.upper()), token=-1 if append else i, units=len(t))

    n_parts = max(len(true_toks), len(toks), len(ev_toks), 1)
    if strategy == "full" or not toks:
        for i in range(n_parts):
            rep.asks.append(spell(i, append=i >= len(toks)))
        return rep
    targets = [i for i, t in enumerate(toks) if i < len(ev_toks) and t.lower() != ev_toks[i].lower()]
    if len(toks) < len(ev_toks):                       # a part of the name is missing
        targets += [-1]
    if not targets:                                    # nothing to localize: spell the whole name
        for i in range(n_parts):
            rep.asks.append(spell(i, append=i >= len(toks)))
        return rep
    for i in targets:
        if i == -1:
            rep.asks.append(spell(len(toks), append=True))
        else:
            rep.asks.append(spell(i))
    return rep


def parse_spelled(text: str) -> str:
    """'T, A, D, H, G.' or 'T-A-D-H-G' or 'Tadhg' -> 'Tadhg'."""
    filler = {"my", "name", "is", "it's", "its", "it", "spelled", "that's", "the", "last", "first", "yes",
              "sure", "so", "as", "in", "uh", "um", "okay", "ok"}
    raw = [w for w in text.replace("-", " ").replace(",", " ").replace(".", " ").split()
           if w.lower() not in filler or len(w) == 1]
    letters = [w for w in raw if len(w) == 1 and w.isalpha()]
    if letters and len(letters) >= len(raw) / 2:
        word = "".join(letters)
    else:
        word = max((w for w in raw if w.isalpha()), key=len, default="")
    return word[:1].upper() + word[1:].lower()


def apply_reply(rep: Repair, model_value: str | None, recognized: list[str], kind: str) -> str:
    """Merge the recognized answers into the value. One recognized string per Ask."""
    if kind in {"phone", "order_id", "card", "number"}:
        m = only_digits(str(model_value or ""))
        a, heard = rep.asks[0], recognized[0] if recognized else ""
        if a.confirms:
            return m
        digits = only_digits(words_to_digits(heard))
        n = rep.total_units
        if a.start == 0 and a.end == n:
            return digits
        suffix_len = n - a.end
        return m[:a.start] + digits + (m[len(m) - suffix_len:] if suffix_len else "")
    toks = str(model_value or "").split()
    for a, heard in zip(rep.asks, recognized, strict=False):
        word = parse_spelled(heard)
        if a.token == -1 or a.token >= len(toks):
            toks.append(word)
        else:
            toks[a.token] = word
    return " ".join(toks)
