"""Bayesian name resolution with a context prior.

    P(c | evidence) ∝ Π_e L(e | c) · P(c | context)

Likelihood of a piece of evidence e (model's tool arg, shadow transcript span)
given candidate c:  L(e | c) = exp(beta * (sim(e, c) - 1)).

Prior: alpha mass spread over roster names (tonight's bookings, CRM), and
1 - alpha on "someone not in the roster". That remaining mass is spread over
the whole space of possible names, so any one specific out-of-roster string
gets (1 - alpha) / open_vocab_size. The evidence strings stand in for that space.
alpha is the bias strength. Too high hallucinates roster names that were never
said, too low misses real ones; tune it on held-out data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .phonetic import name_similarity, split_aliases


@dataclass
class NameHypothesis:
    name: str
    posterior: float
    in_roster: bool


class NameResolver:
    def __init__(self, alpha: float = 0.7, beta: float = 8.0, open_vocab_size: float = 50.0):
        self.alpha = alpha
        self.beta = beta
        self.open_vocab_size = open_vocab_size

    def _lik(self, evidence: str, cand: str) -> float:
        return math.exp(self.beta * (name_similarity(evidence, cand) - 1.0))

    def resolve(self, evidence: list[str], roster: list[str]) -> list[NameHypothesis]:
        evidence = [e for e in evidence if e]
        if not evidence:
            return []
        cands: list[tuple[str, bool, float]] = []
        n = max(len(roster), 1)
        for entry in roster:
            cands.append((entry, True, self.alpha / n))
        # out-of-roster hypotheses: each distinct evidence string
        uniq = list(dict.fromkeys(evidence))
        for e in uniq:
            cands.append((e, False, (1 - self.alpha if roster else 1.0) / self.open_vocab_size))
        scores = []
        for entry, _in_roster, prior in cands:
            lik = 1.0
            for e in evidence:
                lik *= self._lik(e, entry)
            scores.append(prior * lik)
        z = sum(scores) or 1.0
        hyps = [NameHypothesis(split_aliases(c[0])[0], s / z, c[1]) for c, s in zip(cands, scores)]
        # merge duplicates (an evidence string equal to a roster name)
        merged: dict[str, NameHypothesis] = {}
        for h in hyps:
            k = h.name.lower()
            if k in merged:
                merged[k].posterior += h.posterior
                merged[k].in_roster = merged[k].in_roster or h.in_roster
            else:
                merged[k] = h
        return sorted(merged.values(), key=lambda h: -h.posterior)
