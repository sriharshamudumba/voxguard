"""Noisy-channel decoding for digit strings.

Model: the caller says digit d, the phone line plus recognizer outputs h with
probability C[d, h]. Given heard digits (and optional per-digit confidences),
the posterior over the spoken digit at each position is

    P(d | evidence) ∝ c * 1[d == h] + (1 - c) * C[d, h]

and a full string is scored by the sum of log posteriors. Candidates are
restricted to a valid set (format check or known database values).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product

import numpy as np

# Telephony-style confusions. Replace with one estimated on your own audio via fit_confusion().
_CONFUSABLE = [(5, 9), (1, 9), (0, 4), (2, 3), (6, 7), (8, 6)]


def default_confusion(eps: float = 0.06, pair_weight: float = 20.0) -> np.ndarray:
    """Row d is P(heard | spoken d). Off-diagonal mass eps, concentrated on confusable pairs."""
    W = np.ones((10, 10))
    for a, b in _CONFUSABLE:
        W[a, b] += pair_weight
        W[b, a] += pair_weight
    np.fill_diagonal(W, 0)
    C = eps * W / W.sum(axis=1, keepdims=True)
    np.fill_diagonal(C, 1 - eps)
    return C


def fit_confusion(pairs: Iterable[tuple[str, str]], alpha: float = 1.0) -> np.ndarray:
    """Estimate C[d, h] from aligned (spoken, heard) equal-length digit strings.

    alpha is Laplace smoothing so unseen confusions keep a small probability.
    """
    counts = np.full((10, 10), alpha)
    for spoken, heard in pairs:
        if len(spoken) != len(heard):
            continue
        for d, h in zip(spoken, heard):
            counts[int(d), int(h)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


class DigitChannelDecoder:
    def __init__(self, confusion: np.ndarray | None = None, top_k: int = 2):
        self.C = default_confusion() if confusion is None else confusion
        self.top_k = top_k

    def position_posteriors(self, heard: str, conf: list[float] | None = None) -> np.ndarray:
        conf = conf or [0.5] * len(heard)
        post = np.zeros((len(heard), 10))
        for i, (h, c) in enumerate(zip(heard, conf)):
            hi = int(h)
            col = self.C[:, hi] / self.C[:, hi].sum()   # P(d | h) with uniform prior over d
            onehot = np.zeros(10)
            onehot[hi] = 1.0
            post[i] = c * onehot + (1 - c) * col
        return post / post.sum(axis=1, keepdims=True)

    def score(self, candidate: str, post: np.ndarray) -> float:
        if len(candidate) != len(post):
            return -np.inf
        return float(sum(np.log(post[i, int(d)] + 1e-12) for i, d in enumerate(candidate)))

    def decode(
        self,
        heard: str,
        conf: list[float] | None = None,
        valid: Callable[[str], bool] | None = None,
        known_values: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Ranked (candidate, posterior) pairs, normalized over the candidate set."""
        post = self.position_posteriors(heard, conf)
        if known_values:
            cands = [k for k in known_values if len(k) == len(heard)]
        else:
            best = [np.argsort(-post[i])[: self.top_k] for i in range(len(heard))]
            # beam is top_k^len; cap work by expanding only the 8 least certain positions
            uncertain = sorted(range(len(heard)), key=lambda i: post[i].max())[:8]
            base = [str(int(np.argmax(post[i]))) for i in range(len(heard))]
            cands = []
            for combo in product(*[best[i] for i in uncertain[:8]]):
                s = list(base)
                for i, d in zip(uncertain[:8], combo):
                    s[i] = str(int(d))
                cands.append("".join(s))
        if valid is not None:
            cands = [c for c in cands if valid(c)]
        if not cands:
            return []
        scores = np.array([self.score(c, post) for c in cands])
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(-probs)
        return [(cands[i], float(probs[i])) for i in order]
