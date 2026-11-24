"""Decode-time plugins for open-weight models (no training).

Compatible with Hugging Face `generate(logits_processor=[...])`. The same
callables can be adapted to vLLM's per-request logits processors.

TrieBiasProcessor   shallow fusion: score = log p_model + lambda * 1[token continues an entity]
DigitCopyProcessor  grounded copying: inside a digit span, only allow the next
                    digit the shadow recognizer heard, then force the span to end.
                    This removes invented and padded digits by construction.
"""
from __future__ import annotations

from collections.abc import Iterable

try:
    import torch
except ImportError:  # torch is optional; only needed for open-model decoding
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Install torch to use logits processors: pip install 'voxguard[open]'")


class TrieBiasProcessor:
    def __init__(self, entity_token_seqs: Iterable[list[int]], lam: float = 2.5, max_suffix: int = 8):
        _require_torch()
        self.lam = lam
        self.max_suffix = max_suffix
        self.trie: dict = {}
        for seq in entity_token_seqs:
            node = self.trie
            for t in seq:
                node = node.setdefault(t, {})

    def _next_tokens(self, ids: list[int]) -> list[int]:
        nxt: set[int] = set(self.trie.keys())  # any entity may start here
        for start in range(max(0, len(ids) - self.max_suffix), len(ids)):
            node = self.trie
            ok = True
            for t in ids[start:]:
                if t not in node:
                    ok = False
                    break
                node = node[t]
            if ok:
                nxt |= set(node.keys())
        return list(nxt)

    def __call__(self, input_ids, scores):
        for b in range(input_ids.size(0)):
            nxt = self._next_tokens(input_ids[b].tolist())
            if nxt:
                scores[b, nxt] += self.lam
        return scores


class DigitCopyProcessor:
    """Constrain digit generation to the digits actually heard.

    digit_token_ids: token id for each character '0'..'9' (index = digit).
    heard: digit string from the shadow recognizer.
    Once the model emits its first digit, the next allowed digit is heard[k];
    after len(heard) digits, all digit tokens are masked, so it cannot pad.
    Assumes the tokenizer emits single-digit tokens (true for Llama 3, Qwen 2.5+).
    """

    def __init__(self, digit_token_ids: list[int], heard: str, allow_separators: Iterable[int] = ()):
        _require_torch()
        self.digit_ids = digit_token_ids
        self.id_to_digit = {tid: str(i) for i, tid in enumerate(digit_token_ids)}
        self.heard = heard
        self.seps = set(allow_separators)

    def _emitted_digits(self, ids: list[int]) -> int:
        # count digits in the current contiguous digit span (separators allowed inside)
        n = 0
        for t in reversed(ids):
            if t in self.id_to_digit:
                n += 1
            elif t in self.seps and n > 0:
                continue
            else:
                break
        return n

    def __call__(self, input_ids, scores):
        neg = torch.finfo(scores.dtype).min
        for b in range(input_ids.size(0)):
            ids = input_ids[b].tolist()
            k = self._emitted_digits(ids)
            if k == 0:
                # a new span may only start with the first heard digit
                if self.heard:
                    first = self.digit_ids[int(self.heard[0])]
                    mask = [t for t in self.digit_ids if t != first]
                else:
                    mask = self.digit_ids
                scores[b, mask] = neg
            elif k < len(self.heard):
                allowed = self.digit_ids[int(self.heard[k])]
                mask = [t for t in self.digit_ids if t != allowed]
                scores[b, mask] = neg
            else:
                scores[b, self.digit_ids] = neg
        return scores


def digit_token_ids(tokenizer) -> list[int]:
    ids = []
    for d in "0123456789":
        toks = tokenizer.encode(d, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"tokenizer splits digit {d!r} into {toks}; DigitCopyProcessor needs single-digit tokens")
        ids.append(toks[0])
    return ids
