"""Compact phonetic key (Metaphone-style) plus combined name similarity.

Spelling-based keys cannot capture every orthography (Irish "Siobhan" sounds
like "shi-vawn"), so roster entries may carry respellings: "Siobhan|Shivawn".
"""
from __future__ import annotations

import re

from ..core.text import similarity

_VOWELS = set("aeiou")


def phonetic_key(name: str) -> str:
    w = re.sub(r"[^a-z]", "", name.lower())
    if not w:
        return ""
    for pre, rep in (("kn", "n"), ("gn", "n"), ("pn", "n"), ("wr", "r"), ("ps", "s"), ("x", "s")):
        if w.startswith(pre):
            w = rep + w[len(pre):]
            break
    rules = [
        ("sch", "sk"), ("tch", "x"), ("ph", "f"), ("sh", "x"), ("ch", "x"), ("th", "0"),
        ("ck", "k"), ("dg", "j"), ("gh", ""), ("wh", "w"), ("kh", "k"), ("zh", "j"),
        ("q", "k"), ("x", "ks"), ("z", "s"), ("v", "f"),
    ]
    for a, b in rules:
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = re.sub(r"g(?=[eiy])", "j", w)
    w = re.sub(r"y(?![aeiou])", "", w)
    first, rest = w[0], w[1:]
    rest = re.sub(r"[wh](?![aeiou])", "", rest)
    rest = "".join(ch for ch in rest if ch not in _VOWELS)
    key = first + rest
    return re.sub(r"(.)\1+", r"\1", key)


def split_aliases(entry: str) -> tuple[str, list[str]]:
    """'Siobhan|Shivawn' -> ('Siobhan', ['Siobhan', 'Shivawn'])."""
    parts = [p.strip() for p in entry.split("|") if p.strip()]
    return parts[0], parts


def name_similarity(a: str, b: str, w_phonetic: float = 0.6) -> float:
    """Weighted mix of spelling similarity and phonetic-key similarity, max over aliases of b."""
    canonical, aliases = split_aliases(b)
    best = 0.0
    for alias in aliases:
        spell = similarity(a.lower(), alias.lower())
        phon = similarity(phonetic_key(a), phonetic_key(alias))
        best = max(best, w_phonetic * phon + (1 - w_phonetic) * spell)
    return best
