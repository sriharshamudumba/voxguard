"""Convert spoken number phrases to digit strings.

Handles the way people actually say numbers on the phone:
  "six seven eight, nine three two, oh two double eight"  -> 6789320288
  "eight hundred five five five twelve thirty four"        -> 8005551234
  "twenty three"                                           -> 23
"""
from __future__ import annotations

import re

ONES = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
TEENS = {
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
}
TENS = {
    "twenty": "2", "thirty": "3", "forty": "4", "fifty": "5",
    "sixty": "6", "seventy": "7", "eighty": "8", "ninety": "9",
}
REPEAT = {"double": 2, "triple": 3}
# Words that may sit inside a spoken number without ending it.
FILLERS = {"uh", "um", "er", "and", "dash", "hyphen", "point"}

_TOKEN = re.compile(r"[a-z]+|\d+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("-", " "))


def _is_num_token(t: str) -> bool:
    return t.isdigit() or t in ONES or t in TEENS or t in TENS or t in REPEAT or t in {"hundred", "thousand"}


def extract_digit_runs(text: str) -> list[str]:
    """Return each contiguous spoken number span as a digit string."""
    toks = _tokens(text)
    runs: list[str] = []
    cur: list[str] = []
    i = 0

    def flush() -> None:
        nonlocal cur
        if cur:
            runs.append("".join(cur))
            cur = []

    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        # "o"/"oh" only counts as zero when next to other number words
        if t in {"o", "oh"} and not (cur or (nxt and _is_num_token(nxt))):
            flush()
            i += 1
            continue
        if t.isdigit():
            cur.append(t)
        elif t in REPEAT and nxt is not None and (nxt in ONES or nxt.isdigit()):
            d = ONES.get(nxt, nxt)
            cur.append(d * REPEAT[t])
            i += 1
        elif t in ONES:
            if nxt == "hundred":
                d = ONES[t]
                rest = _two_digit_after(toks, i + 2)
                if rest is not None:
                    cur.append(d + rest[0])
                    i = rest[1] - 1
                else:
                    cur.append(d + "00")
                    i += 1
            elif nxt == "thousand":
                cur.append(ONES[t] + "000")
                i += 1
            else:
                cur.append(ONES[t])
        elif t in TEENS:
            cur.append(TEENS[t])
        elif t in TENS:
            if nxt in ONES and nxt not in {"zero", "oh", "o", "nought"}:
                cur.append(TENS[t] + ONES[nxt])
                i += 1
            else:
                cur.append(TENS[t] + "0")
        elif t in FILLERS and cur:
            pass  # keep the run open across "uh", "dash"
        else:
            flush()
        i += 1
    flush()
    return runs


def _two_digit_after(toks: list[str], j: int) -> tuple[str, int] | None:
    """After 'X hundred', read an optional 'and' plus a 1-2 digit number."""
    if j < len(toks) and toks[j] == "and":
        j += 1
    if j >= len(toks):
        return None
    t = toks[j]
    if t in TEENS:
        return TEENS[t], j + 1
    if t in TENS:
        if j + 1 < len(toks) and toks[j + 1] in ONES:
            return TENS[t] + ONES[toks[j + 1]], j + 2
        return TENS[t] + "0", j + 1
    return None


def words_to_digits(text: str) -> str:
    """All digits mentioned in the text, concatenated."""
    return "".join(extract_digit_runs(text))


def only_digits(value: object) -> str:
    return re.sub(r"\D", "", str(value))
