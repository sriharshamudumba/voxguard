"""Spoken-form text normalization applied to agent text before TTS.

The LLM writes "678-932-0288", "$12.50", "7:30 PM", "AB12C". The TTS has to
guess how to read each. We remove the guess by writing the spoken form.
"""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + int_to_words(r))
    for div, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            q, r = divmod(n, div)
            return int_to_words(q) + " " + word + ("" if r == 0 else " " + int_to_words(r))
    return str(n)


def ordinal_words(n: int) -> str:
    w = int_to_words(n)
    last = w.split()[-1]
    if last in _ORD:
        new = _ORD[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return " ".join(w.split()[:-1] + [new])


def digits_spoken(d: str) -> str:
    return " ".join(_ONES[int(c)] for c in d)


_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})(?!\d)")
_MONEY = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?\b|\b(\d{1,2}):(\d{2})\b")
_EMAIL = re.compile(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_CODE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b|\b\d{5,}\b")
_NUM = re.compile(r"\b\d{1,9}\b")


def _phone(m: re.Match) -> str:
    a, b, c = m.groups()
    return f"{digits_spoken(a)}, {digits_spoken(b)}, {digits_spoken(c[:2])} {digits_spoken(c[2:])}"


def _money(m: re.Match) -> str:
    dollars = int(m.group(1).replace(",", ""))
    cents = int(m.group(2)) if m.group(2) else 0
    s = f"{int_to_words(dollars)} dollar{'s' if dollars != 1 else ''}"
    if cents:
        s += f" and {int_to_words(cents)} cent{'s' if cents != 1 else ''}"
    return s


def _time(m: re.Match) -> str:
    if m.group(1):
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        suffix = " a m" if ap == "a" else " p m"
    else:
        h, mi, suffix = int(m.group(4)), int(m.group(5)), ""
    if mi == 0:
        return f"{int_to_words(h)}{suffix}" if suffix else f"{int_to_words(h)} o'clock"
    mins = f"oh {int_to_words(mi)}" if mi < 10 else int_to_words(mi)
    return f"{int_to_words(h)} {mins}{suffix}"


def _email(m: re.Match) -> str:
    user, domain = m.group(1), m.group(2)
    user_s = user.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
    dom_s = domain.replace(".", " dot ")
    return f"{user_s} at {dom_s}"


def _code(m: re.Match) -> str:
    s = m.group(0)
    out = [(_ONES[int(c)] if c.isdigit() else c) for c in s]
    # group in threes so the listener can follow
    groups = [" ".join(out[i:i + 3]) for i in range(0, len(out), 3)]
    return ", ".join(groups)


def normalize_for_tts(text: str, lexicon: dict[str, str] | None = None) -> str:
    if lexicon:
        for word, respell in lexicon.items():
            text = re.sub(rf"\b{re.escape(word)}\b", respell, text, flags=re.IGNORECASE)
    text = _EMAIL.sub(_email, text)
    text = _PHONE.sub(_phone, text)
    text = _MONEY.sub(_money, text)
    text = _TIME.sub(_time, text)
    text = _ORDINAL.sub(lambda m: ordinal_words(int(m.group(1))), text)
    text = _CODE.sub(_code, text)
    text = _NUM.sub(lambda m: int_to_words(int(m.group(0))), text)
    return re.sub(r"\s{2,}", " ", text).strip()
