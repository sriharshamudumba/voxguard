"""Benchmark cases: one caller turn each, with the exact tool arguments expected.

Kinds:
  phone      caller gives a 10-digit US number, in the varied ways people say them
  order_id   6 to 8 digit order number
  name       caller gives a name that IS in tonight's roster (context prior helps)
  name_new   first-time caller, name NOT in the roster (prior cannot help)
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..pronunciation.normalizer import int_to_words
from .names_data import FIRST_TIME_NAMES, NAMES

SYSTEM_PROMPT = (
    "You are the phone agent for a restaurant. As soon as the caller gives their name, phone number, "
    "or order number, call record_details with exactly what they said. Never guess or fill in missing "
    "digits. Do not ask follow-up questions in this turn."
)

TOOLS = [{
    "name": "record_details",
    "description": "Record caller details exactly as spoken.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Caller full name"},
            "phone": {"type": "string", "description": "Caller phone number, digits only"},
            "order_id": {"type": "string", "description": "Order number, digits only"},
        },
    },
}]

KINDS = ("phone", "order_id", "name", "name_new")
_W = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


@dataclass
class Case:
    id: str
    kind: str
    caller_text: str
    expected: dict[str, str]
    roster: list[str] = field(default_factory=list)
    known_values: dict[str, list[str]] = field(default_factory=dict)
    tts_text: str = ""          # text fed to TTS (may use respellings)
    audio_path: str | None = None
    seed: int = 0
    accent: str = ""            # set by synthesis; one case per accent


def _random_phone(rng: random.Random) -> str:
    area = str(rng.randint(2, 9)) + f"{rng.randint(0, 99):02d}"
    if rng.random() < 0.15:
        area = rng.choice(["800", "888", "877"])
    exch = str(rng.randint(2, 9)) + f"{rng.randint(0, 99):02d}"
    line = f"{rng.randint(0, 9999):04d}"
    if rng.random() < 0.3:  # repeated digits are a known failure mode
        d = str(rng.randint(0, 9))
        line = line[:2] + d + d
    return area + exch + line


def _say_group(g: str, rng: random.Random, allow_pairs: bool) -> str:
    out, i = [], 0
    while i < len(g):
        if i + 1 < len(g) and g[i] == g[i + 1] and rng.random() < 0.6:
            out.append("double " + _W[int(g[i])])
            i += 2
            continue
        if allow_pairs and i + 1 < len(g) and g[i] != "0" and rng.random() < 0.35:
            out.append(int_to_words(int(g[i:i + 2])))
            i += 2
            continue
        d = g[i]
        out.append("oh" if d == "0" and rng.random() < 0.5 else _W[int(d)])
        i += 1
    return " ".join(out)


def say_phone(p: str, rng: random.Random) -> str:
    a, b, c = p[:3], p[3:6], p[6:]
    if a in {"800", "888", "877"} and rng.random() < 0.7:
        sa = _W[int(a[0])] + " " + ("hundred" if a == "800" else _W[int(a[1])] + " " + _W[int(a[2])])
    else:
        sa = _say_group(a, rng, False)
    return f"{sa}, {_say_group(b, rng, False)}, {_say_group(c, rng, True)}"


def make_case(kind: str, i: int, rng: random.Random) -> Case:
    cid = f"{kind}-{i:04d}"
    if kind == "phone":
        p = _random_phone(rng)
        lead = rng.choice(["Yeah my number is", "Sure, it's", "You can reach me at", "My cell is"])
        text = f"{lead} {say_phone(p, rng)}."
        return Case(cid, kind, text, {"phone": p}, tts_text=text, seed=rng.randint(0, 10**9))
    if kind == "order_id":
        n = rng.randint(6, 8)
        oid = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))
        spoken = ", ".join(_say_group(oid[j:j + 3], rng, True) for j in range(0, len(oid), 3))
        text = f"{rng.choice(['My order number is', 'The order is', 'It is order'])} {spoken}."
        decoys = ["".join(str(rng.randint(0, 9)) for _ in range(len(oid))) for _ in range(5)]
        return Case(cid, kind, text, {"order_id": oid}, known_values={"order_id": [oid] + decoys},
                    tts_text=text, seed=rng.randint(0, 10**9))
    pool = NAMES if kind == "name" else FIRST_TIME_NAMES
    name = rng.choice(list(pool))
    _, respell = pool[name]
    lead = rng.choice(["Hi, I have a reservation under", "The name is", "It's under", "This is"])
    text = f"{lead} {name}."
    roster = rng.sample([n for n in NAMES if n != name], 9)
    if kind == "name":
        roster.append(name)
        rng.shuffle(roster)
    tts = f"{lead} {respell or name}."
    return Case(cid, kind, text, {"name": name}, roster=roster, tts_text=tts, seed=rng.randint(0, 10**9))


def generate_cases(n_per_kind: int = 50, kinds=KINDS, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    return [make_case(k, i, rng) for k in kinds for i in range(n_per_kind)]


def save_cases(cases: list[Case], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(asdict(c)) + "\n")


def load_cases(path: str | Path) -> list[Case]:
    with open(path) as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def expand_accents(cases: list[Case], accents: list[str]) -> list[Case]:
    """One copy of every case per accent. Same text and expected values, different speaker."""
    out = []
    for c in cases:
        if c.accent:            # already expanded
            out.append(c)
            continue
        for a in accents:
            d = asdict(c)
            d.update(id=f"{c.id}__{a}", accent=a, audio_path=None)
            out.append(Case(**d))
    return out
