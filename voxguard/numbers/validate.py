"""Format validators. A failed check is a hard signal that the value is wrong."""
from __future__ import annotations

from collections.abc import Callable


def us_phone(d: str) -> bool:
    """NANP: 10 digits, area code and exchange cannot start with 0 or 1."""
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return len(d) == 10 and d[0] not in "01" and d[3] not in "01"


def luhn(d: str) -> bool:
    if not d.isdigit() or len(d) < 12:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def fixed_length(n: int) -> Callable[[str], bool]:
    return lambda d: len(d) == n and d.isdigit()


def any_digits(d: str) -> bool:
    return d.isdigit() and len(d) > 0


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "phone": us_phone,
    "card": luhn,
    "order_id": any_digits,
    "number": any_digits,
}


def expected_length(kind: str) -> int | None:
    return {"phone": 10, "card": 16}.get(kind)
