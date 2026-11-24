"""Shared data types passed between agents, plugins, and the benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    PASS = "pass"          # value looks right, forward unchanged
    CORRECT = "correct"    # plugin replaced the value with a better-supported one
    CONFIRM = "confirm"    # plugin is unsure, ask the caller to confirm


@dataclass
class ShadowTranscript:
    """Independent transcript of the caller turn from a second recognizer.

    text: full transcript of the turn.
    confidence: mean recognizer confidence in [0, 1], or None if unknown.
    word_confidences: optional per-word confidences aligned with text.split().
    """
    text: str
    confidence: float | None = None
    word_confidences: list[float] | None = None


@dataclass
class TurnContext:
    """Everything a plugin may use when checking one caller turn."""
    shadow: ShadowTranscript | None = None
    roster: list[str] = field(default_factory=list)      # known names (bookings, CRM)
    known_values: dict[str, list[str]] = field(default_factory=dict)  # e.g. order ids
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class Verdict:
    """A plugin's decision about one tool-call argument."""
    field: str
    action: Action
    original: Any
    value: Any
    p_error: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "action": self.action.value,
            "original": self.original,
            "value": self.value,
            "p_error": round(self.p_error, 4),
            "reason": self.reason,
        }


@dataclass
class GuardResult:
    """Combined output of the plugin chain for one tool call."""
    tool_call: ToolCall            # possibly corrected arguments
    verdicts: list[Verdict]

    @property
    def needs_confirmation(self) -> bool:
        return any(v.action == Action.CONFIRM for v in self.verdicts)

    def tool_error_message(self) -> str | None:
        """Text returned to the agent as a tool error so it asks the caller.

        Returning a tool error works on every agent API, which is what makes the
        plugin usable on closed models without touching their weights.
        """
        asks = [v for v in self.verdicts if v.action == Action.CONFIRM]
        if not asks:
            return None
        parts = [f"{v.field}: {v.reason}" for v in asks]
        return (
            "VALIDATION_FAILED. Do not proceed. Read the value back to the caller "
            "slowly and ask them to confirm or correct it. Details: " + "; ".join(parts)
        )
