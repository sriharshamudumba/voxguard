"""Backend interfaces.

AgentBackend: the voice agent under test. Given one caller turn (audio plus
the case's prompt and tools) it returns the tool calls it made.

ShadowASR: an independent recognizer whose transcript the plugins use as
second evidence. It runs in parallel with the agent, so it is off the
critical path for latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.types import ShadowTranscript, ToolCall


@dataclass
class AgentResult:
    tool_calls: list[ToolCall]
    text: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TurnInput:
    audio: np.ndarray | None          # float32 mono in [-1, 1]
    sr: int
    system_prompt: str
    tools: list[dict]                  # JSON schema function definitions
    caller_text: str = ""              # ground-truth text; only the mock backends read it
    truth: dict[str, Any] = field(default_factory=dict)   # expected args; only mock backends read it
    seed: int = 0
    accent: str = ""                   # only mock backends read it


class AgentBackend:
    name: str = "agent"
    kind: str = "unknown"              # "sts", "cascade", "audio_llm", "mock"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class ShadowASR:
    name: str = "shadow"

    async def transcribe(self, turn: TurnInput, keyterms: list[str] | None = None) -> ShadowTranscript:
        raise NotImplementedError
