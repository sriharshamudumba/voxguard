"""Plugin interface and chain.

A plugin never touches model weights. It observes the caller audio/transcript,
the agent's tool calls, and the agent's outgoing text, and can:
  * correct a tool-call argument,
  * ask for confirmation (returned to the agent as a tool error),
  * rewrite outgoing text before TTS,
  * decide when a caller turn has ended.
"""
from __future__ import annotations

from collections.abc import Iterable

from .types import Action, GuardResult, ToolCall, TurnContext, Verdict


class VoicePlugin:
    name: str = "plugin"

    def check_tool_call(self, call: ToolCall, ctx: TurnContext) -> list[Verdict]:
        """Return verdicts for the fields this plugin is responsible for."""
        return []

    def on_text_out(self, text: str, ctx: TurnContext | None = None) -> str:
        """Rewrite agent text before it is spoken (cascade pipelines only)."""
        return text


class PluginChain:
    """Runs plugins in order. Later plugins see earlier corrections."""

    def __init__(self, plugins: Iterable[VoicePlugin]):
        self.plugins = list(plugins)

    def guard_tool_call(self, call: ToolCall, ctx: TurnContext) -> GuardResult:
        args = dict(call.args)
        verdicts: list[Verdict] = []
        for p in self.plugins:
            for v in p.check_tool_call(ToolCall(call.name, args), ctx):
                verdicts.append(v)
                if v.action == Action.CORRECT:
                    args[v.field] = v.value
        return GuardResult(ToolCall(call.name, args), verdicts)

    def on_text_out(self, text: str, ctx: TurnContext | None = None) -> str:
        for p in self.plugins:
            text = p.on_text_out(text, ctx)
        return text
