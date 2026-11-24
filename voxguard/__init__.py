"""voxguard: training-free plugins that make voice agents reliable on numbers, names,
pronunciation, and turn-taking, plus a before/after benchmark for any model."""
from .core import Action, CostModel, PluginChain, ShadowTranscript, ToolCall, TurnContext, VoicePlugin
from .names import NameGuard
from .numbers import NumberGuard
from .pronunciation import PronunciationGuard
from .vad import EntityAwareEndpointer, FixedEndpointer

__version__ = "0.1.0"

__all__ = [
    "Action", "CostModel", "EntityAwareEndpointer", "FixedEndpointer", "NameGuard", "NumberGuard",
    "PluginChain", "PronunciationGuard", "ShadowTranscript", "ToolCall", "TurnContext", "VoicePlugin",
]
