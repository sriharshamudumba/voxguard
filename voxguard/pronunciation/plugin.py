"""PronunciationGuard: rewrites agent text into an unambiguous spoken form.

Works wherever you control the text that reaches TTS (cascade pipelines, or
speech-to-speech systems that let you route read-backs through your own TTS).
A pure speech-to-speech model speaks its own audio; for those, use
`instruction_hint()` in the system prompt, which is weaker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.plugin import VoicePlugin
from ..core.types import TurnContext
from .normalizer import normalize_for_tts


@dataclass
class PronunciationGuard(VoicePlugin):
    lexicon: dict[str, str] = field(default_factory=dict)  # "Siobhan" -> "Shivawn"
    name: str = "pronunciation"

    def on_text_out(self, text: str, ctx: TurnContext | None = None) -> str:
        return normalize_for_tts(text, self.lexicon)

    def instruction_hint(self) -> str:
        lines = [
            "When you say phone numbers, read digits one at a time in groups of three, three, four.",
            "Read order numbers and codes character by character.",
            "Say prices as words, for example twelve dollars and fifty cents.",
        ]
        for word, respell in self.lexicon.items():
            lines.append(f"Pronounce '{word}' as '{respell}'.")
        return " ".join(lines)
