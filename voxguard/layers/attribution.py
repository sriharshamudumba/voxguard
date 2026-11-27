"""Layer 2: where did a wrong value come from?

Given what the caller said, what the recognizer heard (Layer 1) and what the tool
call contained, name the layer that failed:

  ok             tool value correct
  turn_taking    the platform cut the caller off mid-value (value arrived in pieces)
  recognition    the recognizer heard the wrong words
  spelling       the recognizer heard the right sound, but a name's spelling cannot be
                 known from audio (Aoife vs Eva); only a roster or a spelled answer fixes it
  understanding  the recognizer heard the right words, and the LLM turned them into the
                 wrong value (e.g. "two zero twenty eight" -> 22028), or left the field out
  unknown        no transcript to tell recognition from understanding
"""
from __future__ import annotations


def attribute(before_outcome: str, layer1: str | None, cut_in: bool = False) -> str:
    if before_outcome == "correct":
        return "ok"
    if cut_in and layer1 != "correct":
        return "turn_taking"
    if layer1 is None:
        return "unknown"
    if layer1 == "correct":
        return "understanding"
    if layer1 == "sound_alike":
        return "spelling"
    return "recognition"
