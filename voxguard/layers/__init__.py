"""Three layers of a voice-agent turn, scored separately.

Layer 1  Recognition    did the system hear the right words?
Layer 2  Understanding  did those words become the right value in the tool call?
Layer 3  Recovery       when the value is wrong, can the agent fix it without the
                        caller repeating everything?
"""
from .attribution import attribute
from .recognition import heard_digits, recognize_entity
from .recovery import Repair, apply_reply, plan_repair

__all__ = ["attribute", "heard_digits", "recognize_entity", "Repair", "plan_repair", "apply_reply"]
