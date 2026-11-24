# Adding a backend

A backend is anything that takes one caller turn and returns the tool calls the agent made.

```python
from voxguard.backends.base import AgentBackend, AgentResult, TurnInput
from voxguard.core.types import ToolCall


class MyBackend(AgentBackend):
    kind = "sts"          # or "cascade", "audio_llm"

    def __init__(self, model: str, api_key: str | None = None, name: str | None = None):
        self.model = model
        self.api_key = api_key
        self.name = name or f"my_backend:{model}"

    async def run_turn(self, turn: TurnInput) -> AgentResult:
        # turn.audio: float32 mono in [-1, 1] at turn.sr (8000 for synthesized phone audio)
        # turn.system_prompt, turn.tools: give these to your model
        # Resample with voxguard.backends.audio.resample if your API needs 16 or 24 kHz.
        ...
        return AgentResult(tool_calls=[ToolCall("record_details", {"phone": "..."})], latency_ms=...)
```

Then register it in `voxguard/backends/registry.py` under a new `type` so configs and the UI can build it.

Rules that keep results comparable:

- Disable the provider's automatic turn detection and send the whole turn, so every model hears identical audio.
- Report latency from end of caller audio to the first model output (tool call or text).
- Return errors in `AgentResult.error` instead of raising, so one failure does not stop the run.
- Never read `turn.truth` or `turn.caller_text` outside mock backends.

Shadow recognizers implement `ShadowASR.transcribe(turn, keyterms) -> ShadowTranscript`. Return a confidence when the API provides one; the plugins use it.
