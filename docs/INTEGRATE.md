# Adding voxguard to a live voice agent

voxguard sits at one point in any stack: **after the LLM decides to call a tool, before the tool
runs**. It needs three things your stack already has:

| Input | Where it comes from | Required |
|---|---|---|
| Tool name and arguments | Your tool or function-call handler | Yes |
| The caller's words this turn | Your speech-to-text or platform transcript events | Recommended: enables the number re-check and the partial-name check |
| The caller's audio this turn | Your audio stream | Optional: only for a second recognizer |

It returns a `Decision`:

| `action` | Meaning | What you do |
|---|---|---|
| `pass` | Nothing wrong found | Run the tool with `d.args` |
| `correct` | voxguard fixed a value (digits re-read from the transcript, a name matched to your list) | Run the tool with `d.args` |
| `confirm` | The value is in doubt | Don't run the tool. Return `d.say` to the LLM as the tool result. It tells the LLM to ask one targeted question (`d.question`), then call the tool again. |

voxguard asks at most once per field per call, then accepts the caller's answer, so it can't loop.

## The core pattern (any framework)

```python
from voxguard.sdk import Guard

guard = Guard(
    fields={"phone": "phone", "name": "name", "order_id": "order_id"},  # tool argument -> kind
    roster=todays_reservation_names,          # optional: fixes misspelled names
    known={"order_id": open_order_ids},       # optional: fixes misheard IDs
    transcript_source="pipeline",             # "side_channel" for OpenAI Realtime / Gemini Live
    log_path="voxguard_calls.jsonl",          # for `voxguard calls`
)

call = guard.new_call(call_id)                # once per phone call

def on_caller_transcript(text):               # each final transcript segment
    call.add_transcript(text)

async def on_tool_call(name, args):
    d = await call.acheck(name, args)         # or call.check(...) outside an event loop
    if d.action == "confirm":
        return d.say                          # tool result: makes the LLM ask d.question
    return run_tool(name, d.args)

def on_agent_reply_done():
    call.end_turn()
```

## Pipecat

Register your tool handler through voxguard, and tap user transcriptions with a small processor.

```python
from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

class VoxguardTap(FrameProcessor):            # place after the STT service in the pipeline
    def __init__(self, call):
        super().__init__(); self.call = call
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self.call.add_transcript(frame.text)
        await self.push_frame(frame, direction)

async def record_details(params):             # registered with llm.register_function("record_details", ...)
    d = await call.acheck("record_details", dict(params.arguments))
    if d.action == "confirm":
        await params.result_callback({"error": d.say})
        return
    await params.result_callback(save_reservation(**d.args))
```

## LiveKit Agents

```python
@function_tool
async def record_details(self, context: RunContext, name: str, phone: str):
    d = await self.call.acheck("record_details", {"name": name, "phone": phone})
    if d.action == "confirm":
        return d.say
    return save_reservation(**d.args)

@session.on("user_input_transcribed")
def _on_transcript(ev):
    if ev.is_final:
        call.add_transcript(ev.transcript)
```

## Your own code on OpenAI Realtime or Gemini Live

Turn on input transcription in the session, then:

- on `conversation.item.input_audio_transcription.completed` (OpenAI) or
  `serverContent.inputTranscription` (Gemini): `call.add_transcript(text)`
- on a function call: `d = await call.acheck(name, args)`, and send `d.say` (if `confirm`) or your
  tool's result back as the function output
- use `transcript_source="side_channel"`: these models hear audio, so voxguard only corrects a number
  when a second source agrees, and otherwise asks

## Hosted platforms (Vapi, Retell, anything with a tool webhook)

Run voxguard as a service, and have your tool endpoint call it before doing the work:

```bash
VOXGUARD_TOKEN=pick-a-secret voxguard serve --port 8765 --roster todays_names.txt
```

```python
# inside your existing tool endpoint
d = requests.post("http://localhost:8765/check", headers={"Authorization": "Bearer pick-a-secret"},
                  json={"call_id": call_id, "tool": tool_name, "args": args,
                        "transcript": latest_caller_text}).json()
if d["action"] == "confirm":
    return d["say"]                            # as the tool's result
return do_the_work(d["args"])
```

Call `POST /end_turn` after each agent reply and `POST /end_call` when the call ends. Platforms
include the transcript so far in their webhook payloads; pass the caller's latest lines as
`transcript`.

## Measuring it on your own calls

```bash
voxguard calls --log voxguard_calls.jsonl
```

| Column | Meaning |
|---|---|
| fixed | voxguard corrected a value without asking. Spot-check a few by listening. |
| asked: caller changed it | voxguard asked, and the value changed after the caller answered: **an error caught** |
| asked: same value | The caller confirmed what the model had: **an unnecessary question** |

The log masks digits and names by default (`7192098799` becomes `********99`). Use
`--no-redact` only for your own test calls.

## Settings to decide

| Setting | Default | Choose |
|---|---|---|
| `ask_unlisted_names` | `True` | With a roster, ask callers not on it to spell their name. Set `False` to accept names as heard unless the audio contradicts them. |
| `second_recognizer` | Off | `{"type": "deepgram", "model": "nova-3"}` adds an independent check, at about 0.3 to 0.6 s per tool call and the cost of a second transcription |
| `fields` | phone, order_id, name | Map your tool's argument names to kinds |

These snippets follow each framework's public API at the time of writing. Framework APIs change, so
check the current docs for exact class and event names.
