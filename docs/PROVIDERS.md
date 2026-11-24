# Provider setup

Every provider gets the same audio, the same system prompt, and the same tool:

```
record_details(name: string, phone: string, order_id: string)
```

Put keys in `.env` (never commit it) and `source .env` before running.

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export DEEPGRAM_API_KEY=...
export ASSEMBLYAI_API_KEY=...
export ELEVENLABS_API_KEY=...
export VAPI_API_KEY=...
```

| Provider | Type | What it tests | Setup |
|---|---|---|---|
| OpenAI | `openai_realtime` | gpt-realtime speech-to-speech | Key only |
| Google | `gemini_live` | Gemini Live speech-to-speech | Key only. Free tier: set `concurrency = 1` |
| Deepgram | `deepgram_agent` | Nova-3 STT + chosen LLM + Aura TTS, one API | Key only. Choose the LLM with `think_model` |
| AssemblyAI | `assemblyai_agent` | Universal STT + LLM + TTS, one API | Key only |
| ElevenLabs | `elevenlabs_agent` | ElevenLabs agent platform | Create an agent (below) |
| Vapi | `vapi` | Vapi platform with the STT and LLM you configure | Create an assistant (below) |

## ElevenLabs

1. In ElevenLabs, go to Agents and create a blank agent.
2. Set the system prompt to the text of `SYSTEM_PROMPT` in `voxguard/bench/dataset.py`. Set the first message to empty.
3. Under Tools, add a **Client** tool named `record_details` with three string parameters: `name`, `phone`, `order_id`. Description: "Record caller details exactly as spoken." Leave "Wait for response" off; voxguard ends the session as soon as the call arrives.
4. Copy the agent ID into the config: `agent_id = "..."`.
5. Optional: enable prompt overrides under the agent's Security settings and set `override_prompt = true`, so voxguard sends the prompt itself.

## Vapi

1. In Vapi, create an assistant. Choose the transcriber and model you want to test (for a fair comparison with Deepgram, use Deepgram Nova-3 and gpt-4o-mini).
2. Set the system prompt to `SYSTEM_PROMPT` and the first message to empty.
3. Add a **Function** tool named `record_details` with string parameters `name`, `phone`, `order_id`. No server URL is needed; voxguard reads the tool call from the call record after each call.
4. Copy the assistant ID into the config: `assistant_id = "..."`.

## How latency is measured

| Provider type | Latency |
|---|---|
| OpenAI, Gemini | We commit the turn manually, so latency is model time from end of audio to first output |
| Deepgram, AssemblyAI, ElevenLabs | Server-side turn detection; latency is end of caller speech to tool call, including the platform's endpointing wait |
| Vapi | Taken from the call record's `secondsFromStart`, minus speech duration. Approximate. |

These are different measurements, so compare latency within a group, not across groups. Entity accuracy is comparable across all of them.

## Notes on fairness

- The platforms (Deepgram, AssemblyAI, ElevenLabs, Vapi) run their own STT and an LLM. Results reflect the full configured stack, so report which LLM each used.
- Server-turn-detection platforms get audio at real-time pace, so a 200-case run takes about 200 x 10 s. Start with `--n 10`.
- A platform may speak or ask a question instead of calling the tool. That shows up as `missing`, and the agent text is saved in `rows.jsonl` for inspection.
