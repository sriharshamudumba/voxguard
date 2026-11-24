# voxguard

Training-free plugins that make voice agents reliable on **numbers, names, pronunciation, and turn-taking**, plus a benchmark that measures any model before and after.

Works with cascade pipelines (STT, LLM, TTS) and speech-to-speech models (OpenAI Realtime, Gemini Live, open speech LLMs on vLLM). No model weights are changed.

## Why

Voice agents rarely fail on conversation. They fail on the one token that matters: the caller's phone number, their name, the order ID. Word error rate hides this, because one wrong digit barely moves an average over all words. On a phone call it fails the whole task.

Known failure modes this targets:

| Failure | Example |
|---|---|
| Confusable digits on 8 kHz audio | "five" heard as "nine", "fifteen" as "fifty" |
| Speech-to-speech models inventing digits | caller says 9 digits, tool call gets 10 |
| Transcript right, tool argument wrong | STS model hears correctly, passes a different number to the function |
| Rare and non-English names | "Siobhan" written as "Sharon" |
| Cutting in mid-number | caller pauses between digit groups and the agent starts talking |
| Mispronounced numbers and names in TTS | "58213" read as a garbled word |

## How it works

Every plugin applies one model: **noisy channel + context prior + expected-cost decision**.

```
P(value | audio, context) ∝ P(audio | value) · P(value | context)

confirm  iff  P(error) · cost_error  >  cost_confirm
```

| Plugin | Method | Closed models | Open models |
|---|---|---|---|
| `NumberGuard` | Shadow recognizer as second evidence, spoken-number parsing, format validators, noisy-channel decoding with a digit confusion matrix, known-record matching. Never fills in digits the caller did not say. | Yes, via tool-call interception | Yes, plus `DigitCopyProcessor` at decode time |
| `NameGuard` | Bayesian resolution against a roster (bookings, CRM) with phonetic plus spelling similarity; asks the caller to spell when the posterior is flat | Yes | Yes, plus `TrieBiasProcessor` |
| `PronunciationGuard` | Spoken-form normalization (phones, money, times, emails, codes) and a respelling lexicon applied before TTS | Prompt hint only | Full, in cascades |
| `EntityAwareEndpointer` | Turn ending as optimal stopping: per-state hazard model, threshold `t* = -tau ln(1 - p*)`; waits longer mid-number, shorter after complete sentences | Yes, disable server VAD | Yes |

Full derivations: [docs/MATH.md](docs/MATH.md).

### Using it with a closed model

The plugin sits around the agent. It reads a copy of the caller audio through an independent recognizer, intercepts tool calls, and on doubt returns a tool error. Every agent API supports tool errors, so the model's own reasoning becomes the correction mechanism.

```
caller audio ──┬──────────────► voice agent (any model, untouched)
               │                      │ tool call: record(phone="678-932-0288")
               ▼                      ▼
         shadow ASR  ────────►  PluginChain.guard_tool_call()
                                      │
                    pass / corrected ─┴─ confirm → tool error: "caller said 678 932 028; confirm"
```

```python
from voxguard import NumberGuard, NameGuard, PluginChain, ShadowTranscript, ToolCall, TurnContext

chain = PluginChain([NumberGuard(fields={"phone": "phone"}), NameGuard()])

ctx = TurnContext(
    shadow=ShadowTranscript("my number is six seven eight nine three two zero two eight", 0.92),
    roster=["Siobhan Murphy|Shivawn Murphy", "Jon Smith"],
)
result = chain.guard_tool_call(ToolCall("record_details", {"phone": "678-932-0288"}), ctx)

if result.needs_confirmation:
    tool_output = result.tool_error_message()   # return this to the agent instead of executing
else:
    execute(result.tool_call)                    # possibly corrected arguments
```

## Benchmark

Each case is one caller turn with the exact tool arguments expected. Kinds: `phone`, `order_id`, `name` (in roster), `name_new` (first-time caller, prior cannot help). Audio is synthesized with several voices and accents, then passed through a simulated phone line: 300 to 3400 Hz band, 8 kHz, G.711 mu-law, noise, packet loss.

The agent is called once per case. "Before" is its raw tool call. "After" is the same tool call through the plugin chain. Same audio, same output, so the difference is the plugin alone.

| Metric | Meaning |
|---|---|
| `before_error` | wrong or missing value in the raw tool call |
| `after_silent_error` | wrong value reaching the backend without the caller being asked. **The headline number.** |
| `caught` | wrong value, plugin asked the caller |
| `corrected` | wrong value, plugin fixed it silently |
| `false_confirm` | right value, plugin asked anyway. The cost of the fix. |
| `broken` | right value, plugin changed it to a wrong one |
| `agent_p50_ms`, `guard_p95_ms` | agent latency, and time added by the plugins |

### Accents and repeats

`voxguard synth --accents all` records every case once per accent: us, british, irish, scottish, indian, nigerian, mexican, chinese, arabic, fast_casual. Pick a subset with `--accents us,indian,scottish`. Set `repeats` in the config to run the whole set several times, since model outputs vary between runs. The report adds before/after heatmaps by accent and 95% bootstrap confidence intervals that resample whole test cases, so accent copies and repeats of one case are not counted as independent evidence.

A plugin can always "win" by confirming everything, so `after_silent_error` and `false_confirm` must be read together.

### Quick start

```bash
pip install -e ".[ui,dev]"

# offline demo, no keys (simulated backends)
voxguard gen --n 100 --out data/mock_cases.jsonl
voxguard bench --config configs/mock.toml
voxguard vad --n 100

# real models (keys in .env, see docs/PROVIDERS.md)
source .env
voxguard gen --n 10 --out data/cases.jsonl
voxguard synth --cases data/cases.jsonl --provider openai --accents all   # every case in 10 accents
voxguard bench --config configs/all_providers.toml
voxguard report                                                          # charts + REPORT.md

# UI
streamlit run ui/app.py
```

### Supported backends

| Type | Covers | Status |
|---|---|---|
| `openai_realtime` | gpt-realtime, gpt-realtime-2 (GA and beta event schemas) | Written against the public docs, not yet verified on a live key |
| `gemini_live` | Gemini Live models | Written against the public docs, not yet verified on a live key |
| `deepgram_agent` | Deepgram Voice Agent API (Nova-3 + chosen LLM + Aura) | Written against the public docs, not yet verified on a live key |
| `assemblyai_agent` | AssemblyAI Voice Agent API | Written against the public docs, not yet verified on a live key |
| `elevenlabs_agent` | ElevenLabs Agents (needs an agent with a client tool) | Written against the public docs, not yet verified on a live key |
| `vapi` | Vapi over the WebSocket transport (needs an assistant) | Written against the public docs, not yet verified on a live key |
| `audio_llm` | Any OpenAI-compatible chat endpoint that accepts `input_audio`: gpt-4o-audio, Ultravox / Qwen-Omni on vLLM | Written against the public docs, not yet verified |
| `cascade` | Any STT below + any OpenAI-compatible LLM (OpenAI, Groq, Together, vLLM, Ollama) | Written against the public docs, not yet verified |
| shadow `openai_stt` | OpenAI, Groq, or vLLM `/audio/transcriptions` | Not yet verified |
| shadow `deepgram` | Nova-3, with keyterm prompting | Not yet verified |
| shadow `faster_whisper` | Local, free | Not yet verified |
| `mock` | Simulated error models for tests and CI | Tested |

Provider setup, including ElevenLabs and Vapi: [docs/PROVIDERS.md](docs/PROVIDERS.md). All six providers in one config: `configs/all_providers.toml`.

Charts and a markdown summary from saved runs: `voxguard report` writes `report/REPORT.md` plus `silent_error.png`, `silent_error_by_entity.png`, `false_confirm.png`, and `latency.png`.

Adding a backend means implementing one method, `run_turn(TurnInput) -> AgentResult`. See [docs/ADDING_A_BACKEND.md](docs/ADDING_A_BACKEND.md).

### Simulated results (not real models)

From `configs/mock.toml`. The mock agent injects digit substitutions, drops, and STS-style padding; the mock shadow makes partly correlated errors (40% of errors are shared, as with real recognizers on the same bad audio). These numbers check that the pipeline and math behave as intended. **They are not claims about any real model.** Replace them with real runs.

| Simulated agent | before error | after silent error | caught | corrected | false confirm |
|---|---|---|---|---|---|
| sts_typical | 14.0% | 1.2% | 4.8% | 8.0% | 3.0% |
| cascade_typical | 11.8% | 1.5% | 3.8% | 6.5% | 3.2% |

Endpointing, simulated timelines (about 700 ms pauses between digit groups):

| Endpointer | cut-in rate (all) | cut-in rate (phone) | silence after a complete sentence |
|---|---|---|---|
| fixed 500 ms | 69.2% | 99.0% | 500 ms |
| fixed 800 ms | 25.2% | 54.0% | 800 ms |
| entity-aware | 13.0% | 12.0% | 257 ms |

Known tradeoff: when a number of unknown length ends (order IDs), the entity-aware endpointer cannot tell it is over and waits about 1.2 s. Passing the expected length from your records fixes this.

### Relation to vendor benchmarks

STT vendors publish entity error rates for transcription. AssemblyAI, for example, reports that Universal-3.6 Pro Realtime misses 10.9% of names, 10.0% of codes and IDs, and 2.4% of phone numbers on its voice-agent benchmark. Those measure the recognizer. voxguard measures the **tool-call argument** end to end, on any agent, and the plugins sit on top of any STT, including those. Better STT and these plugins stack.

## Limits

- The plugins detect and route errors; they cannot recover information the phone line removed. When two names sound the same on 8 kHz audio and neither is in your records, the right action is to ask, and the plugin does.
- Correlated errors: if the agent and the shadow recognizer mishear the same way, nothing catches it. Use a shadow from a different model family than the agent.
- Pure speech-to-speech output audio cannot be re-pronounced from outside the model.
- Synthetic accents approximate real ones. Add consented real recordings before making claims about specific accents.
- Prior error rates and costs in the plugins are defaults. Measure them on your traffic with this benchmark and set them.

## Repository layout

```
voxguard/
  core/           types, plugin chain, decision theory and calibration
  numbers/        spoken-number parsing, validators, noisy-channel decoder, NumberGuard
  names/          phonetic keys, Bayesian resolver, NameGuard
  pronunciation/  spoken-form normalizer, PronunciationGuard
  vad/            entity-aware endpointing
  logits/         decode-time processors for open models (torch optional)
  backends/       agent and shadow-recognizer adapters, audio and phone-line simulation
  bench/          dataset, synthesis, runner, metrics, endpointing simulation
ui/app.py         Streamlit UI
configs/          example run configs
docs/             math and extension guides
tests/
```

## License

Apache-2.0
