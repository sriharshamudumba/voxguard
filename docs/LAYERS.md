# Three layers of a voice-agent turn

voxguard scores every turn on where it broke, and fixes each layer differently.

| Layer | Question | Measured by | voxguard's fix |
|---|---|---|---|
| 1. Recognition | Did the system hear the right words? | The platform's own caller transcript vs what was said | A second recognizer as independent evidence |
| 2. Understanding | Did those words become the right value? | The tool call vs the transcript | Re-read numbers from the transcript with a grammar; flag a number the tool call left out; flag a name with a part missing |
| 3. Recovery | When the value is wrong, can it be fixed without the caller repeating everything? | Replaying each confirmation as a follow-up turn | Ask only for the part in doubt |

## Layer 1: recognition

Cascade platforms (Deepgram, AssemblyAI, ElevenLabs) send the transcript their LLM receives, so
Layer 1 is measured exactly. Speech-to-speech models (OpenAI Realtime, Gemini Live) hear audio;
their transcript comes from a separate recognizer, so their Layer 1 is an estimate and rows carry
`heard_source = side_channel`.

Outcomes per field: `correct`, `wrong`, and for names `sound_alike` (the right sound, another
spelling, which no recognizer can resolve from audio) and `partial` (part of the name missing).

## Layer 2: where a wrong value came from

```
ok             tool value correct
turn_taking    the platform ended the turn mid-value (the value arrived in pieces)
recognition    the recognizer heard the wrong words
spelling       right sound, unknowable spelling (Aoife heard as Eva)
understanding  the words were heard right and the LLM wrote the wrong value
unknown        no transcript, so hearing and understanding cannot be told apart
```

Example from a real run (Deepgram Flux + gpt-4o-mini): Flux heard "eight eight eight two seven two
two zero twenty eight" correctly; the LLM wrote `88827222028`. That is an understanding error.
Turning spoken numbers into digits is a grammar problem, so voxguard redoes it with code:

- transcript the LLM received (pipeline), no other evidence disagrees: correct to the parsed value
- side-channel transcript: correct only when the second recognizer agrees

## Layer 3: recovery

For each confirmation, voxguard plans a question from the tool value and the evidence, never the
true value:

1. Split the value the way people say it (3-3-4 for a phone number).
2. Find the groups where the tool value and the evidence disagree (aligned position by position,
   or by common prefix and suffix when lengths differ).
3. Read the value back and ask only for those groups: "I have 719 209 3799. Could you say the last
   4 digits again?" For a name, ask to spell only the part that disagrees or is missing.

`voxguard recover` replays this on saved runs. The simulated caller answers with the digits or
letters asked for, or "yes, that's right" when the read-back part was already correct. The answer is
spoken with the same accent voice, passed through the phone-line simulation, and recognized by the
configured recognizer, so recovery can fail the way the first attempt did. The same rows are also
replayed with the usual fallback, "could you repeat the whole number", for comparison.

Reported per model: values saved correctly per call (without voxguard; with voxguard and full
repeat; with voxguard and targeted repair), the share of the value the caller had to say again, and
the caller's extra seconds per call.

## Limits

- The simulated caller always answers cooperatively and exactly what was asked; real callers
  repeat more than asked, or correct a different part.
- One follow-up round is scored. A failed repair is counted as failed, not retried.
- Targeted repair depends on the evidence: if the tool value and the evidence agree on a wrong
  digit, voxguard cannot localize it and asks for the whole value.
- Caller voices are synthetic (OpenAI TTS with accent instructions).
