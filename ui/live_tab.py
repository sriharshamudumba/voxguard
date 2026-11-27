"""Live tab: play a sample call or record your own, run several models, and see
what the backend would receive with and without voxguard.

Keys, in order of precedence:
  1. keys the visitor pastes (per request only)
  2. the server's keys, unlocked by the demo passcode (Streamlit secrets)
  3. environment variables (local use with .env)
Keys are passed to providers per request. They are never written to os.environ,
shown in the page, or saved.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from voxguard.backends.audio import read_wav, wav_bytes
from voxguard.live import FIELDS, asked, compare, decode_wav

SAMPLES_DIR = Path("demo/samples")
FIELD_LABEL = {"name": "Name", "phone": "Phone", "order_id": "Order ID"}

PRESETS = {
    "OpenAI gpt-realtime": ({"type": "openai_realtime", "model": "gpt-realtime"}, "OPENAI_API_KEY"),
    "Google Gemini Live 3.8": ({"type": "gemini_live", "model": "gemini-3.8-live",
                                "response_modality": "AUDIO"}, "GEMINI_API_KEY"),
    "Deepgram Voice Agent": ({"type": "deepgram_agent", "think_model": "gpt-4o-mini"}, "DEEPGRAM_API_KEY"),
    "AssemblyAI Voice Agent": ({"type": "assemblyai_agent"}, "ASSEMBLYAI_API_KEY"),
    "ElevenLabs Agents": ({"type": "elevenlabs_agent"}, "ELEVENLABS_API_KEY"),
    "Vapi": ({"type": "vapi"}, "VAPI_API_KEY"),
    "Simulated (no key)": ({"type": "mock", "profile": "sts_typical"}, None),
}
KEY_NAMES = ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY",
             "ELEVENLABS_API_KEY", "VAPI_API_KEY"]


def _secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:  # no secrets file locally
        return {}


@st.cache_resource
def _daily_counter() -> dict:
    return {}


def _under_daily_limit(limit: int) -> bool:
    c = _daily_counter()
    today = dt.date.today().isoformat()
    return c.get(today, 0) < limit


def _count_run() -> None:
    c = _daily_counter()
    today = dt.date.today().isoformat()
    c[today] = c.get(today, 0) + 1
    st.session_state["live_runs"] = st.session_state.get("live_runs", 0) + 1


def load_samples() -> list[dict]:
    path = SAMPLES_DIR / "manifest.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _status(field: str, value, truth: str, action: str | None) -> tuple[str, str]:
    """(kind, text) where kind is ok | bad | ask | none."""
    from voxguard.bench.metrics import norm
    kind = {"phone": "phone", "order_id": "order_id"}.get(field, "name")
    if action == "confirm":
        return "ask", "Asks the caller to confirm"
    if not value:
        return "none", "No value recorded"
    if truth:
        return ("ok", "Correct") if norm(kind, value) == norm(kind, truth) else ("bad", "Wrong")
    return "none", ""


def _show(kind: str, value, text: str) -> None:
    line = f"**{value}**" if value else ("" if value is None and text else "(nothing)")
    if text:
        line = f"{line}  \n{text}" if line else text
    {"ok": st.success, "bad": st.error, "ask": st.warning}.get(kind, st.info)(line)


def render() -> None:
    secrets = _secrets()
    demo_pw = secrets.get("DEMO_PASSWORD")
    session_limit = int(secrets.get("DEMO_SESSION_LIMIT", 15))
    daily_limit = int(secrets.get("DEMO_DAILY_LIMIT", 150))

    st.write("Play a sample call or record your own, choose models, and see what your backend would "
             "receive without and with voxguard.")

    hosted = bool(demo_pw)
    unlocked = False
    if demo_pw:
        code = st.text_input("Demo passcode (unlocks the hosted keys)", type="password", key="demo_code")
        unlocked = bool(code) and hmac.compare_digest(code, str(demo_pw))
        if code and not unlocked:
            st.error("Wrong passcode.")

    with st.expander("Or use your own API keys (sent only with your request, never stored)"):
        cols = st.columns(3)
        user_keys = {k: cols[i % 3].text_input(k.replace("_API_KEY", "").title(), type="password",
                                                key=f"uk_{k}") for i, k in enumerate(KEY_NAMES)}

    def key_for(env: str | None) -> str | None:
        if env is None:
            return None
        if user_keys.get(env):
            return user_keys[env]
        if hosted:  # Streamlit Cloud also exposes secrets as environment variables: passcode only
            return secrets.get(env) if unlocked else None
        return os.environ.get(env) or None

    available = [k for k, (_, env) in PRESETS.items() if env is None or key_for(env)]
    chosen = st.multiselect("Models", available,
                            default=[m for m in available if m != "Simulated (no key)"][:3] or available[:1])
    ids = st.columns(2)
    el_id = ids[0].text_input("ElevenLabs agent ID", value=secrets.get("ELEVENLABS_AGENT_ID", ""),
                              key="live_el") if "ElevenLabs Agents" in chosen else ""
    vapi_id = ids[1].text_input("Vapi assistant ID", value=secrets.get("VAPI_ASSISTANT_ID", ""),
                                key="live_vapi") if "Vapi" in chosen else ""

    # ---- input: sample call or microphone
    samples = load_samples()
    modes = (["Sample call"] if samples else []) + ["Record my voice"]
    mode = st.radio("Input", modes, horizontal=True)
    audio = sr = None
    truth = {f: "" for f in FIELDS}
    roster_default = "Siobhan Murphy\nAoife Byrne\nYusuf Demir\nEwa Kowalczyk\nJon Smith"
    orders_default = ""
    phone_line = True

    if mode == "Sample call":
        labels = [s["label"] for s in samples]
        pick = samples[labels.index(st.selectbox("Sample", labels))]
        audio, sr = read_wav(str(SAMPLES_DIR / pick["file"]))
        st.audio(wav_bytes(audio, sr), format="audio/wav")
        st.caption(f"Caller says: \"{pick['text']}\"  (synthetic voice, {pick.get('accent', '')} accent, "
                   "already passed through a phone line)")
        truth.update(pick.get("truth", {}))
        roster_default = "\n".join(pick.get("roster", [])) or roster_default
        orders_default = "\n".join(pick.get("order_ids", []))
        phone_line = False
    else:
        st.caption("Try: \"My number is six one seven, five five five, oh one double eight.\"  "
                   "Pause between the groups the way people do on the phone.")
        rec = st.audio_input("Record one caller turn")
        if rec is not None:
            audio, sr = decode_wav(rec.getvalue())
        phone_line = st.checkbox("Pass my audio through a simulated phone line (8 kHz, 300 to 3400 Hz)", True)
        with st.expander("What I actually said (to score each model)"):
            t = st.columns(3)
            truth = {"name": t[0].text_input("Name"), "phone": t[1].text_input("Phone"),
                     "order_id": t[2].text_input("Order ID")}

    with st.expander("Context the plugins can use"):
        c1, c2 = st.columns(2)
        roster_txt = c1.text_area("Known names (bookings, CRM)", roster_default, height=120,
                                  key=f"roster_{mode}_{truth.get('name', '')}")
        orders_txt = c2.text_area("Known order IDs", orders_default, height=120,
                                  key=f"orders_{mode}_{truth.get('order_id', '')}")

    shadow_key = key_for("DEEPGRAM_API_KEY")
    if shadow_key:
        shadow_cfg = {"type": "deepgram", "model": "nova-3", "api_key": shadow_key}
    elif key_for("OPENAI_API_KEY"):
        shadow_cfg = {"type": "openai_stt", "model": "whisper-1", "api_key": key_for("OPENAI_API_KEY")}
    else:
        shadow_cfg = {"type": "mock"}

    used = st.session_state.get("live_runs", 0)
    over = demo_pw and unlocked and (used >= session_limit or not _under_daily_limit(daily_limit))
    if over:
        st.warning("Demo limit reached for this session. Use your own keys above to keep testing.")
    ready = audio is not None and chosen and not over
    if st.button("Compare", type="primary", disabled=not ready):
        cfgs = []
        for label in chosen:
            base, env = PRESETS[label]
            cfg = dict(base, name=label)
            if env:
                cfg["api_key"] = key_for(env)
            if cfg["type"] == "elevenlabs_agent":
                if not el_id:
                    st.warning("ElevenLabs needs an agent ID; skipped.")
                    continue
                cfg["agent_id"] = el_id
            if cfg["type"] == "vapi":
                if not vapi_id:
                    st.warning("Vapi needs an assistant ID; skipped.")
                    continue
                cfg["assistant_id"] = vapi_id
            cfgs.append(cfg)
        if unlocked:
            _count_run()
        with st.spinner(f"Running {len(cfgs)} models on {len(audio) / sr:.1f} s of audio"):
            results, shadow_tr, (heard, heard_sr) = asyncio.run(compare(
                audio, sr, cfgs, shadow_cfg, roster_txt.splitlines(), orders_txt.splitlines(), phone_line, truth))
        fields_used = [f for f in FIELDS if truth.get(f)] or \
            [f for f in FIELDS if any(r.before.get(f) or r.after.get(f) for r in results)] or ["phone"]
        # results stay in this browser session so the caller can answer voxguard's questions;
        # keys are not kept here: answers use the recognizer built from the current keys
        st.session_state["live"] = {"results": results, "shadow_text": shadow_tr.text, "audio": (heard, heard_sr),
                                    "truth": truth, "fields": fields_used, "phone_line": phone_line}
    state = st.session_state.get("live")
    if state:
        _render_results(state, shadow_cfg)


LAYER1_TEXT = {"correct": ("ok", "Heard it right"), "wrong": ("bad", "Heard it wrong"),
               "sound_alike": ("ask", "Same sound, different spelling"),
               "partial": ("bad", "Heard only part of it")}
LAYER2_TEXT = {
    "turn_taking": "Without voxguard this failed at turn-taking: the agent ended the turn mid-value.",
    "recognition": "Without voxguard this failed at recognition: the recognizer heard the wrong words.",
    "spelling": "Without voxguard this failed at name spelling: the right sound, but spelling cannot be known "
                "from audio.",
    "understanding": "Without voxguard this failed at understanding: the words were heard right, and the LLM "
                     "wrote the wrong value.",
    "unknown": "This provider sends no transcript, so hearing and understanding cannot be told apart.",
}


def _render_results(state: dict, shadow_cfg: dict) -> None:
    from voxguard.bench.metrics import norm
    from voxguard.live import answer_repair

    results, truth, fields_used = state["results"], state["truth"], state["fields"]
    for r in results:
        st.divider()
        st.subheader(r.model)
        if r.error:
            st.error(f"Provider error: {r.error[:300]}")
            continue
        if r.heard:
            src = "what its LLM received" if r.heard_source == "pipeline" else \
                "a side transcript; this model hears audio directly"
            st.caption(f"Transcript ({src}): \"{r.heard[:300]}\"")
        if asked(r):
            st.info(f"The model asked the caller instead of recording a value: \"{r.agent_text[:200]}\"")
        heads = st.columns(3)
        heads[0].caption("1. Heard")
        heads[1].caption("2. Understood: without voxguard, then with it")
        heads[2].caption("3. Recovery")
        for f in fields_used:
            act, why = r.actions.get(f, (None, ""))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(FIELD_LABEL[f])
                l1 = r.layer1.get(f)
                if l1:
                    _show(LAYER1_TEXT[l1][0], None, LAYER1_TEXT[l1][1])
                elif r.heard:
                    st.caption("Add what you said (above) to score this.")
                else:
                    st.caption("This provider sends no transcript.")
            with c2:
                st.write(FIELD_LABEL[f])
                kind, text = _status(f, r.before.get(f), truth.get(f, ""), None)
                _show(kind, r.before.get(f), text)
                kind, text = _status(f, r.after.get(f), truth.get(f, ""), act)
                _show(kind, r.after.get(f) if act != "confirm" else r.before.get(f), text)
                if why:
                    st.caption(why.replace("shadow", "second recognizer"))
            with c3:
                st.write(FIELD_LABEL[f])
                rep = r.repairs.get(f)
                if act == "correct":
                    st.caption("Fixed without asking the caller.")
                elif rep is None:
                    st.caption("Nothing to recover." if act else "")
                else:
                    whole = f != "name" and any(a.start == 0 and a.end == rep.total_units for a in rep.asks)
                    asked_units = "the whole number" if whole else "only the part in doubt"
                    st.warning(" ".join(a.question for a in rep.asks))
                    if shadow_cfg.get("type") == "mock":
                        st.caption(f"voxguard asks for {asked_units}. Answering out loud needs a real "
                                   "recognizer: enter the passcode or a Deepgram or OpenAI key above.")
                    else:
                        st.caption(f"voxguard asks for {asked_units}. Answer as the caller:")
                        ans = st.audio_input("Answer", key=f"ans_{r.model}_{f}", label_visibility="collapsed")
                        if ans is not None:
                            x, asr = decode_wav(ans.getvalue())
                            with st.spinner("Recognizing your answer"):
                                heard_ans, final = asyncio.run(answer_repair(
                                    rep, r.before.get(f), x, asr, shadow_cfg, state.get("phone_line", True)))
                            exp = truth.get(f, "")
                            k = {"phone": "phone", "order_id": "order_id"}.get(f, "name")
                            ok = ("ok" if norm(k, final) == norm(k, exp) else "bad") if exp else "none"
                            _show(ok, final, f"heard: \"{heard_ans}\"")
            l2 = r.layer2.get(f)
            if l2 and l2 != "ok":
                st.caption(LAYER2_TEXT[l2])
        notes = []
        if r.latency_ms == r.latency_ms:
            notes.append(f"latency {r.latency_ms:.0f} ms")
        if r.cut_in:
            notes.append("recorded a value before the caller finished speaking")
        if notes:
            st.caption("; ".join(notes))

    st.divider()
    with st.expander("Details: what the models heard"):
        heard, heard_sr = state["audio"]
        st.audio(wav_bytes(heard, heard_sr), format="audio/wav")
        st.write(f"Second recognizer: {state['shadow_text'] or '(empty)'}")
        st.dataframe(pd.DataFrame([{
            "Model": r.model, "Transcript": r.heard or "", "Before": json.dumps(r.before),
            "After": json.dumps(r.after), "Agent said": r.agent_text[:200], "Error": r.error or ""}
            for r in results]), hide_index=True, width="stretch")
        st.caption("Latency is measured differently for providers that detect the end of the turn "
                   "themselves; compare it within each group. Sample voices are synthetic.")
