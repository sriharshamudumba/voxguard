"""voxguard benchmark UI.

    streamlit run ui/app.py

Run: pick data, a shadow recognizer, and any number of models; run before/after.
Results: compare saved runs.
Endpointing: tune the turn-taking math offline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from voxguard.backends.registry import AGENT_TYPES, SHADOW_TYPES, build_agent, build_shadow
from voxguard.bench.dataset import generate_cases, load_cases, save_cases
from voxguard.bench.metrics import overall, summarize
from voxguard.bench.runner import RunConfig, load_runs, run_benchmark, save_run
from voxguard.core.decision import CostModel
from voxguard.core.plugin import PluginChain
from voxguard.names import NameGuard
from voxguard.numbers import NumberGuard

st.set_page_config(page_title="voxguard", layout="wide")

RUNS_DIR = "runs"
PCT = ["before_error", "after_silent_error", "relative_reduction", "corrected", "caught", "false_confirm", "broken",
       "missing", "cut_in"]


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in PCT:
        if c in d:
            d[c] = (d[c] * 100).round(1)
    for c in ["agent_p50_ms", "agent_p95_ms", "guard_p95_ms"]:
        if c in d:
            d[c] = d[c].round(0)
    return d.rename(columns={c: c + " %" for c in PCT if c in d})


def strip_keys(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if "key" in k:
            continue
        out[k] = strip_keys(v) if isinstance(v, dict) else v
    return out


if "models" not in st.session_state:
    st.session_state.models = []

st.title("voxguard")
st.caption("Entity accuracy for voice agents, before and after the plugins. "
           "The main metric is silent error: a wrong value reaching your backend without the caller being asked.")

tab_run, tab_results, tab_vad = st.tabs(["Run", "Results", "Endpointing"])

# --------------------------------------------------------------------------- Run
with tab_run:
    st.subheader("1. Data")
    c1, c2, c3 = st.columns([2, 1, 1])
    cases_path = c1.text_input("Cases file", "data/cases.jsonl")
    n_per_kind = c2.number_input("Cases per kind", 5, 1000, 50, step=5)
    if c3.button("Generate cases"):
        save_cases(generate_cases(int(n_per_kind)), cases_path)
        st.success(f"Wrote {int(n_per_kind) * 4} cases")

    cases = load_cases(cases_path) if Path(cases_path).exists() else []
    with_audio = sum(1 for c in cases if c.audio_path and Path(c.audio_path).exists())
    st.write(f"{len(cases)} cases, {with_audio} with audio.")

    with st.expander("Synthesize caller audio (needed for real models)"):
        s1, s2, s3, s4 = st.columns(4)
        provider = s1.selectbox("TTS", ["openai", "piper", "espeak"])
        snr = s2.number_input("SNR dB", 0.0, 60.0, 20.0)
        loss = s3.number_input("Packet loss", 0.0, 0.2, 0.01, step=0.01)
        tts_key = s4.text_input("API key (openai only)", type="password")
        from voxguard.bench.synth import ACCENTS
        accents = st.multiselect("Accents (each case is recorded once per accent)", list(ACCENTS),
                                 default=list(ACCENTS))
        if st.button("Synthesize") and cases:
            from voxguard.bench.dataset import expand_accents
            from voxguard.bench.synth import synthesize_cases
            with st.spinner("Synthesizing"):
                cases = expand_accents(cases, accents or ["us"])
                synthesize_cases(cases, "data/audio", provider, snr, loss, api_key=tts_key or None)
                save_cases(cases, cases_path)
            st.success("Done. Audio is 8 kHz mu-law through a 300-3400 Hz band, like a phone line.")

    st.subheader("2. Shadow recognizer")
    st.caption("An independent recognizer the plugins compare against. It runs in parallel with the agent.")
    h1, h2, h3, h4 = st.columns(4)
    sh_type = h1.selectbox("Type", SHADOW_TYPES, index=0)
    sh_model = h2.text_input("Model", {"openai_stt": "whisper-1", "deepgram": "nova-3",
                                       "faster_whisper": "small.en"}.get(sh_type, ""))
    sh_base = h3.text_input("Base URL (openai_stt)", "https://api.openai.com/v1")
    sh_key = h4.text_input("API key", type="password", key="shkey")
    shadow_cfg = {"type": sh_type, "model": sh_model, "base_url": sh_base, "api_key": sh_key or None}

    st.subheader("3. Models")
    with st.form("add_model", clear_on_submit=False):
        m1, m2, m3 = st.columns(3)
        a_type = m1.selectbox("Type", AGENT_TYPES)
        a_name = m2.text_input("Display name (optional)")
        a_model = m3.text_input("Model, or agent/assistant ID for ElevenLabs and Vapi", "")
        m4, m5, m6 = st.columns(3)
        a_base = m4.text_input("Base URL (audio_llm, cascade LLM)", "https://api.openai.com/v1")
        a_key = m5.text_input("API key", type="password")
        a_profile = m6.selectbox("Mock profile", ["sts_typical", "cascade_typical"])
        m7, m8 = st.columns(2)
        a_stt_type = m7.selectbox("Cascade STT", ["openai_stt", "deepgram", "faster_whisper"])
        a_stt_model = m8.text_input("Cascade STT model", "whisper-1")
        if st.form_submit_button("Add model"):
            cfg: dict = {"type": a_type}
            if a_name:
                cfg["name"] = a_name
            if a_type == "mock":
                cfg["profile"] = a_profile
            elif a_type in ("openai_realtime", "gemini_live"):
                default = "gpt-realtime" if a_type == "openai_realtime" else "gemini-live-2.5-flash"
                cfg.update(model=a_model or default, api_key=a_key or None)
            elif a_type == "deepgram_agent":
                cfg.update(think_model=a_model or "gpt-4o-mini", api_key=a_key or None)
            elif a_type == "assemblyai_agent":
                cfg.update(api_key=a_key or None)
            elif a_type == "elevenlabs_agent":
                cfg.update(agent_id=a_model, api_key=a_key or None)
            elif a_type == "vapi":
                cfg.update(assistant_id=a_model, api_key=a_key or None)
            elif a_type == "audio_llm":
                cfg.update(base_url=a_base, model=a_model, api_key=a_key or None)
            elif a_type == "cascade":
                cfg.update(llm_base_url=a_base, llm_model=a_model or "gpt-4.1-mini", api_key=a_key or None,
                           stt={"type": a_stt_type, "model": a_stt_model, "api_key": a_key or None})
            st.session_state.models.append(cfg)

    if st.session_state.models:
        for i, cfg in enumerate(st.session_state.models):
            r1, r2 = st.columns([6, 1])
            r1.code(str(strip_keys(cfg)), language=None)
            if r2.button("Remove", key=f"rm{i}"):
                st.session_state.models.pop(i)
                st.rerun()
    else:
        st.write("No models added.")

    st.subheader("4. Plugins")
    p1, p2, p3, p4 = st.columns(4)
    use_numbers = p1.checkbox("Numbers", True)
    use_names = p2.checkbox("Names", True)
    phone_cost = p3.number_input("Cost of a wrong phone number vs one extra turn", 1.0, 200.0, 20.0)
    name_cost = p4.number_input("Cost of a wrong name vs one extra turn", 1.0, 200.0, 8.0)
    k1, k2 = st.columns(2)
    use_keyterms = k1.checkbox("Pass roster names to the shadow recognizer as keyterms", True)
    repeats = k2.number_input("Repeats (run the whole set N times)", 1, 10, 1)

    if st.button("Run benchmark", type="primary", disabled=not (cases and st.session_state.models)):
        plugins = []
        if use_numbers:
            ng = NumberGuard(fields={"phone": "phone", "order_id": "order_id"})
            ng.costs["phone"] = CostModel(phone_cost, 1.0)
            plugins.append(ng)
        if use_names:
            plugins.append(NameGuard(cost=CostModel(name_cost, 1.0)))
        chain = PluginChain(plugins)
        shadow = build_shadow({k: v for k, v in shadow_cfg.items() if v})
        for cfg in st.session_state.models:
            agent = build_agent(cfg)
            bar = st.progress(0.0, text=agent.name)
            try:
                rows, summ = asyncio.run(run_benchmark(
                    cases, agent, shadow, chain, RunConfig(use_keyterms=use_keyterms, repeats=int(repeats)),
                    progress=lambda d, n, bar=bar, nm=agent.name: bar.progress(d / n, text=f"{nm}: {d}/{n}")))
            except Exception as e:
                st.error(f"{agent.name}: {e}")
                continue
            path = save_run(rows, summ, RUNS_DIR, meta={"agent": strip_keys(cfg), "shadow": strip_keys(shadow_cfg),
                                                        "plugins": [p.name for p in plugins]})
            st.write(f"{agent.name} saved to {path}")
            st.dataframe(fmt(summ), hide_index=True, width="stretch")
        if any(c["type"] == "mock" for c in st.session_state.models):
            st.caption("Mock models are simulated error models for testing. Do not report them as model results.")

# --------------------------------------------------------------------------- Results
with tab_results:
    runs = load_runs(RUNS_DIR)
    if runs.empty:
        st.write("No runs yet.")
    else:
        names = sorted(runs["run"].unique(), reverse=True)
        chosen = st.multiselect("Runs", names, default=names[: min(4, len(names))])
        df = runs[runs["run"].isin(chosen)].copy()
        df["model"] = df["model"] + "  [" + df["run"].str[:15] + "]"
        if not df.empty:
            by = "kind"
            if "accent" in df and df["accent"].nunique() > 1:
                by = "accent" if st.radio("Break down by", ["Entity", "Accent"], horizontal=True) == "Accent" \
                    else "kind"
            summ = pd.concat([summarize(df, by), overall(df, by)], ignore_index=True)
            kinds = ["all"] + sorted(k for k in summ[by].unique() if k != "all")
            kind = st.radio("Group", kinds, horizontal=True)
            view = summ[summ[by] == kind]
            st.dataframe(fmt(view.drop(columns=[by])), hide_index=True, width="stretch")

            chart = view.set_index("model")[["before_error", "after_silent_error"]] * 100
            chart.columns = ["before %", "after %"]
            st.write("Silent error rate, before and after")
            st.bar_chart(chart, stack=False, horizontal=True)

            st.write("Cost of the fix")
            cost = view.set_index("model")[["false_confirm", "caught"]] * 100
            cost.columns = ["false confirm %", "caught %"]
            st.bar_chart(cost, stack=False, horizontal=True)

            st.write("Cases")
            f1, f2 = st.columns(2)
            outcome = f1.multiselect("After outcome", sorted(df["after"].unique()),
                                     default=[o for o in ["silent_error", "broken", "caught"] if o in set(df["after"])])
            kind_f = f2.multiselect("Kind", sorted(df["kind"].unique()))
            if "accent" in df and df["accent"].nunique() > 1:
                acc_f = st.multiselect("Accent", sorted(df["accent"].unique()))
                if acc_f:
                    df = df[df["accent"].isin(acc_f)]
            cdf = df[df["after"].isin(outcome)] if outcome else df
            if kind_f:
                cdf = cdf[cdf["kind"].isin(kind_f)]
            cols = ["model", "case_id", "accent", "expected", "before_value", "after_value", "action", "after",
                    "p_error", "reason", "shadow_text", "caller_text"]
            st.dataframe(cdf[cols], hide_index=True, width="stretch")
            st.download_button("Download rows (CSV)", df.to_csv(index=False), "voxguard_rows.csv")
            st.download_button("Download summary (CSV)", summ.to_csv(index=False), "voxguard_summary.csv")

# --------------------------------------------------------------------------- Endpointing
with tab_vad:
    from voxguard.bench.vad_sim import simulate, summarize_vad
    from voxguard.vad import EntityAwareEndpointer, FixedEndpointer

    st.write("Commit the turn when P(done) > C_cutin / (C_cutin + C_deadair), "
             "with P(done | t) = 1 - exp(-t / tau[state]). Threshold per state: t* = -tau ln(1 - p*).")
    ep = EntityAwareEndpointer()
    cols = st.columns(5)
    for i, s in enumerate(["complete", "incomplete", "filler", "mid_number", "spelling"]):
        with cols[i]:
            st.write(s)
            ep.tau_ms[s] = st.number_input("tau ms", 50.0, 3000.0, float(ep.tau_ms[s]), key=f"tau{s}")
            ep.c_cutin[s] = st.number_input("cut-in cost", 0.1, 50.0, float(ep.c_cutin[s]), key=f"cc{s}")
            st.write(f"threshold {ep.threshold_ms(s):.0f} ms")
    f1, f2 = st.columns(2)
    fixed_a = f1.number_input("Fixed baseline A (ms)", 100, 3000, 500)
    fixed_b = f2.number_input("Fixed baseline B (ms)", 100, 3000, 800)
    vcases = load_cases(cases_path) if Path(cases_path).exists() else generate_cases(50)
    res = simulate(vcases, {f"fixed_{fixed_a}": FixedEndpointer(fixed_a), f"fixed_{fixed_b}": FixedEndpointer(fixed_b),
                            "entity_aware": ep})
    sv = summarize_vad(res)
    sv["cut_in_rate"] = (sv["cut_in_rate"] * 100).round(1)
    sv = sv.rename(columns={"cut_in_rate": "cut_in %"}).round(0)
    st.dataframe(sv, hide_index=True, width="stretch")
    st.caption("Simulated word timelines: about 140 ms between words, about 700 ms between digit groups. "
               "Cut-in means the agent would have started talking while the caller was mid-turn.")
