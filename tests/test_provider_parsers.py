"""Message parsing for each provider, using the shapes from their public docs."""
import numpy as np

from voxguard.backends.assemblyai_agent import AssemblyAIAgentBackend, parse_tool_call
from voxguard.backends.base import TurnInput
from voxguard.backends.deepgram_agent import DeepgramAgentBackend, parse_function_calls
from voxguard.backends.elevenlabs_agent import input_rate, parse_client_tool_call
from voxguard.backends.registry import AGENT_TYPES, build_agent
from voxguard.backends.streaming import pcm_chunks
from voxguard.backends.vapi import parse_call_record
from voxguard.bench.dataset import SYSTEM_PROMPT, TOOLS


def test_deepgram_function_call_request():
    msg = {"type": "FunctionCallRequest",
           "functions": [{"id": "f1", "name": "record_details", "arguments": '{"phone": "6789320288"}',
                          "client_side": True}]}
    calls = parse_function_calls(msg)
    assert calls[0].name == "record_details" and calls[0].args["phone"] == "6789320288"
    assert parse_function_calls({"type": "ConversationText"}) == []


def test_deepgram_settings_shape():
    turn = TurnInput(np.zeros(8000, np.float32), 8000, SYSTEM_PROMPT, TOOLS)
    s = DeepgramAgentBackend(api_key="x").settings(turn)
    assert s["type"] == "Settings"
    assert s["agent"]["think"]["prompt"] == SYSTEM_PROMPT
    assert s["agent"]["think"]["functions"][0]["name"] == "record_details"


def test_assemblyai_tool_call():
    call = parse_tool_call({"type": "tool.call", "call_id": "c1", "name": "record_details",
                            "arguments": {"name": "Siobhan Murphy"}})
    assert call.args["name"] == "Siobhan Murphy"
    turn = TurnInput(np.zeros(10, np.float32), 8000, SYSTEM_PROMPT, TOOLS)
    s = AssemblyAIAgentBackend(api_key="x").session(turn)
    assert s["type"] == "session.update" and s["session"]["tools"][0]["type"] == "function"


def test_elevenlabs_tool_call_and_rate():
    call = parse_client_tool_call({"type": "client_tool_call", "client_tool_call": {
        "tool_name": "record_details", "tool_call_id": "t1", "parameters": {"order_id": "429173"}}})
    assert call.args["order_id"] == "429173"
    meta = {"type": "conversation_initiation_metadata",
            "conversation_initiation_metadata_event": {"user_input_audio_format": "pcm_16000"}}
    assert input_rate(meta) == 16000


def test_vapi_call_record():
    rec = {"status": "ended", "artifact": {"messages": [
        {"role": "user", "message": "my number is"},
        {"role": "tool_calls", "secondsFromStart": 6.2,
         "toolCalls": [{"function": {"name": "record_details", "arguments": '{"phone": "6789320288"}'}}]},
    ]}}
    calls, times = parse_call_record(rec)
    assert calls[0].args["phone"] == "6789320288" and times == [6.2]


def test_call_collector_merges_partial_calls_and_flags_cut_in():
    from voxguard.backends.streaming import CallCollector
    from voxguard.core.types import ToolCall
    col = CallCollector()
    col.add(ToolCall("record_details", {"phone": "888272"}))        # during speech
    col.speech_ended()
    col.add(ToolCall("record_details", {"phone": "8882722028", "name": ""}))
    merged = col.merged()[0]
    st = col.stats()
    assert merged.args == {"phone": "8882722028"}
    assert st["cut_in"] and st["n_tool_calls"] == 2 and st["latency_ms"] >= 0


def test_pcm_chunks_cover_audio():
    x = np.zeros(8000, np.float32)
    chunks = pcm_chunks(x, 8000, 16000, chunk_ms=100)
    assert sum(len(c) for c in chunks) == 16000 * 2


def test_registry_builds_every_type():
    cfgs = {"elevenlabs_agent": {"agent_id": "a"}, "vapi": {"assistant_id": "b"},
            "audio_llm": {"base_url": "http://x", "model": "m"}, "cascade": {"llm_model": "m"}}
    for t in AGENT_TYPES:
        agent = build_agent({"type": t, **cfgs.get(t, {})})
        assert agent.name


def test_deepgram_flux_listen_settings():
    from voxguard.backends.deepgram_agent import DeepgramAgentBackend

    nova = DeepgramAgentBackend(listen_model="nova-3", api_key="x").listen_provider()
    assert nova == {"type": "deepgram", "model": "nova-3", "smart_format": False}
    flux = DeepgramAgentBackend(listen_model="flux-general-en", api_key="x",
                                listen_options={"eot_threshold": 0.9, "eot_timeout_ms": 5000}).listen_provider()
    assert flux["version"] == "v2" and flux["eot_threshold"] == 0.9 and "smart_format" not in flux


def test_deepgram_think_options():
    from voxguard.backends.deepgram_agent import DeepgramAgentBackend
    b = DeepgramAgentBackend(think_model="gpt-6-luna", api_key="x",
                             think_options={"reasoning_effort": "none", "temperature": "omit"})
    assert b.think_settings() == {"type": "open_ai", "model": "gpt-6-luna", "reasoning_effort": "none"}
