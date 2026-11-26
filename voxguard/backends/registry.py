"""Build backends from plain dicts (TOML config or the UI form).

Agent config examples:
  {type = "openai_realtime", model = "gpt-realtime"}
  {type = "gemini_live", model = "gemini-3.8-live"}
  {type = "audio_llm", base_url = "http://localhost:8000/v1",
   model = "fixie-ai/ultravox-v0_5-llama-3_1-8b", api_key = "x"}
  {type = "cascade", stt = {type = "openai_stt", model = "whisper-1"},
   llm_base_url = "https://api.openai.com/v1", llm_model = "gpt-4.1-mini"}
  {type = "deepgram_agent", think_model = "gpt-4o-mini"}
  {type = "assemblyai_agent"}
  {type = "elevenlabs_agent", agent_id = "..."}
  {type = "vapi", assistant_id = "..."}
  {type = "mock", profile = "sts_typical"}
"""
from __future__ import annotations

from typing import Any

from .base import AgentBackend, ShadowASR


def build_shadow(cfg: dict[str, Any]) -> ShadowASR:
    t = cfg.get("type", "mock")
    if t == "mock":
        from .mock import MockShadow
        return MockShadow(profile=cfg.get("profile", "shadow_typical"))
    if t == "openai_stt":
        from .openai_compat import OpenAICompatSTT
        return OpenAICompatSTT(base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                               model=cfg.get("model", "whisper-1"), api_key=cfg.get("api_key"),
                               api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"))
    if t == "deepgram":
        from .deepgram import DeepgramSTT
        return DeepgramSTT(model=cfg.get("model", "nova-3"), api_key=cfg.get("api_key"))
    if t == "faster_whisper":
        from .local_whisper import FasterWhisperSTT
        return FasterWhisperSTT(model=cfg.get("model", "small.en"), device=cfg.get("device", "auto"))
    raise ValueError(f"unknown shadow type {t!r}")


def build_agent(cfg: dict[str, Any]) -> AgentBackend:
    t = cfg.get("type", "mock")
    name = cfg.get("name")
    if t == "mock":
        from .mock import MockAgent
        prof = cfg.get("profile", "sts_typical")
        return MockAgent(profile=prof, name=name or f"mock:{prof}")
    if t == "openai_realtime":
        from .openai_realtime import OpenAIRealtimeBackend
        return OpenAIRealtimeBackend(model=cfg.get("model", "gpt-realtime"), api_key=cfg.get("api_key"),
                                     schema=cfg.get("schema", "ga"), name=name)
    if t == "gemini_live":
        from .gemini_live import GeminiLiveBackend
        return GeminiLiveBackend(model=cfg.get("model", "gemini-3.8-live"), api_key=cfg.get("api_key"),
                                 response_modality=cfg.get("response_modality", "AUDIO"), name=name)
    if t == "audio_llm":
        from .openai_compat import AudioLLMBackend
        return AudioLLMBackend(base_url=cfg["base_url"], model=cfg["model"], api_key=cfg.get("api_key"),
                               api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"), name=name)
    if t == "cascade":
        from .openai_compat import CascadeBackend
        return CascadeBackend(stt=build_shadow(cfg.get("stt", {"type": "openai_stt"})),
                              llm_base_url=cfg.get("llm_base_url", "https://api.openai.com/v1"),
                              llm_model=cfg["llm_model"], api_key=cfg.get("api_key"),
                              api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"), name=name)
    if t == "deepgram_agent":
        from .deepgram_agent import DeepgramAgentBackend
        return DeepgramAgentBackend(listen_model=cfg.get("listen_model", "nova-3"),
                                    think_provider=cfg.get("think_provider", "open_ai"),
                                    think_model=cfg.get("think_model", "gpt-4o-mini"),
                                    api_key=cfg.get("api_key"), name=name)
    if t == "assemblyai_agent":
        from .assemblyai_agent import AssemblyAIAgentBackend
        return AssemblyAIAgentBackend(api_key=cfg.get("api_key"), voice=cfg.get("voice"), name=name)
    if t == "elevenlabs_agent":
        from .elevenlabs_agent import ElevenLabsAgentBackend
        return ElevenLabsAgentBackend(agent_id=cfg["agent_id"], api_key=cfg.get("api_key"),
                                      override_prompt=cfg.get("override_prompt", False), name=name)
    if t == "vapi":
        from .vapi import VapiBackend
        return VapiBackend(assistant_id=cfg["assistant_id"], api_key=cfg.get("api_key"), name=name)
    raise ValueError(f"unknown agent type {t!r}")


AGENT_TYPES = ["mock", "openai_realtime", "gemini_live", "deepgram_agent", "assemblyai_agent",
               "elevenlabs_agent", "vapi", "audio_llm", "cascade"]
SHADOW_TYPES = ["mock", "openai_stt", "deepgram", "faster_whisper"]
