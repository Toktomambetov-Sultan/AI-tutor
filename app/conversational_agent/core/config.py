"""Runtime configuration for conversational agent.

Environment values are parsed in one place so low-latency and quality
tuning can be adjusted without changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


@dataclass(frozen=True)
class AudioRuntimeConfig:
    vad_threshold: float
    silence_sec: float
    min_silence_sec: float
    max_silence_sec: float
    silence_sec_with_partial_text: float
    min_speech_sec: float
    max_utterance_sec: float
    pre_speech_chunks: int
    partial_update_every_frames: int
    torch_num_threads: int


@dataclass(frozen=True)
class LLMRuntimeConfig:
    model: str
    summary_model: str
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int


@dataclass(frozen=True)
class RuntimeConfig:
    audio: AudioRuntimeConfig
    llm: LLMRuntimeConfig


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        audio=AudioRuntimeConfig(
            vad_threshold=_get_env_float("AGENT_VAD_THRESHOLD", 0.45),
            silence_sec=_get_env_float("AGENT_SILENCE_SEC", 0.6),
            min_silence_sec=_get_env_float("AGENT_MIN_SILENCE_SEC", 0.25),
            max_silence_sec=_get_env_float("AGENT_MAX_SILENCE_SEC", 0.9),
            silence_sec_with_partial_text=_get_env_float(
                "AGENT_SILENCE_SEC_WITH_PARTIAL_TEXT", 0.25
            ),
            min_speech_sec=_get_env_float("AGENT_MIN_SPEECH_SEC", 0.35),
            max_utterance_sec=_get_env_float("AGENT_MAX_UTTERANCE_SEC", 4.0),
            pre_speech_chunks=_get_env_int("AGENT_PRE_SPEECH_CHUNKS", 6),
            partial_update_every_frames=_get_env_int(
                "AGENT_PARTIAL_UPDATE_EVERY_FRAMES", 2
            ),
            torch_num_threads=max(1, _get_env_int("AGENT_TORCH_NUM_THREADS", 1)),
        ),
        llm=LLMRuntimeConfig(
            model=_get_env_str("AGENT_CHAT_MODEL", "gpt-4o-mini"),
            summary_model=_get_env_str("AGENT_SUMMARY_MODEL", "gpt-4o-mini"),
            temperature=_get_env_float("AGENT_LLM_TEMPERATURE", 0.35),
            top_p=_get_env_float("AGENT_LLM_TOP_P", 0.9),
            presence_penalty=_get_env_float("AGENT_LLM_PRESENCE_PENALTY", 0.0),
            frequency_penalty=_get_env_float("AGENT_LLM_FREQUENCY_PENALTY", 0.15),
            max_tokens=max(48, _get_env_int("AGENT_LLM_MAX_TOKENS", 120)),
        ),
    )


RUNTIME_CONFIG = load_runtime_config()
