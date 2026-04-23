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


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
class TTSRuntimeConfig:
    enable_emotion: bool
    emotion_strength: float


@dataclass(frozen=True)
class TurnPolicyConfig:
    """Timing-aware turn decision thresholds."""

    # Utterances with fewer words than this AND no terminal punctuation
    # are treated as partial thoughts that should wait for continuation.
    min_words_for_complete: int
    # Maximum seconds of silence after a partial utterance before the
    # agent is forced to reply regardless.
    force_reply_sec: float
    # Minimum recent silence window required before treating a boundary-less
    # complete-length utterance as ready to respond.
    min_recent_silence_sec: float
    # If a likely partial thought is followed by at least this much silence,
    # allow an early response without waiting for full force-reply timeout.
    partial_respond_silence_sec: float
    # Minimum silence accepted for punctuation-ended utterances to reduce
    # over-eager responses on clipped ASR boundaries.
    min_strong_boundary_silence_sec: float


@dataclass(frozen=True)
class TempoConfig:
    """Speech tempo adaptation settings."""

    adapt_enabled: bool
    # WPM below this value → SLOW hint
    slow_wpm_threshold: float
    # WPM at or above this value → FAST hint
    fast_wpm_threshold: float
    # Synthetic first-turn timing prior used when no timing history exists.
    neutral_turn_duration_sec: float
    # Lower/upper bounds used to clamp per-turn timing in safe ranges.
    min_turn_duration_sec: float
    max_turn_duration_sec: float


@dataclass(frozen=True)
class RuntimeConfig:
    audio: AudioRuntimeConfig
    llm: LLMRuntimeConfig
    tts: TTSRuntimeConfig
    turn_policy: TurnPolicyConfig
    tempo: TempoConfig


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
        tts=TTSRuntimeConfig(
            enable_emotion=_get_env_bool("AGENT_TTS_ENABLE_EMOTION", True),
            emotion_strength=min(
                1.0, max(0.0, _get_env_float("AGENT_TTS_EMOTION_STRENGTH", 0.5))
            ),
        ),
        turn_policy=TurnPolicyConfig(
            min_words_for_complete=max(1, _get_env_int("AGENT_TURN_MIN_WORDS", 4)),
            force_reply_sec=max(0.5, _get_env_float("AGENT_TURN_FORCE_REPLY_SEC", 2.2)),
            min_recent_silence_sec=max(
                0.0, _get_env_float("AGENT_TURN_MIN_RECENT_SILENCE_SEC", 0.3)
            ),
            partial_respond_silence_sec=max(
                0.2,
                _get_env_float("AGENT_TURN_PARTIAL_RESPOND_SILENCE_SEC", 0.9),
            ),
            min_strong_boundary_silence_sec=max(
                0.0,
                _get_env_float("AGENT_TURN_MIN_STRONG_BOUNDARY_SILENCE_SEC", 0.08),
            ),
        ),
        tempo=TempoConfig(
            adapt_enabled=_get_env_bool("AGENT_TEMPO_ADAPT_ENABLED", True),
            slow_wpm_threshold=_get_env_float("AGENT_TEMPO_SLOW_WPM", 80.0),
            fast_wpm_threshold=_get_env_float("AGENT_TEMPO_FAST_WPM", 160.0),
            neutral_turn_duration_sec=max(
                0.5, _get_env_float("AGENT_TEMPO_NEUTRAL_TURN_DURATION_SEC", 2.5)
            ),
            min_turn_duration_sec=max(
                0.1, _get_env_float("AGENT_TEMPO_MIN_TURN_DURATION_SEC", 0.4)
            ),
            max_turn_duration_sec=max(
                1.0, _get_env_float("AGENT_TEMPO_MAX_TURN_DURATION_SEC", 12.0)
            ),
        ),
    )


RUNTIME_CONFIG = load_runtime_config()
