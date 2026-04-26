"""
test_streaming.py — Unit tests for the streaming conversation pipeline.

Tests sentence splitting, interrupt handling, TTS queueing, and the
LLM-streaming + sentence-chunked TTS flow WITHOUT requiring actual
models / gRPC / audio hardware.

Run:
    cd /path/to/conversational_agent
    OPENAI_API_KEY=sk-... pytest __tests__/test_streaming.py -v -p no:ethereum
"""

import asyncio
import io
import sys
import threading
import time
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ─────────────────────────────────────────────────────────────────────
# Stub out heavy C-extension / Docker-only dependencies so the module
# can be imported on a dev machine without pocket_tts, grpc stubs, etc.
# ─────────────────────────────────────────────────────────────────────

try:
    import pocket_tts  # noqa: F401
except ImportError:
    _pocket_tts = ModuleType("pocket_tts")
    _pocket_tts.TTSModel = MagicMock()
    sys.modules["pocket_tts"] = _pocket_tts

try:
    import torch  # noqa: F401
except ImportError:
    _torch = ModuleType("torch")
    _torch.is_tensor = lambda x: False
    sys.modules["torch"] = _torch

# ── scipy: only stub if it's truly missing ──
try:
    import scipy.io.wavfile  # noqa: F401
except ImportError:
    _scipy = ModuleType("scipy")
    _scipy_io = ModuleType("scipy.io")
    _scipy_wavfile = ModuleType("scipy.io.wavfile")
    _scipy_wavfile.write = MagicMock()
    _scipy.io = _scipy_io
    _scipy_io.wavfile = _scipy_wavfile
    sys.modules["scipy"] = _scipy
    sys.modules["scipy.io"] = _scipy_io
    sys.modules["scipy.io.wavfile"] = _scipy_wavfile

try:
    import vosk  # noqa: F401
except ImportError:
    _vosk = ModuleType("vosk")
    _vosk.Model = MagicMock
    _vosk.KaldiRecognizer = MagicMock
    _vosk.SetLogLevel = MagicMock()
    sys.modules["vosk"] = _vosk

# ── openai: ensure the OpenAI class exists (old pip version lacks it) ──
try:
    from openai import OpenAI  # noqa: F401
except (ImportError, AttributeError):
    import openai as _openai_mod

    _openai_mod.OpenAI = MagicMock()

# ── torch.hub: stub torch.hub.load for Silero VAD ──
_torch_mod = sys.modules.get("torch") or sys.modules.setdefault(
    "torch", ModuleType("torch")
)
if not hasattr(_torch_mod, "hub"):
    _torch_hub = ModuleType("torch.hub")
    _torch_hub.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    _torch_mod.hub = _torch_hub
    sys.modules["torch.hub"] = _torch_hub
if not hasattr(_torch_mod, "FloatTensor"):
    _torch_mod.FloatTensor = MagicMock()
if not hasattr(_torch_mod, "no_grad"):
    _torch_mod.no_grad = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )

# ── chromadb: stub only if truly missing ──
try:
    import chromadb  # noqa: F401
except ImportError:
    _chromadb = ModuleType("chromadb")
    _chromadb.Client = MagicMock()
    sys.modules["chromadb"] = _chromadb




class TestReducedVADThresholds:
    """Verify the default VAD silence thresholds are reduced for faster response."""

    def test_default_silence_values(self):
        import inspect
        from core.audio_processor import RealtimeAudioProcessor

        sig = inspect.signature(RealtimeAudioProcessor.__init__)
        assert sig.parameters["silence_sec"].default == 0.45
        assert sig.parameters["min_silence_sec"].default == 0.25
        assert sig.parameters["max_silence_sec"].default == 0.9


class TestLatencyGuardrails:
    """Tests that enforce low-latency turn endpointing constraints."""

    def test_latency_budget_defaults_under_five_seconds(self):
        from core.config import RUNTIME_CONFIG

        assert RUNTIME_CONFIG.audio.silence_sec < 5.0
        assert RUNTIME_CONFIG.audio.silence_sec_with_partial_text < 5.0
        assert RUNTIME_CONFIG.audio.max_utterance_sec < 5.0

    def test_partial_text_uses_faster_silence_window(self):
        from core.audio_processor import RealtimeAudioProcessor

        proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
        proc.silence_sec = 0.9
        proc._silence_sec_with_partial_text = 0.25
        proc._latest_partial_text = "hello"

        effective = proc._effective_silence_sec()
        assert effective == 0.25

        frames = proc._silence_frames_needed(0.032)
        assert frames * 0.032 < 0.5

    def test_force_finalize_timeout_triggers_before_five_seconds(self):
        from core.audio_processor import RealtimeAudioProcessor

        proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
        proc._max_utterance_sec = 4.0
        proc._speech_frames = int(4.0 / 0.032)
        proc._silence_frames = 1
        proc._latest_partial_text = "hello"
        proc.min_speech_bytes = 1000
        proc._pcm_buf = bytearray(b"x" * 5000)
        proc.sample_rate = 16000

        assert proc._should_force_finalize(0.032)


class TestSTTTranscriptFallback:
    """Regression tests for transcript selection in RealtimeAudioProcessor."""

    def test_pick_transcript_prefers_final(self):
        from core.audio_processor import RealtimeAudioProcessor

        text = RealtimeAudioProcessor._pick_transcript_text(
            "final transcript", "partial transcript"
        )
        assert text == "final transcript"

    def test_pick_transcript_uses_partial_when_final_empty(self):
        from core.audio_processor import RealtimeAudioProcessor

        text = RealtimeAudioProcessor._pick_transcript_text("", "partial transcript")
        assert text == "partial transcript"


class TestLLMStreamingParameters:
    """Ensure runtime LLM settings are applied for stable spoken replies."""

    def _make_agent(self):
        import numpy as np
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        mock_tts.generate_audio = MagicMock(
            return_value=np.zeros(1000, dtype=np.float32)
        )
        mock_openai = MagicMock()
        resources = SharedResources(
            openai_client=mock_openai,
            tts_model=mock_tts,
            voice_state=MagicMock(),
            ru_tts_model=None,
            ru_speaker=None,
            ru_sample_rate=None,
            vosk_model_en=MagicMock(),
            vosk_model_ru=None,
        )

        with patch("core.conversation.RealtimeAudioProcessor"), patch(
            "core.conversation.LessonRAG"
        ):
            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop, resources=resources)

        return agent, mock_openai, loop

    def test_stream_uses_runtime_llm_parameters(self):
        from core.config import RUNTIME_CONFIG

        agent, mock_openai, loop = self._make_agent()
        try:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Short answer."

            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter([chunk])
            mock_openai.chat.completions.create.return_value = mock_stream

            agent._stream_llm_and_speak()

            kwargs = mock_openai.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == RUNTIME_CONFIG.llm.model
            assert kwargs["temperature"] == RUNTIME_CONFIG.llm.temperature
            assert kwargs["top_p"] == RUNTIME_CONFIG.llm.top_p
            assert kwargs["presence_penalty"] == RUNTIME_CONFIG.llm.presence_penalty
            assert kwargs["frequency_penalty"] == RUNTIME_CONFIG.llm.frequency_penalty
            assert kwargs["max_tokens"] == RUNTIME_CONFIG.llm.max_tokens
        finally:
            loop.close()


