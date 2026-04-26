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




class TestSileroVADAudioProcessor:
    """Tests for the Silero-based RealtimeAudioProcessor (mocked model)."""

    def test_adaptive_silence_shortens_for_fast_speaker(self):
        """Feed several short utterance lengths and verify silence_sec decreases."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                # Simulate 5 short utterances (~1.5s each)
                for _ in range(5):
                    short_bytes = int(1.5 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                    proc._adapt_silence_threshold(short_bytes)

                # Silence should have moved toward the minimum
                assert proc.silence_sec < 1.3, f"Expected < 1.3, got {proc.silence_sec}"

    def test_adaptive_silence_lengthens_for_slow_speaker(self):
        """Feed several long utterance lengths and verify silence_sec increases."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                # Simulate 5 long utterances (~7s each)
                for _ in range(5):
                    long_bytes = int(7.0 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                    proc._adapt_silence_threshold(long_bytes)

                # Silence should have moved toward the maximum
                assert proc.silence_sec > 2.0, f"Expected > 2.0, got {proc.silence_sec}"

    def test_adaptive_needs_minimum_data(self):
        """With only 1 utterance, silence_sec shouldn't change."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                original = proc.silence_sec
                proc._adapt_silence_threshold(
                    int(3.0 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                )
                assert proc.silence_sec == original


