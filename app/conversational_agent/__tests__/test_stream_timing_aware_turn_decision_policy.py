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




class TestTurnDecisionPolicy:
    """Tests for the ``decide_turn`` pure function in ``core.turn_policy``."""

    def test_complete_utterance_many_words_responds(self):
        from core.turn_policy import decide_turn

        # Well above the default min_words_for_complete (4)
        assert decide_turn("I think photosynthesis uses sunlight energy") == "respond"

    def test_sentence_ending_punctuation_responds(self):
        from core.turn_policy import decide_turn

        # Short but ends with "."
        assert decide_turn("Yes.") == "respond"

    def test_question_mark_responds(self):
        from core.turn_policy import decide_turn

        assert decide_turn("Really?") == "respond"

    def test_exclamation_mark_responds(self):
        from core.turn_policy import decide_turn

        assert decide_turn("Wow!") == "respond"

    def test_partial_short_no_punctuation_waits(self):
        from core.turn_policy import decide_turn

        # 1 word, no punctuation → WAIT
        assert decide_turn("Um") == "wait"

    def test_two_words_no_punctuation_waits(self):
        from core.turn_policy import decide_turn

        assert decide_turn("I think") == "wait"

    def test_empty_string_responds(self):
        from core.turn_policy import decide_turn

        # Edge case: empty text — no words, no boundary → "wait",
        # but feed() strips+skips empties before calling decide_turn.
        # decide_turn itself returns "wait" for empty string (0 < 4).
        assert decide_turn("") == "wait"

    def test_exactly_min_words_responds(self):
        from core.turn_policy import decide_turn

        # Default min_words_for_complete is 4; exactly 4 words → respond
        assert decide_turn("one two three four") == "respond"

    def test_low_quality_ending_with_short_recent_silence_waits(self):
        from core.turn_policy import decide_turn

        text = "I was thinking about"
        assert (
            decide_turn(
                text,
                silence_after_end_sec=0.10,
                recent_silence_sec=0.15,
            )
            == "wait"
        )

    def test_low_quality_ending_with_long_silence_responds(self):
        from core.turn_policy import decide_turn

        text = "I was thinking about"
        assert (
            decide_turn(
                text,
                silence_after_end_sec=1.10,
                recent_silence_sec=1.00,
            )
            == "respond"
        )


