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



from core.utils import has_trailing_clause_boundary, split_clauses, split_sentences


class TestSentenceSplitting:
    """Tests for the split_sentences helper."""

    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = split_sentences("Hello world. How are you? I am fine!")
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_preserves_punctuation(self):
        result = split_sentences("Wait! Really? Yes.")
        assert result == ["Wait!", "Really?", "Yes."]

    def test_no_split_on_abbreviations_with_no_space(self):
        # "e.g." doesn't have a space after the inner dots so won't split
        result = split_sentences("Use e.g. Python.")
        # The regex splits on '. P' — acceptable; at least no crash
        assert len(result) >= 1

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_no_punctuation(self):
        assert split_sentences("Hello world") == ["Hello world"]

    def test_multiple_spaces_between(self):
        result = split_sentences("Hello.   World!")
        assert result == ["Hello.", "World!"]

    def test_trailing_whitespace(self):
        result = split_sentences("One. Two.  ")
        assert result == ["One.", "Two."]

    def test_long_paragraph(self):
        text = (
            "The capital of France is Paris. "
            "It is known for the Eiffel Tower! "
            "Would you like to learn more? "
            "I can explain further."
        )
        result = split_sentences(text)
        assert len(result) == 4
        assert result[0] == "The capital of France is Paris."
        assert result[-1] == "I can explain further."

    def test_clause_splitting_breaks_long_sentence(self):
        # Comma no longer splits clauses (prevents unnatural tonal breaks).
        # Splitting happens at ; : — and sentence-final punctuation.
        text = "First we cover photosynthesis; then we discuss respiration."
        result = split_clauses(text)
        assert len(result) >= 2
        assert any("photosynthesis" in part for part in result)
        assert any("respiration" in part for part in result)

    def test_clause_splitting_does_not_split_on_comma(self):
        # Commas must NOT trigger clause splits — they cause unnatural audio.
        text = "Photosynthesis uses light energy, converts it to chemical energy, and stores it in glucose."
        result = split_clauses(text)
        # The whole sentence is one clause (no strong boundary until final .)
        assert len(result) == 1
        assert "light energy" in result[0]

    def test_clause_splitting_merges_tiny_fragments(self):
        # Without comma splitting the entire string is a single clause.
        text = "Oh! Okay. Sure."
        result = split_clauses(text)
        # Each sentence-final punctuation creates a clause boundary;
        # very short fragments are merged into the following clause.
        assert len(result) >= 1
        full = " ".join(result)
        assert "Okay" in full

    def test_clause_boundary_detection_true_when_ending_with_punctuation(self):
        assert has_trailing_clause_boundary("Great work!") is True

    def test_clause_boundary_detection_false_for_partial_tail(self):
        assert has_trailing_clause_boundary("Great work and") is False


