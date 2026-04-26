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




class TestRussianTTS:
    """Tests for Russian TTS routing in _synthesise_sentence."""

    def _make_agent(self):
        import numpy as np
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        mock_tts.generate_audio = MagicMock(
            return_value=np.zeros(1000, dtype=np.float32)
        )

        mock_ru_tts = MagicMock()
        mock_ru_tts.apply_tts = MagicMock(return_value=np.zeros(1000, dtype=np.float32))

        mock_openai = MagicMock()
        resources = SharedResources(
            openai_client=mock_openai,
            tts_model=mock_tts,
            voice_state=MagicMock(),
            ru_tts_model=mock_ru_tts,
            ru_speaker="baya",
            ru_sample_rate=24000,
            vosk_model_en=MagicMock(),
            vosk_model_ru=None,
        )

        with patch("core.conversation.RealtimeAudioProcessor"), patch(
            "core.conversation.LessonRAG"
        ) as mock_rag:
            mock_rag_inst = MagicMock()
            mock_rag.return_value = mock_rag_inst
            mock_rag_inst.ingest = MagicMock()
            mock_rag_inst.build_system_prompt = MagicMock(return_value="sys")
            mock_rag_inst.chunk_count = 0
            mock_rag_inst.lesson_title = "T"

            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop, resources=resources)

        return agent, mock_tts, mock_ru_tts

    def test_english_uses_pocket_tts(self):
        agent, mock_tts, mock_ru_tts = self._make_agent()
        agent._synthesise_sentence("Hello, how are you today?")
        mock_tts.generate_audio.assert_called_once()
        mock_ru_tts.apply_tts.assert_not_called()

    def test_russian_uses_silero_tts(self):
        agent, mock_tts, mock_ru_tts = self._make_agent()
        agent._synthesise_sentence("Привет, как дела?")
        mock_ru_tts.apply_tts.assert_called_once_with(
            text="Привет, как дела?",
            speaker="baya",
            sample_rate=24000,
        )
        mock_tts.generate_audio.assert_not_called()

    def test_russian_fallback_when_model_missing(self):
        agent, mock_tts, mock_ru_tts = self._make_agent()
        from dataclasses import replace

        agent._resources = replace(agent._resources, ru_tts_model=None)
        agent._synthesise_sentence("Привет, как дела?")
        mock_tts.generate_audio.assert_called_once()

    def test_emotion_style_boosts_english_positive_sentence(self):
        from dataclasses import replace

        from core.config import RUNTIME_CONFIG

        agent, mock_tts, _ = self._make_agent()
        tts_cfg = replace(RUNTIME_CONFIG.tts, enable_emotion=True, emotion_strength=1.0)
        runtime_cfg = replace(RUNTIME_CONFIG, tts=tts_cfg)

        with patch("core.conversation.RUNTIME_CONFIG", runtime_cfg):
            agent._synthesise_sentence("Great work.")

        rendered_text = mock_tts.generate_audio.call_args.args[1]
        assert rendered_text.endswith("!")


