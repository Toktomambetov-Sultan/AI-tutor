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




class TestVoskStreamingSTT:
    """Test that ConversationalAgent uses Vosk streaming STT correctly."""

    def _make_agent(self):
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        mock_openai = MagicMock()
        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
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

        return agent, mock_openai

    def test_recognizer_parameter_accepted(self):
        """Agent init should succeed and create a processor with a recognizer."""
        agent, _ = self._make_agent()
        # Processor is mocked, just verify agent was created
        assert agent._processor is not None

    def test_handle_turn_receives_text(self):
        """_handle_turn now receives text (str), not PCM bytes."""
        agent, mock_openai = self._make_agent()
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_openai.chat.completions.create.return_value = mock_stream

        agent._lesson_ready = False
        agent._handle_turn("Hello world")

        assert any(
            msg.get("role") == "user" and "Hello world" in msg.get("content", "")
            for msg in agent.messages
        )

    def test_empty_text_skipped(self):
        """Empty/whitespace-only transcriptions should be ignored."""
        agent, mock_openai = self._make_agent()
        agent._lesson_ready = False
        agent._handle_turn("   ")

        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        assert len(user_msgs) == 0
        mock_openai.chat.completions.create.assert_not_called()


class TestLanguageFromMaterials:
    """Test that language is detected from lesson materials."""

    def _make_agent(self, lesson_context=""):
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        resources = SharedResources(
            openai_client=MagicMock(),
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
        ) as mock_rag:
            mock_rag_inst = MagicMock()
            mock_rag.return_value = mock_rag_inst
            mock_rag_inst.ingest = MagicMock()
            mock_rag_inst.build_system_prompt = MagicMock(return_value="sys")
            mock_rag_inst.chunk_count = 1
            mock_rag_inst.lesson_title = "T"

            loop = asyncio.new_event_loop()
            q = asyncio.Queue()
            agent = ConversationalAgent(
                q, loop, lesson_context=lesson_context, resources=resources
            )

        return agent

    def test_english_materials_set_en(self):
        agent = self._make_agent("Photosynthesis is the process plants use.")
        assert agent._language == "en"

    def test_russian_materials_set_ru(self):
        agent = self._make_agent(
            "Фотосинтез — это процесс, при котором растения используют свет."
        )
        assert agent._language == "ru"

    def test_no_materials_defaults_to_en(self):
        agent = self._make_agent("")
        assert agent._language == "en"


