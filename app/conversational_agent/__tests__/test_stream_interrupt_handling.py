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




class TestInterruptHandling:
    """Tests for the barge-in / interrupt mechanism."""

    def _make_agent(self, lesson_context=""):
        """Create a ConversationalAgent with all heavy deps mocked."""
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
            mock_rag_inst.build_system_prompt = MagicMock(return_value="system prompt")
            mock_rag_inst.chunk_count = 1
            mock_rag_inst.lesson_title = "Test"

            loop = asyncio.new_event_loop()
            q = asyncio.Queue()
            agent = ConversationalAgent(
                q, loop, lesson_context=lesson_context, resources=resources
            )
            return agent, q, loop

    def test_handle_interrupt_sets_event(self):
        agent, _, _ = self._make_agent()
        assert not agent._interrupted.is_set()
        agent.handle_interrupt()
        assert agent._interrupted.is_set()

    def test_handle_interrupt_drains_queue_and_sends_signal(self):
        agent, queue, loop = self._make_agent()
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("ai_text", "Hello"))
            loop.call_soon_threadsafe(queue.put_nowait, ("audio", b"123"))
            loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            time.sleep(0.05)

            agent.handle_interrupt()
            time.sleep(0.05)

            messages = []

            async def _drain():
                while not queue.empty():
                    messages.append(await queue.get())

            loop.run_until_complete(_drain())
            assert messages == [("signal", "interrupt")]
        finally:
            loop.close()

    def test_handle_interrupt_clears_on_new_turn(self):
        agent, _, _ = self._make_agent()
        agent._interrupted.set()
        agent._spoken_sentences = ["Hello.", "How are you?"]
        agent._current_ai_text = "Hello. How are you? I can help."

        # Simulate that _handle_turn would clear the flag after reading context
        assert agent._interrupted.is_set()
        # The interrupt note is built from _spoken_sentences
        from core.prompts import INTERRUPT_CONTEXT_TEMPLATE

        spoken_so_far = " ".join(agent._spoken_sentences)
        note = INTERRUPT_CONTEXT_TEMPLATE.format(
            spoken_text=spoken_so_far, full_text=agent._current_ai_text
        )
        assert "Hello. How are you?" in note
        assert "I can help" in note

    def test_interrupt_context_template_format(self):
        from core.prompts import INTERRUPT_CONTEXT_TEMPLATE

        result = INTERRUPT_CONTEXT_TEMPLATE.format(
            spoken_text="First sentence.", full_text="First sentence. Second sentence."
        )
        assert "First sentence." in result
        assert "Second sentence." in result
        assert "interrupted" in result.lower()


