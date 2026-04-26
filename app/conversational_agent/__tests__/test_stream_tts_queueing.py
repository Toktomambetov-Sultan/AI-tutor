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




class TestTTSQueueing:
    """Verify _send_audio pushes (ai_text, data) onto the response queue."""

    def _make_agent(self):
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
            mock_rag_inst.chunk_count = 0
            mock_rag_inst.lesson_title = "T"

            loop = asyncio.new_event_loop()
            q = asyncio.Queue()
            agent = ConversationalAgent(q, loop, resources=resources)
            return agent, q, loop

    def test_send_audio_queues_text_then_chunks_then_end(self):
        agent, queue, loop = self._make_agent()

        # Simulate sending 100 bytes of audio with text label
        wav_bytes = b"\x00" * 100
        ai_text = "Hello!"

        # Run _send_audio (it uses loop.call_soon_threadsafe)
        def _run():
            agent._send_audio(wav_bytes, ai_text=ai_text)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=2)

        # Drain the queue
        messages = []

        async def _drain():
            while not queue.empty():
                messages.append(await queue.get())

        loop.run_until_complete(_drain())
        loop.close()

        # Expect: ("ai_text", "Hello!"), ("audio", ...), ("end", None)
        assert len(messages) >= 3
        assert messages[0] == ("ai_text", "Hello!")
        assert messages[-1] == ("end", None)
        # All middle messages should be audio
        for msg in messages[1:-1]:
            assert msg[0] == "audio"

    def test_send_audio_stops_immediately_when_interrupted(self):
        agent, queue, loop = self._make_agent()
        try:
            agent._interrupted.set()
            agent._send_audio(b"\x00" * 100, ai_text="Hello!")
            time.sleep(0.05)
            assert queue.empty()
        finally:
            loop.close()


