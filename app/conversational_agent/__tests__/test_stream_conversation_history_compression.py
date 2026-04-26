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




class TestHistoryCompression:
    """Tests for _compress_history in ConversationalAgent."""

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

    def test_no_compression_below_threshold(self):
        agent, mock_openai = self._make_agent()
        # Start with system + a few turns (well below _MAX_MESSAGES=20)
        agent.messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        agent._compress_history()
        # Should remain unchanged
        assert len(agent.messages) == 3
        mock_openai.chat.completions.create.assert_not_called()

    def test_compression_above_threshold(self):
        agent, mock_openai = self._make_agent()
        # Build 25 messages (above _MAX_MESSAGES=20)
        agent.messages = [{"role": "system", "content": "system prompt"}]
        for i in range(12):
            agent.messages.append({"role": "user", "content": f"Question {i}"})
            agent.messages.append({"role": "assistant", "content": f"Answer {i}"})

        assert len(agent.messages) == 25

        # Mock the summarization response
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Summary of the conversation."
        mock_openai.chat.completions.create.return_value = mock_resp

        agent._compress_history()

        # Should be: system + summary + last 6 messages = 8
        assert len(agent.messages) == 8
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "system prompt"
        assert "[CONVERSATION SUMMARY]" in agent.messages[1]["content"]
        assert "Summary of the conversation." in agent.messages[1]["content"]

    def test_compression_failure_leaves_messages_intact(self):
        agent, mock_openai = self._make_agent()
        agent.messages = [{"role": "system", "content": "system prompt"}]
        for i in range(12):
            agent.messages.append({"role": "user", "content": f"Q {i}"})
            agent.messages.append({"role": "assistant", "content": f"A {i}"})

        original_count = len(agent.messages)

        # Make summarization fail
        mock_openai.chat.completions.create.side_effect = Exception("API error")

        agent._compress_history()

        # Messages should be unchanged
        assert len(agent.messages) == original_count


