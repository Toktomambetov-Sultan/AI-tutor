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




class TestNewPatchBehaviors:
    def _make_agent(self):
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        import numpy as np

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        mock_tts.generate_audio = MagicMock(return_value=np.zeros(64, dtype=np.float32))

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
            q = asyncio.Queue()
            agent = ConversationalAgent(q, loop, resources=resources)

        return agent, q, loop

    def test_low_confidence_transcript_triggers_repeat_prompt(self):
        from core.prompts import STT_CLARIFY_FALLBACK

        agent, _, loop = self._make_agent()
        try:
            agent._speak_direct_text = MagicMock(return_value=STT_CLARIFY_FALLBACK)
            agent._stream_llm_and_speak = MagicMock()
            agent._reset_silence_timer = MagicMock()

            agent._handle_turn("????")

            agent._speak_direct_text.assert_called_once_with(STT_CLARIFY_FALLBACK)
            agent._stream_llm_and_speak.assert_not_called()
        finally:
            loop.close()

    def test_explicit_finish_request_emits_lesson_end_signal(self):
        agent, q, loop = self._make_agent()
        try:
            agent._speak_direct_text = MagicMock(return_value="Great work today.")
            agent._reset_silence_timer = MagicMock()

            agent._handle_turn("Can we finish the lesson now?")

            msgs = []

            async def _drain():
                while not q.empty():
                    msgs.append(await q.get())

            loop.run_until_complete(_drain())
            assert any(m[0] == "signal" and m[1] == "lesson_end" for m in msgs)
        finally:
            loop.close()

    def test_proactive_silence_followup_calls_llm(self):
        agent, _, loop = self._make_agent()
        try:
            agent._stream_llm_and_speak = MagicMock(return_value="Let us continue.")
            agent._reset_silence_timer = MagicMock()
            agent._process_pending_utterance = MagicMock()

            agent._on_silence_timeout(8.0)
            time.sleep(0.15)

            agent._stream_llm_and_speak.assert_called_once()
            kwargs = agent._stream_llm_and_speak.call_args.kwargs
            assert "silent for 8.0 seconds" in kwargs["extra_system_msg"]
        finally:
            loop.close()


class TestTranscriptQualityHeuristics:
    def test_detects_low_confidence_noise(self):
        from core.utils import is_low_confidence_transcript

        assert is_low_confidence_transcript("????") is True
        assert is_low_confidence_transcript("a") is True

    def test_keeps_normal_short_acknowledgements_out_of_low_confidence(self):
        from core.utils import is_low_confidence_transcript

        assert is_low_confidence_transcript("okay") is False

    def test_detects_explicit_lesson_end_request(self):
        from core.utils import is_lesson_end_request

        assert is_lesson_end_request("Please finish the lesson") is True
        assert is_lesson_end_request("Can we end class now?") is True

from core.turn_policy import calculate_proactive_delay
from core.config import RUNTIME_CONFIG

def test_calculate_proactive_delay_none():
    assert calculate_proactive_delay(None) == RUNTIME_CONFIG.turn_policy.proactive_silence_sec

def test_calculate_proactive_delay_text():
    text = "Hello there. How are you?"
    delay = calculate_proactive_delay(text)
    assert delay > RUNTIME_CONFIG.turn_policy.proactive_silence_sec
    assert delay <= RUNTIME_CONFIG.turn_policy.max_proactive_silence_sec
