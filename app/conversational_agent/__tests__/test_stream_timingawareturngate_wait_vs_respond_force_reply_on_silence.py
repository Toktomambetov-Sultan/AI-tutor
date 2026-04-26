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




class TestTimingAwareTurnGate:
    """Integration tests for ``TimingAwareTurnGate``."""

    def test_complete_utterance_dispatched_immediately(self):
        from core.turn_policy import TimingAwareTurnGate

        received = []
        gate = TimingAwareTurnGate(lambda t: received.append(t))
        gate.feed("I understand photosynthesis now.")
        assert received == ["I understand photosynthesis now."]

    def test_partial_utterance_not_dispatched_immediately(self):
        from core.turn_policy import TimingAwareTurnGate

        received = []
        gate = TimingAwareTurnGate(lambda t: received.append(t))
        gate.feed("Um")  # 1 word, no punctuation → wait
        # Nothing dispatched yet
        assert received == []
        gate.close()  # cancel timer

    def test_force_reply_on_silence_after_partial(self):
        """After a partial utterance, the gate must dispatch after force_reply_sec."""
        from unittest.mock import patch
        from core.turn_policy import TimingAwareTurnGate
        from dataclasses import replace

        received = []
        gate = TimingAwareTurnGate(lambda t: received.append(t))

        # Patch RUNTIME_CONFIG.turn_policy so force_reply_sec is tiny (0.05s)
        import core.turn_policy as _tp_mod
        import core.config as _cfg_mod

        original_cfg = _cfg_mod.RUNTIME_CONFIG
        fast_turn_policy = replace(original_cfg.turn_policy, force_reply_sec=0.05)
        fast_cfg = replace(original_cfg, turn_policy=fast_turn_policy)

        with patch.object(_cfg_mod, "RUNTIME_CONFIG", fast_cfg), patch.object(
            _tp_mod, "RUNTIME_CONFIG", fast_cfg
        ):
            gate2 = TimingAwareTurnGate(lambda t: received.append(t))
            gate2.feed("Hmm")  # partial → wait, timer starts at 0.05s
            time.sleep(0.2)  # wait past the force-reply window

        assert received == ["Hmm"], f"Expected force-reply, got: {received}"

    def test_second_partial_merges_and_resets_timer(self):
        """Two consecutive partials should be merged; the gate decides on combined text."""
        from core.turn_policy import TimingAwareTurnGate

        received = []
        gate = TimingAwareTurnGate(lambda t: received.append(t))
        # Feed two short partials quickly; together they exceed min_words threshold
        gate.feed("I think")  # 2 words → wait
        gate.feed("it is clear")  # merged: "I think it is clear" = 5 words → respond
        assert received == ["I think it is clear"]
        gate.close()

    def test_close_cancels_pending_timer(self):
        """Closing the gate must prevent force-reply from firing."""
        from core.turn_policy import TimingAwareTurnGate

        received = []
        gate = TimingAwareTurnGate(lambda t: received.append(t))
        gate.feed("Uh")  # partial → timer starts
        gate.close()  # must cancel timer
        time.sleep(0.3)  # wait past any reasonable default
        assert received == [], f"Expected no dispatch after close, got: {received}"

    def test_force_reply_passes_elapsed_sec_to_callback(self):
        """T02: force-reply should include elapsed silence in callback kwargs."""
        from unittest.mock import patch
        from dataclasses import replace
        from core.turn_policy import TimingAwareTurnGate
        import core.turn_policy as _tp_mod
        import core.config as _cfg_mod

        received = []

        def _cb(text, **kwargs):
            received.append((text, kwargs.get("elapsed_sec")))

        original_cfg = _cfg_mod.RUNTIME_CONFIG
        fast_turn_policy = replace(original_cfg.turn_policy, force_reply_sec=0.05)
        fast_cfg = replace(original_cfg, turn_policy=fast_turn_policy)

        with patch.object(_cfg_mod, "RUNTIME_CONFIG", fast_cfg), patch.object(
            _tp_mod, "RUNTIME_CONFIG", fast_cfg
        ):
            gate = TimingAwareTurnGate(_cb)
            gate.feed("Hmm")
            time.sleep(0.2)

        assert len(received) == 1
        text, elapsed = received[0]
        assert text == "Hmm"
        assert elapsed is not None and elapsed >= 0.04


class TestFillerInterruptSuppression:
    """T04: filler back-channels should not interrupt active AI turn."""

    def test_filler_utterance_does_not_trigger_interrupt(self):
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
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

        try:
            # Simulate active AI turn.
            with agent._processing_lock:
                agent._processing = True
            agent.handle_interrupt = MagicMock()

            agent._on_utterance_detected("okay")

            assert agent._pending_utterance is None
            agent.handle_interrupt.assert_not_called()
        finally:
            loop.close()


