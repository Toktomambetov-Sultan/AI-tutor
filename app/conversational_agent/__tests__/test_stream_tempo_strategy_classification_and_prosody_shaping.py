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




class TestTempoStrategy:
    """Tests for ``core.tempo`` classification and shaping helpers."""

    def test_no_utterances_returns_normal(self):
        from core.tempo import classify_tempo, NORMAL

        assert classify_tempo([]) == NORMAL

    def test_slow_speaker_short_utterances(self):
        from core.tempo import classify_tempo, SLOW

        # Average < 4 words → SLOW
        short_utts = ["Um", "Yes", "OK", "Sure", "Yeah"]
        assert classify_tempo(short_utts) == SLOW

    def test_fast_speaker_long_utterances(self):
        from core.tempo import classify_tempo, FAST

        # Average > 12 words → FAST
        long_utts = [
            "Photosynthesis converts light energy into chemical energy that is stored in glucose molecules",
            "The chloroplasts in plant cells contain chlorophyll pigment which actively absorbs sunlight for the reaction",
            "Carbon dioxide and water are the primary raw materials consumed during photosynthesis in leaves",
        ]
        assert classify_tempo(long_utts) == FAST

    def test_normal_speaker_medium_utterances(self):
        from core.tempo import classify_tempo, NORMAL

        medium_utts = [
            "I understand the concept now",
            "Can you explain that in more detail",
            "That approach makes complete sense",
        ]
        assert classify_tempo(medium_utts) == NORMAL

    def test_wpm_based_slow(self):
        from core.tempo import classify_tempo, SLOW

        # ~60 WPM → SLOW (below default threshold 80)
        utts = ["hello world"]
        durations = [2.0]  # 2 words / 2s = 60 WPM
        assert classify_tempo(utts, durations) == SLOW

    def test_wpm_based_fast(self):
        from core.tempo import classify_tempo, FAST

        # ~200 WPM → FAST (above default threshold 160)
        utts = ["one two three four five six seven eight nine ten"]  # 10 words
        durations = [3.0]  # 10 words / 3s = 200 WPM
        assert classify_tempo(utts, durations) == FAST

    def test_adapt_disabled_always_normal(self):
        from unittest.mock import patch
        from dataclasses import replace
        import core.tempo as _tempo_mod
        import core.config as _cfg_mod
        from core.tempo import classify_tempo, NORMAL, SLOW

        original_cfg = _cfg_mod.RUNTIME_CONFIG
        disabled_tempo = replace(original_cfg.tempo, adapt_enabled=False)
        disabled_cfg = replace(original_cfg, tempo=disabled_tempo)

        with patch.object(_cfg_mod, "RUNTIME_CONFIG", disabled_cfg), patch.object(
            _tempo_mod, "RUNTIME_CONFIG", disabled_cfg
        ):
            # Even with slow-looking utterances, should return NORMAL
            assert classify_tempo(["Um", "Yes", "OK"]) == NORMAL

    def test_timing_inputs_are_safely_clamped(self):
        from core.tempo import classify_tempo, NORMAL

        # Invalid/edge timing values should not create extreme tempo hints.
        utts = ["short text", "medium text", "another medium text"]
        durations = [-5.0, 0.0, 9999.0]
        assert classify_tempo(utts, durations) == NORMAL

    def test_slow_shaping_adds_pause_after_conjunction(self):
        from core.tempo import apply_tempo_shaping, SLOW

        result = apply_tempo_shaping(
            "It converts light and stores it in glucose.", SLOW
        )
        assert "and," in result

    def test_fast_shaping_removes_ellipsis(self):
        from core.tempo import apply_tempo_shaping, FAST

        result = apply_tempo_shaping("Wait... let me think... OK.", FAST)
        assert "..." not in result
        assert "." in result

    def test_normal_shaping_is_identity(self):
        from core.tempo import apply_tempo_shaping, NORMAL

        text = "This is a normal sentence."
        assert apply_tempo_shaping(text, NORMAL) == text

    def test_tempo_hint_emitted_in_stream_and_speak(self):
        """_stream_and_speak must emit a TEMPO_HINT message before audio."""
        import asyncio
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources
        from core.protocol import MessageType
        import numpy as np

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        mock_tts.generate_audio = MagicMock(
            return_value=np.zeros(1000, dtype=np.float32)
        )
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

        try:
            # Minimal stream returning one sentence
            tokens = ["Good."]
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter([_make_chunk(t) for t in tokens])
            mock_openai.chat.completions.create.return_value = mock_stream

            t = threading.Thread(target=agent._stream_llm_and_speak, daemon=True)
            t.start()
            t.join(timeout=10)

            # Drain the queue
            messages = []

            async def _drain():
                while not q.empty():
                    messages.append(await q.get())

            loop.run_until_complete(_drain())

            types = [m[0] for m in messages]
            assert (
                MessageType.TEMPO_HINT in types
            ), f"Expected TEMPO_HINT in queue, got: {types}"
            # TEMPO_HINT should appear before any audio
            hint_idx = types.index(MessageType.TEMPO_HINT)
            audio_indices = [i for i, t in enumerate(types) if t == MessageType.AUDIO]
            if audio_indices:
                assert hint_idx < audio_indices[0]
        finally:
            loop.close()

    def test_handle_turn_passes_timing_window_to_tempo_classifier(self):
        import asyncio
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

        try:
            agent._lesson_ready = False
            agent._stream_llm_and_speak = MagicMock(return_value="ok")

            with patch("core.conversation.classify_tempo", return_value="normal") as m:
                agent._handle_turn("This should be enough words for tempo context")
                assert m.call_count == 1
                assert len(m.call_args.args) == 2
        finally:
            loop.close()


# ── helper used by TestTempoStrategy.test_tempo_hint_emitted_in_stream_and_speak ─


def _make_chunk(token: str):
    c = MagicMock()
    c.choices = [MagicMock()]
    c.choices[0].delta = MagicMock()
    c.choices[0].delta.content = token
    return c


