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




class TestStreamLLMAndSpeak:
    """Tests for _stream_llm_and_speak with mocked OpenAI streaming."""

    def _make_agent(self):
        import numpy as np
        from core.conversation import ConversationalAgent
        from core.resources import SharedResources

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

        return agent, q, loop, mock_openai

    def _make_stream_chunks(self, tokens):
        """Build a list of mock streaming chunk objects."""
        chunks = []
        for tok in tokens:
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta = MagicMock()
            c.choices[0].delta.content = tok
            chunks.append(c)
        return chunks

    def test_stream_produces_sentences(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            tokens = [
                "Photosynthesis happens in chloroplasts.",
                " It converts light energy into glucose?",
            ]
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(self._make_stream_chunks(tokens))
            mock_openai.chat.completions.create.return_value = mock_stream

            # Run in a thread since it calls loop.call_soon_threadsafe
            result = [None]
            exc = [None]

            def _run():
                try:
                    result[0] = agent._stream_llm_and_speak()
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=10)

            if exc[0]:
                raise exc[0]
            assert result[0] is not None
            assert "Photosynthesis happens in chloroplasts." in result[0]
            assert "It converts light energy into glucose?" in result[0]
            assert len(agent._spoken_sentences) == 2
            mock_stream.close.assert_called_once()
        finally:
            loop.close()

    def test_interrupt_stops_after_current_sentence(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            # Three sentences worth of tokens
            tokens = [
                "First sentence is definitely long enough.",
                " ",
                "Second sentence is also long enough.",
                " ",
                "Third sentence is also long enough.",
            ]
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(self._make_stream_chunks(tokens))
            mock_openai.chat.completions.create.return_value = mock_stream

            # Inject interrupt during the first TTS call — this simulates the
            # student starting to speak while the first sentence is being synthesised.
            import numpy as np
            from core.conversation import ConversationalAgent

            call_count = [0]
            orig_synth = agent._synthesise_sentence

            def _synth_with_interrupt(sentence):
                call_count[0] += 1
                if call_count[0] == 1:
                    agent.handle_interrupt()  # trigger barge-in during first sentence
                return np.zeros(1000, dtype=np.float32)

            agent._synthesise_sentence = _synth_with_interrupt

            result = [None]
            exc = [None]

            def _run():
                try:
                    result[0] = agent._stream_llm_and_speak()
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=10)

            if exc[0]:
                raise exc[0]
            assert result[0] is not None
            assert (
                len(agent._spoken_sentences) == 1
            ), f"Expected 1 spoken sentence, got {agent._spoken_sentences}"
            assert result[0] == "First sentence is definitely long enough."
            mock_stream.close.assert_called_once()
        finally:
            loop.close()

    def test_interrupted_reply_only_keeps_spoken_text_in_context(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            agent._lesson_ready = False
            agent._stream_llm_and_speak = MagicMock(
                return_value="Actually spoken part."
            )
            agent._spoken_sentences = ["Actually spoken part."]
            agent._current_ai_text = "Actually spoken part. Unspoken remainder."
            agent._interrupted.set()

            agent._handle_turn("What did you say?")

            assert agent.messages[-1] == {
                "role": "assistant",
                "content": "Actually spoken part.",
            }
            system_msgs = [
                msg["content"]
                for msg in agent.messages
                if msg["role"] == "system" and "interrupted" in msg["content"].lower()
            ]
            assert system_msgs
            assert "Actually spoken part." in system_msgs[-1]
            assert "Unspoken remainder." in system_msgs[-1]
        finally:
            loop.close()

    def test_pending_utterance_is_processed_after_interrupt(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            pending = "pending text from user"
            agent._processing = True
            agent._handle_turn = MagicMock()

            agent._on_utterance_detected(pending)

            assert agent._pending_utterance == pending
            assert agent._interrupted.is_set()

            with agent._processing_lock:
                agent._processing = False
            agent._process_pending_utterance()
            time.sleep(0.05)

            agent._handle_turn.assert_called_once_with(pending, elapsed_sec=None)
        finally:
            loop.close()


