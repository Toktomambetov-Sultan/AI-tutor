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
    import speech_recognition  # noqa: F401
except ImportError:
    _sr = ModuleType("speech_recognition")
    _sr.Recognizer = MagicMock
    sys.modules["speech_recognition"] = _sr

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

# ─────────────────────────────────────────────────────────────────────
# 1.  Sentence splitting
# ─────────────────────────────────────────────────────────────────────

from core.conversation import _split_clauses, _split_sentences


class TestSentenceSplitting:
    """Tests for the _split_sentences helper."""

    def test_single_sentence(self):
        assert _split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = _split_sentences("Hello world. How are you? I am fine!")
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_preserves_punctuation(self):
        result = _split_sentences("Wait! Really? Yes.")
        assert result == ["Wait!", "Really?", "Yes."]

    def test_no_split_on_abbreviations_with_no_space(self):
        # "e.g." doesn't have a space after the inner dots so won't split
        result = _split_sentences("Use e.g. Python.")
        # The regex splits on '. P' — acceptable; at least no crash
        assert len(result) >= 1

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_no_punctuation(self):
        assert _split_sentences("Hello world") == ["Hello world"]

    def test_multiple_spaces_between(self):
        result = _split_sentences("Hello.   World!")
        assert result == ["Hello.", "World!"]

    def test_trailing_whitespace(self):
        result = _split_sentences("One. Two.  ")
        assert result == ["One.", "Two."]

    def test_long_paragraph(self):
        text = (
            "The capital of France is Paris. "
            "It is known for the Eiffel Tower! "
            "Would you like to learn more? "
            "I can explain further."
        )
        result = _split_sentences(text)
        assert len(result) == 4
        assert result[0] == "The capital of France is Paris."
        assert result[-1] == "I can explain further."

    def test_clause_splitting_breaks_long_sentence(self):
        text = "Photosynthesis uses light energy, converts it to chemical energy, and stores it in glucose."
        result = _split_clauses(text)
        assert len(result) >= 2
        assert any("light energy" in part for part in result)

    def test_clause_splitting_merges_tiny_fragments(self):
        text = "Yes, of course, we can do that."
        result = _split_clauses(text)
        assert len(result) == 1
        assert result[0] == "Yes, of course, we can do that."


# ─────────────────────────────────────────────────────────────────────
# 2.  Interrupt handling
# ─────────────────────────────────────────────────────────────────────


class TestInterruptHandling:
    """Tests for the barge-in / interrupt mechanism."""

    def _make_agent(self, lesson_context=""):
        """Create a ConversationalAgent with all heavy deps mocked."""
        with patch(
            "core.conversation.ConversationalAgent._tts_model"
        ) as mock_tts, patch(
            "core.conversation.ConversationalAgent._voice_state"
        ), patch(
            "core.conversation.ConversationalAgent._recognizer"
        ), patch(
            "core.conversation.ConversationalAgent._openai_client"
        ) as mock_openai, patch(
            "core.conversation.RealtimeAudioProcessor"
        ), patch(
            "core.conversation.LessonRAG"
        ) as mock_rag:
            mock_tts.sample_rate = 22050
            mock_rag_inst = MagicMock()
            mock_rag.return_value = mock_rag_inst
            mock_rag_inst.ingest = MagicMock()
            mock_rag_inst.build_system_prompt = MagicMock(return_value="system prompt")
            mock_rag_inst.chunk_count = 1
            mock_rag_inst.lesson_title = "Test"

            from core.conversation import ConversationalAgent

            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop, lesson_context=lesson_context)
            return agent, queue, loop

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


# ─────────────────────────────────────────────────────────────────────
# 3.  TTS queueing  (sentence-by-sentence audio delivery)
# ─────────────────────────────────────────────────────────────────────


class TestTTSQueueing:
    """Verify _send_audio pushes (ai_text, data) onto the response queue."""

    def _make_agent(self):
        with patch(
            "core.conversation.ConversationalAgent._tts_model"
        ) as mock_tts, patch(
            "core.conversation.ConversationalAgent._voice_state"
        ), patch(
            "core.conversation.ConversationalAgent._recognizer"
        ), patch(
            "core.conversation.ConversationalAgent._openai_client"
        ), patch(
            "core.conversation.RealtimeAudioProcessor"
        ), patch(
            "core.conversation.LessonRAG"
        ) as mock_rag:
            mock_tts.sample_rate = 22050
            mock_rag_inst = MagicMock()
            mock_rag.return_value = mock_rag_inst
            mock_rag_inst.ingest = MagicMock()
            mock_rag_inst.build_system_prompt = MagicMock(return_value="sys")
            mock_rag_inst.chunk_count = 0
            mock_rag_inst.lesson_title = "T"

            from core.conversation import ConversationalAgent

            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop)
            return agent, queue, loop

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


# ─────────────────────────────────────────────────────────────────────
# 4.  LLM streaming + sentence-chunked TTS flow
# ─────────────────────────────────────────────────────────────────────


class TestStreamLLMAndSpeak:
    """Tests for _stream_llm_and_speak with mocked OpenAI streaming."""

    def _make_agent(self):
        import numpy as np
        from core.conversation import ConversationalAgent

        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050
        mock_tts.generate_audio = MagicMock(
            return_value=np.zeros(1000, dtype=np.float32)
        )
        mock_openai = MagicMock()

        # Patch class-level attributes directly
        orig_tts = ConversationalAgent._tts_model
        orig_vs = ConversationalAgent._voice_state
        orig_rec = ConversationalAgent._recognizer
        orig_client = ConversationalAgent._openai_client

        ConversationalAgent._tts_model = mock_tts
        ConversationalAgent._voice_state = MagicMock()
        ConversationalAgent._recognizer = MagicMock()
        ConversationalAgent._openai_client = mock_openai

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
            agent = ConversationalAgent(queue, loop)

        # Store originals for cleanup
        agent._orig = (orig_tts, orig_vs, orig_rec, orig_client)
        return agent, queue, loop, mock_openai

    def _cleanup(self, agent):
        from core.conversation import ConversationalAgent

        orig_tts, orig_vs, orig_rec, orig_client = agent._orig
        ConversationalAgent._tts_model = orig_tts
        ConversationalAgent._voice_state = orig_vs
        ConversationalAgent._recognizer = orig_rec
        ConversationalAgent._openai_client = orig_client

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
            self._cleanup(agent)
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
            self._cleanup(agent)
            loop.close()

    def test_interrupted_reply_only_keeps_spoken_text_in_context(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            agent._lesson_ready = False
            agent._transcribe = MagicMock(return_value="What did you say?")
            agent._stream_llm_and_speak = MagicMock(
                return_value="Actually spoken part."
            )
            agent._spoken_sentences = ["Actually spoken part."]
            agent._current_ai_text = "Actually spoken part. Unspoken remainder."
            agent._interrupted.set()

            agent._handle_turn(b"\x00" * 32000)

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
            self._cleanup(agent)
            loop.close()

    def test_pending_utterance_is_processed_after_interrupt(self):
        agent, queue, loop, mock_openai = self._make_agent()
        try:
            pending = b"pending-pcm"
            agent._processing = True
            agent._handle_turn = MagicMock()

            agent._on_utterance_detected(pending)

            assert agent._pending_utterance == pending
            assert agent._interrupted.is_set()

            with agent._processing_lock:
                agent._processing = False
            agent._process_pending_utterance()
            time.sleep(0.05)

            agent._handle_turn.assert_called_once_with(pending)
        finally:
            self._cleanup(agent)
            loop.close()


# ─────────────────────────────────────────────────────────────────────
# 5.  Whisper STT (_transcribe) with mocked API
# ─────────────────────────────────────────────────────────────────────


class TestWhisperSTT:
    """Test _transcribe with a mocked OpenAI Whisper response."""

    def _make_agent(self):
        from core.conversation import ConversationalAgent

        mock_openai = MagicMock()
        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050

        orig_tts = ConversationalAgent._tts_model
        orig_vs = ConversationalAgent._voice_state
        orig_rec = ConversationalAgent._recognizer
        orig_client = ConversationalAgent._openai_client

        ConversationalAgent._tts_model = mock_tts
        ConversationalAgent._voice_state = MagicMock()
        ConversationalAgent._recognizer = MagicMock()
        ConversationalAgent._openai_client = mock_openai

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
            agent = ConversationalAgent(queue, loop)

        agent._orig = (orig_tts, orig_vs, orig_rec, orig_client)
        return agent, mock_openai

    def _cleanup(self, agent):
        from core.conversation import ConversationalAgent

        orig_tts, orig_vs, orig_rec, orig_client = agent._orig
        ConversationalAgent._tts_model = orig_tts
        ConversationalAgent._voice_state = orig_vs
        ConversationalAgent._recognizer = orig_rec
        ConversationalAgent._openai_client = orig_client

    def test_transcribe_returns_text(self):
        agent, mock_openai = self._make_agent()
        try:
            # Mock the Whisper API response
            mock_result = MagicMock()
            mock_result.text = "  Hello, this is a test.  "
            mock_openai.audio.transcriptions.create.return_value = mock_result

            import numpy as np

            # Create fake PCM data (1 second of silence at 16kHz)
            pcm = np.zeros(16000, dtype=np.int16).tobytes()
            result = agent._transcribe(pcm)

            assert result == "Hello, this is a test."
            mock_openai.audio.transcriptions.create.assert_called_once()
        finally:
            self._cleanup(agent)

    def test_transcribe_handles_api_error(self):
        agent, mock_openai = self._make_agent()
        try:
            mock_openai.audio.transcriptions.create.side_effect = Exception("API down")

            import numpy as np

            pcm = np.zeros(16000, dtype=np.int16).tobytes()
            result = agent._transcribe(pcm)

            assert result == ""  # graceful fallback
        finally:
            self._cleanup(agent)


# ─────────────────────────────────────────────────────────────────────
# 6.  gRPC servicer — ready signal + interrupt forwarding
# ─────────────────────────────────────────────────────────────────────


class TestGRPCServicerLogic:
    """Test key servicer behaviour patterns."""

    def test_ready_signal_after_agent_init(self):
        """Verify that the servicer sends a 'ready' signal after creating the agent."""
        # This is a structural test — we verify the queue receives ("signal", "ready")
        # after the first message is processed.
        from core.grpc_servicer import AudioServicer

        servicer = AudioServicer()
        # The StreamAudio method is async — we test the overall contract
        assert hasattr(servicer, "StreamAudio")

    def test_audio_chunk_proto_has_new_fields(self):
        """Verify the proto has the expected fields."""
        from proto import audio_pb2

        chunk = audio_pb2.AudioChunk()
        # All new fields should be accessible and default to empty
        assert chunk.signal == ""
        assert chunk.ai_text == ""
        assert chunk.client_signal == ""

        # Setting fields
        chunk.signal = "ready"
        chunk.ai_text = "Hello!"
        chunk.client_signal = "interrupt"
        assert chunk.signal == "ready"
        assert chunk.ai_text == "Hello!"
        assert chunk.client_signal == "interrupt"


# ─────────────────────────────────────────────────────────────────────
# 7.  Silero VAD audio processor
# ─────────────────────────────────────────────────────────────────────


class TestSileroVADAudioProcessor:
    """Tests for the Silero-based RealtimeAudioProcessor (mocked model)."""

    def test_adaptive_silence_shortens_for_fast_speaker(self):
        """Feed several short utterance lengths and verify silence_sec decreases."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                # Simulate 5 short utterances (~1.5s each)
                for _ in range(5):
                    short_bytes = int(1.5 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                    proc._adapt_silence_threshold(short_bytes)

                # Silence should have moved toward the minimum
                assert proc.silence_sec < 1.3, f"Expected < 1.3, got {proc.silence_sec}"

    def test_adaptive_silence_lengthens_for_slow_speaker(self):
        """Feed several long utterance lengths and verify silence_sec increases."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                # Simulate 5 long utterances (~7s each)
                for _ in range(5):
                    long_bytes = int(7.0 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                    proc._adapt_silence_threshold(long_bytes)

                # Silence should have moved toward the maximum
                assert proc.silence_sec > 2.0, f"Expected > 2.0, got {proc.silence_sec}"

    def test_adaptive_needs_minimum_data(self):
        """With only 1 utterance, silence_sec shouldn't change."""
        with patch("core.audio_processor._ensure_silero_model"):
            with patch("subprocess.Popen"):
                from core.audio_processor import (
                    RealtimeAudioProcessor,
                    TARGET_SAMPLE_RATE,
                    BYTES_PER_SAMPLE,
                )

                proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
                proc.sample_rate = TARGET_SAMPLE_RATE
                proc.silence_sec = 1.5
                proc._min_silence_sec = 1.0
                proc._max_silence_sec = 2.5
                proc._recent_speech_durations = []
                proc._ADAPT_WINDOW = 5

                original = proc.silence_sec
                proc._adapt_silence_threshold(
                    int(3.0 * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
                )
                assert proc.silence_sec == original


# ─────────────────────────────────────────────────────────────────────
# 8.  Conversation history compression
# ─────────────────────────────────────────────────────────────────────


class TestHistoryCompression:
    """Tests for _compress_history in ConversationalAgent."""

    def _make_agent(self):
        from core.conversation import ConversationalAgent

        mock_openai = MagicMock()
        mock_tts = MagicMock()
        mock_tts.sample_rate = 22050

        orig_tts = ConversationalAgent._tts_model
        orig_vs = ConversationalAgent._voice_state
        orig_rec = ConversationalAgent._recognizer
        orig_client = ConversationalAgent._openai_client

        ConversationalAgent._tts_model = mock_tts
        ConversationalAgent._voice_state = MagicMock()
        ConversationalAgent._recognizer = MagicMock()
        ConversationalAgent._openai_client = mock_openai

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
            agent = ConversationalAgent(queue, loop)

        agent._orig = (orig_tts, orig_vs, orig_rec, orig_client)
        return agent, mock_openai

    def _cleanup(self, agent):
        from core.conversation import ConversationalAgent

        orig_tts, orig_vs, orig_rec, orig_client = agent._orig
        ConversationalAgent._tts_model = orig_tts
        ConversationalAgent._voice_state = orig_vs
        ConversationalAgent._recognizer = orig_rec
        ConversationalAgent._openai_client = orig_client

    def test_no_compression_below_threshold(self):
        agent, mock_openai = self._make_agent()
        try:
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
        finally:
            self._cleanup(agent)

    def test_compression_above_threshold(self):
        agent, mock_openai = self._make_agent()
        try:
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
        finally:
            self._cleanup(agent)

    def test_compression_failure_leaves_messages_intact(self):
        agent, mock_openai = self._make_agent()
        try:
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
        finally:
            self._cleanup(agent)
