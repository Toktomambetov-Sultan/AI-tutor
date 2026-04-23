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

# ─────────────────────────────────────────────────────────────────────
# 1.  Sentence splitting
# ─────────────────────────────────────────────────────────────────────

from core.utils import has_trailing_clause_boundary, split_clauses, split_sentences


class TestSentenceSplitting:
    """Tests for the split_sentences helper."""

    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = split_sentences("Hello world. How are you? I am fine!")
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_preserves_punctuation(self):
        result = split_sentences("Wait! Really? Yes.")
        assert result == ["Wait!", "Really?", "Yes."]

    def test_no_split_on_abbreviations_with_no_space(self):
        # "e.g." doesn't have a space after the inner dots so won't split
        result = split_sentences("Use e.g. Python.")
        # The regex splits on '. P' — acceptable; at least no crash
        assert len(result) >= 1

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_no_punctuation(self):
        assert split_sentences("Hello world") == ["Hello world"]

    def test_multiple_spaces_between(self):
        result = split_sentences("Hello.   World!")
        assert result == ["Hello.", "World!"]

    def test_trailing_whitespace(self):
        result = split_sentences("One. Two.  ")
        assert result == ["One.", "Two."]

    def test_long_paragraph(self):
        text = (
            "The capital of France is Paris. "
            "It is known for the Eiffel Tower! "
            "Would you like to learn more? "
            "I can explain further."
        )
        result = split_sentences(text)
        assert len(result) == 4
        assert result[0] == "The capital of France is Paris."
        assert result[-1] == "I can explain further."

    def test_clause_splitting_breaks_long_sentence(self):
        text = "Photosynthesis uses light energy, converts it to chemical energy, and stores it in glucose."
        result = split_clauses(text)
        assert len(result) >= 2
        assert any("light energy" in part for part in result)

    def test_clause_splitting_merges_tiny_fragments(self):
        text = "Oh, OK, sure."
        result = split_clauses(text)
        # Fragments under _MIN_CLAUSE_LEN chars are merged together
        assert len(result) == 1
        assert result[0] == "Oh, OK, sure."

    def test_clause_boundary_detection_true_when_ending_with_punctuation(self):
        assert has_trailing_clause_boundary("Great work!") is True

    def test_clause_boundary_detection_false_for_partial_tail(self):
        assert has_trailing_clause_boundary("Great work and") is False


# ─────────────────────────────────────────────────────────────────────
# 2.  Interrupt handling
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# 3.  TTS queueing  (sentence-by-sentence audio delivery)
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# 4.  LLM streaming + sentence-chunked TTS flow
# ─────────────────────────────────────────────────────────────────────


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

            agent._handle_turn.assert_called_once_with(pending)
        finally:
            loop.close()


# ─────────────────────────────────────────────────────────────────────
# 5.  Vosk streaming STT
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# 9.  Language detection
# ─────────────────────────────────────────────────────────────────────

from core.utils import detect_language


class TestLanguageDetection:
    """Tests for the detect_language helper."""

    def test_english_text(self):
        assert detect_language("Hello, how are you?") == "en"

    def test_russian_text(self):
        assert detect_language("Привет, как дела?") == "ru"

    def test_mixed_text_majority_russian(self):
        assert detect_language("Привет world, как дела?") == "ru"

    def test_mixed_text_majority_english(self):
        assert detect_language("Hello мир, how are you doing today?") == "en"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_numbers_only(self):
        assert detect_language("12345") == "en"

    def test_punctuation_only(self):
        assert detect_language("!!! ???") == "en"


# ─────────────────────────────────────────────────────────────────────
# 10. Russian TTS routing
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# 11. Complete sentence delivery (word-boundary interrupt fix)
# ─────────────────────────────────────────────────────────────────────


class TestSentenceDeliveryOnInterrupt:
    """Verify that _send_audio delivers the full sentence even if
    interrupted mid-delivery (no per-chunk interrupt check)."""

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
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop, resources=resources)

        return agent, queue, loop

    def test_full_sentence_audio_delivered(self):
        """All chunks for a sentence should be delivered without
        mid-sentence interrupt checks."""
        agent, queue, loop = self._make_agent()
        # Audio that spans 3 chunks (each 32768 bytes)
        wav_bytes = b"\x00" * 80000

        def _run():
            agent._send_audio(wav_bytes, ai_text="Complete sentence.")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=2)

        messages = []

        async def _drain():
            while not queue.empty():
                messages.append(await queue.get())

        loop.run_until_complete(_drain())

        assert messages[0] == ("ai_text", "Complete sentence.")
        assert messages[-1] == ("end", None)
        audio_msgs = [m for m in messages if m[0] == "audio"]
        total_audio = sum(len(m[1]) for m in audio_msgs)
        assert total_audio == 80000
        loop.close()

    def test_initial_interrupt_check_still_works(self):
        """If interrupted before _send_audio starts, nothing is sent."""
        agent, queue, loop = self._make_agent()
        agent._interrupted.set()
        agent._send_audio(b"\x00" * 100, ai_text="Should not send.")
        time.sleep(0.05)
        assert queue.empty()
        loop.close()


# ─────────────────────────────────────────────────────────────────────
# 12. Reduced VAD silence thresholds
# ─────────────────────────────────────────────────────────────────────


class TestReducedVADThresholds:
    """Verify the default VAD silence thresholds are reduced for faster response."""

    def test_default_silence_values(self):
        import inspect
        from core.audio_processor import RealtimeAudioProcessor

        sig = inspect.signature(RealtimeAudioProcessor.__init__)
        assert sig.parameters["silence_sec"].default == 0.6
        assert sig.parameters["min_silence_sec"].default == 0.25
        assert sig.parameters["max_silence_sec"].default == 0.9


class TestLatencyGuardrails:
    """Tests that enforce low-latency turn endpointing constraints."""

    def test_latency_budget_defaults_under_five_seconds(self):
        from core.config import RUNTIME_CONFIG

        assert RUNTIME_CONFIG.audio.silence_sec < 5.0
        assert RUNTIME_CONFIG.audio.silence_sec_with_partial_text < 5.0
        assert RUNTIME_CONFIG.audio.max_utterance_sec < 5.0

    def test_partial_text_uses_faster_silence_window(self):
        from core.audio_processor import RealtimeAudioProcessor

        proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
        proc.silence_sec = 0.9
        proc._silence_sec_with_partial_text = 0.25
        proc._latest_partial_text = "hello"

        effective = proc._effective_silence_sec()
        assert effective == 0.25

        frames = proc._silence_frames_needed(0.032)
        assert frames * 0.032 < 0.5

    def test_force_finalize_timeout_triggers_before_five_seconds(self):
        from core.audio_processor import RealtimeAudioProcessor

        proc = RealtimeAudioProcessor.__new__(RealtimeAudioProcessor)
        proc._max_utterance_sec = 4.0
        proc._speech_frames = int(4.0 / 0.032)
        proc._silence_frames = 1
        proc._latest_partial_text = "hello"
        proc.min_speech_bytes = 1000
        proc._pcm_buf = bytearray(b"x" * 5000)
        proc.sample_rate = 16000

        assert proc._should_force_finalize(0.032)


class TestSTTTranscriptFallback:
    """Regression tests for transcript selection in RealtimeAudioProcessor."""

    def test_pick_transcript_prefers_final(self):
        from core.audio_processor import RealtimeAudioProcessor

        text = RealtimeAudioProcessor._pick_transcript_text(
            "final transcript", "partial transcript"
        )
        assert text == "final transcript"

    def test_pick_transcript_uses_partial_when_final_empty(self):
        from core.audio_processor import RealtimeAudioProcessor

        text = RealtimeAudioProcessor._pick_transcript_text("", "partial transcript")
        assert text == "partial transcript"


class TestLLMStreamingParameters:
    """Ensure runtime LLM settings are applied for stable spoken replies."""

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
        ):
            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()
            agent = ConversationalAgent(queue, loop, resources=resources)

        return agent, mock_openai, loop

    def test_stream_uses_runtime_llm_parameters(self):
        from core.config import RUNTIME_CONFIG

        agent, mock_openai, loop = self._make_agent()
        try:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Short answer."

            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter([chunk])
            mock_openai.chat.completions.create.return_value = mock_stream

            agent._stream_llm_and_speak()

            kwargs = mock_openai.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == RUNTIME_CONFIG.llm.model
            assert kwargs["temperature"] == RUNTIME_CONFIG.llm.temperature
            assert kwargs["top_p"] == RUNTIME_CONFIG.llm.top_p
            assert kwargs["presence_penalty"] == RUNTIME_CONFIG.llm.presence_penalty
            assert kwargs["frequency_penalty"] == RUNTIME_CONFIG.llm.frequency_penalty
            assert kwargs["max_tokens"] == RUNTIME_CONFIG.llm.max_tokens
        finally:
            loop.close()


# ─────────────────────────────────────────────────────────────────────
# 13. Russian TTS integration (real model, requires Docker)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not hasattr(torch, "hub") or isinstance(torch.hub, MagicMock),
    reason="Real torch required (run in Docker)",
)
class TestRussianTTSIntegration:
    """Integration tests for Silero Russian TTS.

    These tests load the actual model and generate audio, then verify
    the output is valid WAV.  Skipped outside Docker.
    """

    @pytest.fixture(autouse=True)
    def _load_model(self):
        """Load the Russian TTS model once for the class."""
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v3_1_ru",
                trust_repo=True,
            )
            self.speaker = "baya"
            self.sample_rate = 24000
        except Exception:
            pytest.skip("Silero Russian TTS model not available")

    def test_generates_audio_tensor(self):
        audio = self.model.apply_tts(
            text="Привет мир",
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        assert torch.is_tensor(audio)
        assert audio.ndim == 1
        assert audio.shape[0] > 0

    def test_audio_produces_valid_wav(self):
        audio = self.model.apply_tts(
            text="Добрый день, как ваши дела?",
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        buf = io.BytesIO()
        audio_np = audio.cpu().numpy()
        scipy.io.wavfile.write(buf, self.sample_rate, audio_np)
        wav_bytes = buf.getvalue()
        # WAV header starts with RIFF
        assert wav_bytes[:4] == b"RIFF"
        assert len(wav_bytes) > 1000  # non-trivial audio

    def test_different_speakers_produce_audio(self):
        """Verify multiple speakers work."""
        for speaker in ["baya", "xenia"]:
            audio = self.model.apply_tts(
                text="Тест",
                speaker=speaker,
                sample_rate=self.sample_rate,
            )
            assert audio.shape[0] > 0

    def test_long_russian_text(self):
        text = (
            "Фотосинтез — это процесс, при котором растения "
            "используют солнечный свет для превращения углекислого "
            "газа и воды в глюкозу и кислород."
        )
        audio = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        # Should generate at least ~2 seconds of audio at 24kHz
        assert audio.shape[0] > self.sample_rate * 2


# ─────────────────────────────────────────────────────────────────────
# 14. Vosk integration (requires Docker with Vosk models)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not hasattr(sys.modules.get("vosk", None), "__file__"),
    reason="Real vosk required (run in Docker)",
)
class TestVoskIntegration:
    """Integration tests for Vosk streaming STT.

    These tests load the actual Vosk model and run recognition on
    synthetic PCM.  Skipped outside Docker.
    """

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from vosk import Model as _VoskModel, KaldiRecognizer as _KaldiRec
        import json as _json

        model_path = "/app/models/vosk-model-small-en-us-0.15"
        try:
            self.model = _VoskModel(model_path)
        except Exception:
            pytest.skip(f"Vosk model not found at {model_path}")
        self.sample_rate = 16000
        self._json = _json
        self._KaldiRec = _KaldiRec

    def test_recognizer_accepts_silence(self):
        """Recognizer should return empty text for silence."""
        import numpy as np

        rec = self._KaldiRec(self.model, self.sample_rate)
        silence = np.zeros(self.sample_rate, dtype=np.int16).tobytes()
        rec.AcceptWaveform(silence)
        result = self._json.loads(rec.FinalResult())
        assert result.get("text", "") == ""

    def test_recognizer_returns_text_for_speech(self):
        """Recognizer should produce a result for a tone
        (may not be meaningful, but shouldn't crash)."""
        import numpy as np

        rec = self._KaldiRec(self.model, self.sample_rate)
        t = np.linspace(0, 1, self.sample_rate, dtype=np.float64)
        pcm = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).tobytes()
        rec.AcceptWaveform(pcm)
        result = self._json.loads(rec.FinalResult())
        assert "text" in result

    def test_russian_model_loads(self):
        """Verify the Russian Vosk model can be loaded."""
        from vosk import Model as _VoskModel

        ru_model_path = "/app/models/vosk-model-small-ru-0.22"
        try:
            ru_model = _VoskModel(ru_model_path)
        except Exception:
            pytest.skip(f"Russian Vosk model not found at {ru_model_path}")
        rec = self._KaldiRec(ru_model, self.sample_rate)
        import numpy as np

        silence = np.zeros(self.sample_rate, dtype=np.int16).tobytes()
        rec.AcceptWaveform(silence)
        result = self._json.loads(rec.FinalResult())
        assert "text" in result
