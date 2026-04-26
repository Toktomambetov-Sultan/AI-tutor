"""
Per-session conversational agent.

Thin orchestrator that coordinates:
- **LessonManager** — RAG, language detection, STT model selection
- **SessionState** — interrupts, turn queuing, message history
- **LLMPipeline** — LLM streaming, sentence chunking, TTS synthesis

Key design:
  - **Vosk streaming STT** — audio is transcribed in real-time as the
    student speaks; by the time the utterance boundary fires the text
    is already available (zero STT latency).
  - **LLM streaming** with sentence-boundary chunking
  - **Per-sentence TTS** so the first audio reaches the student sooner
  - **Barge-in support** — student can interrupt mid-sentence; the agent
    knows where in the response text it was stopped and resumes cleanly
"""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
from typing import Callable

import numpy as np
import scipy.io.wavfile
import torch
from openai import OpenAI
from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel

from core.audio_processor import TARGET_SAMPLE_RATE, RealtimeAudioProcessor
from core.config import RUNTIME_CONFIG
from core.lesson_manager import LessonManager
from core.pipeline import LLMPipeline
from core.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    INTERRUPT_CONTEXT_TEMPLATE,
    LESSON_END_TOKEN,
    OPENING_GREETING_INSTRUCTION,
    SILENCE_CONTEXT_TEMPLATE,
    STT_CLARIFY_FALLBACK,
    VOICE_GENDER_SYSTEM_MSG,
    VOICE_STYLE_GUARDRAIL,
)
from core.protocol import MessageType, QueueMessage
from core.rag import LessonRAG
from core.resources import SharedResources, build_default_shared_resources
from core.session_state import SessionState
from core.tempo import NORMAL, apply_tempo_shaping, classify_tempo
from core.turn_policy import TimingAwareTurnGate, calculate_proactive_delay
from core.utils import (
    detect_language,
    extract_text_from_lesson_context,
    has_trailing_clause_boundary,
    is_lesson_end_request,
    is_low_confidence_transcript,
    split_clauses,
    is_filler_utterance,
)

logger = logging.getLogger(__name__)

_POSITIVE_TTS_HINT_RE = re.compile(
    r"\b(great|excellent|well done|nice work|good job|awesome|perfect)\b",
    re.IGNORECASE,
)

# Suppress Vosk's own logging (we log results ourselves)
SetLogLevel(-1)


class ConversationalAgent:
    """Per-session agent.  Receives pre-loaded shared resources to avoid
    re-loading the TTS model on every call."""

    # ── Process-wide shared resources (set once at startup) ──
    _default_resources: SharedResources | None = None

    @classmethod
    def apply_shared_resources(cls, resources: SharedResources) -> None:
        """Store process-wide shared resources for use by all new sessions."""
        cls._default_resources = resources

    @classmethod
    def load_shared_resources(
        cls,
        resource_loader: Callable[[], SharedResources] | None = None,
    ):
        """Call once at server startup to pre-load heavy models.

        A custom *resource_loader* enables dependency injection for tests
        and alternative runtime bootstraps.
        """
        loader = resource_loader or build_default_shared_resources
        cls.apply_shared_resources(loader())
        logger.info("All shared resources loaded.")

    # ────────────────────────────────────────────────────────────
    # Instance
    # ────────────────────────────────────────────────────────────

    def __init__(
        self,
        response_queue,
        loop,
        lesson_context: str = "",
        resources: SharedResources | None = None,
        rag_factory: Callable[[OpenAI], LessonRAG] | None = None,
        recognizer_factory: Callable[[VoskModel, int], object] | None = None,
        audio_processor_factory: Callable[..., RealtimeAudioProcessor] | None = None,
    ):
        self.response_queue = response_queue
        self.loop = loop
        self._resources = resources or self.__class__._default_resources
        if self._resources is None:
            raise RuntimeError(
                "SharedResources are not configured. Call load_shared_resources() "
                "or inject resources."
            )

        _rag_factory = rag_factory or LessonRAG
        _recognizer_factory = recognizer_factory or KaldiRecognizer
        _audio_processor_factory = audio_processor_factory or RealtimeAudioProcessor

        self._init_rag(_rag_factory, lesson_context)
        self._init_session_state()
        self._init_processor(_recognizer_factory, _audio_processor_factory)

    def _init_rag(
        self,
        rag_factory: Callable[[OpenAI], LessonRAG],
        lesson_context: str,
    ):
        """Set up RAG, lesson context, and language detection."""
        self.rag = rag_factory(self._resources.openai_client)
        self._lesson_ready = False
        self._language = "en"

        if lesson_context:
            self._init_lesson(lesson_context)
        else:
            self.context = FALLBACK_SYSTEM_PROMPT

        self.messages: list[dict] = [
            {"role": "system", "content": self.context},
            {"role": "system", "content": VOICE_STYLE_GUARDRAIL},
            {
                "role": "system",
                "content": VOICE_GENDER_SYSTEM_MSG.format(
                    gender=RUNTIME_CONFIG.llm.voice_gender
                ),
            },
        ]

    def _init_session_state(self):
        """Initialise per-session mutable state (interrupts, turn tracking)."""
        self._interrupted = threading.Event()
        self._current_ai_text: str = ""
        self._spoken_sentences: list[str] = []

        self._processing = False
        self._processing_lock = threading.Lock()
        self._pending_utterance: str | None = None
        self._first_turn = True
        self._session_finished = False
        self._silence_timer: threading.Timer | None = None
        self._silence_timer_lock = threading.Lock()

        # Turn gate and tempo state — fully initialised in _init_processor
        self._turn_gate: TimingAwareTurnGate | None = None
        self._student_utterances: list[str] = []
        self._student_turn_durations_sec: list[float] = []
        self._last_student_turn_monotonic: float | None = None
        self._current_tempo_hint: str = NORMAL

    def _init_processor(
        self,
        recognizer_factory: Callable[[VoskModel, int], object],
        audio_processor_factory: Callable[..., RealtimeAudioProcessor],
    ):
        """Pick the correct Vosk model for the session language and build the processor."""
        vosk_model = (
            self._resources.vosk_model_ru
            if self._language == "ru" and self._resources.vosk_model_ru
            else self._resources.vosk_model_en
        )
        recognizer = (
            recognizer_factory(vosk_model, TARGET_SAMPLE_RATE) if vosk_model else None
        )
        logger.info(
            "Session language=%s, Vosk recognizer=%s",
            self._language,
            "ready" if recognizer else "unavailable",
        )
        self._turn_gate = TimingAwareTurnGate(self._on_utterance_detected)
        self._processor = audio_processor_factory(
            on_utterance=self._turn_gate.feed,
            recognizer=recognizer,
            on_partial_utterance=self._on_partial_utterance_detected,
        )

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def process_audio_chunk(self, chunk_bytes: bytes):
        """Called by the gRPC servicer when new WebM/Opus data arrives."""
        if self._first_turn and self._lesson_ready:
            self._first_turn = False
            threading.Thread(target=self._send_opening_greeting, daemon=True).start()
        self._processor.feed(chunk_bytes)

    def handle_interrupt(self):
        """Called when the student starts speaking while AI is talking.

        Sets the interrupt flag, drains any queued audio so the client
        stops hearing stale speech, and sends an explicit ``interrupt``
        signal so the frontend can halt playback immediately.
        """
        self._interrupted.set()

        now = time.monotonic()
        if hasattr(self, "_spoken_schedule"):
            actually_spoken = []
            for item in self._spoken_schedule:
                if now >= item["end"]:
                    actually_spoken.append(item["clause"])
                elif now >= item["start"]:
                    # Partial sentence: approximate word count based on elapsed time vs total length
                    fraction = min(
                        1.0,
                        max(0.0, (now - item["start"]) / (item["end"] - item["start"])),
                    )
                    words = item["clause"].split()
                    num_words = int(len(words) * fraction)
                    if num_words > 0:
                        actually_spoken.append(" ".join(words[:num_words]) + "...")
                        logger.info(
                            "Interrupted mid-sentence. Trimmed to: %s",
                            actually_spoken[-1],
                        )
                    break
                else:
                    break

            if actually_spoken:
                self._spoken_sentences = actually_spoken
            else:
                self._spoken_sentences = []

        def _drain_and_signal():
            drained = 0
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                    drained += 1
                except Exception:
                    break
            self.response_queue.put_nowait(
                QueueMessage(MessageType.SIGNAL, "interrupt")
            )
            logger.info(
                "Interrupt flag set — drained %d queued messages, sent interrupt signal",
                drained,
            )

        self.loop.call_soon_threadsafe(_drain_and_signal)

    def close(self):
        """Clean up ffmpeg and RAG resources."""
        self._cancel_silence_timer()
        self._processor.close()
        self.rag.close()
        if self._turn_gate is not None:
            self._turn_gate.close()

    # ────────────────────────────────────────────────────────────
    # Conversation history compression
    # ────────────────────────────────────────────────────────────

    # When the message list exceeds this count, older turns are
    # summarised into a single system message to keep the context
    # window efficient while preserving continuity.
    _MAX_MESSAGES = 20
    _KEEP_RECENT = 6  # always keep the N most recent messages

    def _compress_history(self):
        """Summarise older conversation turns into one system message.

        Keeps the first system message (base prompt) and the last
        ``_KEEP_RECENT`` messages untouched.  Everything in between
        is fed to the LLM as a summarisation task and collapsed.
        """
        if len(self.messages) <= self._MAX_MESSAGES:
            return

        head = self.messages[:1]  # system prompt
        middle = self.messages[1 : -self._KEEP_RECENT]
        tail = self.messages[-self._KEEP_RECENT :]

        if not middle:
            return

        # Build the summary request
        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in middle
        )
        try:
            resp = self._resources.openai_client.chat.completions.create(
                model=RUNTIME_CONFIG.llm.summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise summariser. Summarise the following "
                            "tutoring conversation into 2-4 sentences, preserving "
                            "the key topics discussed and any important facts "
                            "the student mentioned.  Reply with ONLY the summary."
                        ),
                    },
                    {"role": "user", "content": convo_text},
                ],
            )
            summary = resp.choices[0].message.content.strip()
            logger.info(
                "History compressed: %d messages → summary (%d chars)",
                len(middle),
                len(summary),
            )
        except Exception as e:
            logger.warning("History compression failed: %s", e)
            return

        summary_msg = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY]\n{summary}",
        }
        self.messages = head + [summary_msg] + tail

    # ────────────────────────────────────────────────────────────
    # Lesson init
    # ────────────────────────────────────────────────────────────

    def _init_lesson(self, lesson_context: str):
        try:
            self.rag.ingest(lesson_context)
            self.context = self.rag.build_system_prompt()
            self._lesson_ready = True
            self._language = detect_language(
                extract_text_from_lesson_context(lesson_context)
            )
            logger.info(
                "Lesson RAG ready: %d chunks indexed for %r (language=%s)",
                self.rag.chunk_count,
                self.rag.lesson_title,
                self._language,
            )
        except Exception as e:
            logger.error("Failed to initialise lesson RAG: %s", e, exc_info=True)
            self.context = FALLBACK_SYSTEM_PROMPT

    # ────────────────────────────────────────────────────────────
    # TTS helper (unchanged engine, but per-sentence now)
    # ────────────────────────────────────────────────────────────

    def _get_tts_sample_rate(self, lang: str) -> int:
        """Return the sample rate for the TTS engine used for *lang*."""
        if lang == "ru" and self._resources.ru_tts_model is not None:
            return self._resources.ru_sample_rate
        return self._resources.tts_model.sample_rate

    def _make_silence_wav(self, duration_ms: int, lang: str = "en") -> bytes:
        """Return WAV bytes containing *duration_ms* of silence."""
        sample_rate = self._get_tts_sample_rate(lang)
        n_samples = int(sample_rate * duration_ms / 1000)
        silence = np.zeros(n_samples, dtype=np.float32)
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sample_rate, silence)
        return buf.getvalue()

    def _emotion_style_tts_text(self, sentence: str, lang: str) -> str:
        """Apply lightweight prosody hints for English speech output only."""
        if lang != "en":
            return sentence

        text = sentence.strip()
        if not text:
            return sentence

        # Apply tempo prosody shaping before emotion markup
        if RUNTIME_CONFIG.tempo.adapt_enabled:
            text = apply_tempo_shaping(text, self._current_tempo_hint)

        if not RUNTIME_CONFIG.tts.enable_emotion:
            return text

        strength = RUNTIME_CONFIG.tts.emotion_strength
        extra_exclamations = 1 + int(strength * 2)

        if _POSITIVE_TTS_HINT_RE.search(text) and text.endswith("."):
            return text[:-1] + ("!" * extra_exclamations)

        if text.endswith("!"):
            return re.sub(r"!+$", "!" * extra_exclamations, text)

        return text

    def _synthesise_sentence(self, sentence: str) -> bytes:
        """Return WAV bytes for a single sentence (English or Russian)."""
        lang = detect_language(sentence)
        if lang == "ru" and self._resources.ru_tts_model is not None:
            audio_tensor = self._resources.ru_tts_model.apply_tts(
                text=sentence,
                speaker=self._resources.ru_speaker,
                sample_rate=self._resources.ru_sample_rate,
            )
            sample_rate = self._resources.ru_sample_rate
        else:
            tts_text = self._emotion_style_tts_text(sentence, lang)
            audio_tensor = self._resources.tts_model.generate_audio(
                self._resources.voice_state, tts_text
            )
            sample_rate = self._resources.tts_model.sample_rate

        buf = io.BytesIO()
        audio_np = (
            audio_tensor.cpu().numpy()
            if torch.is_tensor(audio_tensor)
            else audio_tensor
        )
        scipy.io.wavfile.write(buf, sample_rate, audio_np)
        return buf.getvalue()

    def _send_audio(self, wav_bytes: bytes, ai_text: str = ""):
        """Queue WAV audio + metadata for the gRPC servicer to stream.

        Delivers the complete sentence audio to avoid cutting off
        mid-word.  The interrupt flag is checked only before starting
        a new sentence — the frontend handles smooth fade-out.
        """
        if self._interrupted.is_set():
            return

        # Send the text label first so the servicer can attach it
        if ai_text:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AI_TEXT, ai_text),
            )

        chunk_size = 32768
        for i in range(0, len(wav_bytes), chunk_size):
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AUDIO, wav_bytes[i : i + chunk_size]),
            )
        self.loop.call_soon_threadsafe(
            self.response_queue.put_nowait, QueueMessage(MessageType.END)
        )

    def _check_lesson_end(self) -> None:
        """Emit lesson_end signal to the transport queue."""
        self._session_finished = True
        self._cancel_silence_timer()
        self.loop.call_soon_threadsafe(
            self.response_queue.put_nowait,
            QueueMessage(MessageType.SIGNAL, "lesson_end"),
        )

    def _cancel_silence_timer(self) -> None:
        with self._silence_timer_lock:
            if self._silence_timer is not None:
                self._silence_timer.cancel()
                self._silence_timer = None

    def _reset_silence_timer(self, delay_sec: float | None = None) -> None:
        """Schedule proactive re-engagement when the student stays silent.
        If `delay_sec` is not provided, defaults to the base proactive silence timeout.
        """
        if self._session_finished:
            return

        if delay_sec is None:
            delay_sec = RUNTIME_CONFIG.turn_policy.proactive_silence_sec

        self._cancel_silence_timer()
        with self._silence_timer_lock:
            self._silence_timer = threading.Timer(
                delay_sec,
                lambda: self._on_silence_timeout(delay_sec),
            )
            self._silence_timer.daemon = True
            self._silence_timer.start()

    def _on_silence_timeout(self, elapsed_sec: float) -> None:
        """Trigger a proactive tutor follow-up when no utterance arrives."""
        with self._processing_lock:
            if self._session_finished:
                return
            if self._processing:
                # Busy now; try again later.
                self._reset_silence_timer()
                return
            self._processing = True

        threading.Thread(
            target=self._handle_silence_followup,
            args=(elapsed_sec,),
            daemon=True,
        ).start()

    def _handle_silence_followup(self, elapsed_sec: float) -> None:
        try:
            silence_note = SILENCE_CONTEXT_TEMPLATE.format(elapsed=elapsed_sec)
            spoken_reply = self._stream_llm_and_speak(extra_system_msg=silence_note)
            logger.info("AI proactive follow-up after silence: %s", spoken_reply[:200])
            self.messages.append({"role": "assistant", "content": spoken_reply})
        except Exception as e:
            logger.error(
                "Error during proactive silence follow-up: %s", e, exc_info=True
            )
        finally:
            with self._processing_lock:
                self._processing = False
            self._process_pending_utterance()

    def _speak_direct_text(self, text: str) -> str:
        """Speak a deterministic text response without a new LLM call."""
        self._interrupted.clear()
        self._spoken_sentences = []
        for clause in split_clauses(text):
            if self._interrupted.is_set():
                break
            wav = self._synthesise_sentence(clause)
            self._send_audio(wav, ai_text=clause)
            self._spoken_sentences.append(clause)
        spoken = " ".join(self._spoken_sentences) if self._spoken_sentences else text
        self._current_ai_text = spoken
        return spoken

    # ────────────────────────────────────────────────────────────
    # LLM streaming + per-sentence TTS
    # ────────────────────────────────────────────────────────────

    def _stream_and_speak(self, messages: list[dict]) -> str:
        """Stream the LLM for *messages*, split into TTS clauses, return spoken text.

        Uses a **producer / consumer** pattern: the main thread streams
        LLM tokens and pushes complete clauses into a queue; a worker
        thread synthesises and sends TTS audio in parallel.  This
        overlaps LLM generation with TTS, reducing perceived latency.

        Returns the *spoken* reply text (truncated at the interrupt
        point when the student barges in).
        """
        self._interrupted.clear()
        self._spoken_sentences = []
        self._current_ai_text = ""
        self._spoken_schedule = []
        self._audio_playhead = 0.0

        # Emit advisory tempo hint so clients can adjust playback speed.
        if RUNTIME_CONFIG.tempo.adapt_enabled:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.TEMPO_HINT, self._current_tempo_hint),
            )

        # ── TTS worker (consumer) ──
        sentence_q: queue.Queue[str | None] = queue.Queue()

        def _synthesise_and_send(clause: str):
            """Synthesise *clause* and queue the resulting audio."""
            logger.info("TTS clause: %s", clause)
            wav = self._synthesise_sentence(clause)

            # Track playhead for word-level interruption alignment
            now = time.monotonic()
            if self._audio_playhead < now:
                self._audio_playhead = now

            start_time = self._audio_playhead
            # Estimate duration based on TTS model rate, or roughly 2.5 words/sec fallback
            duration_sec = 0.0
            try:
                import io, wave

                with wave.open(io.BytesIO(wav), "rb") as w:
                    duration_sec = w.getnframes() / float(w.getframerate())
            except Exception:
                duration_sec = max(0.5, len(clause.split()) / 2.5)

            end_time = start_time + duration_sec
            self._audio_playhead = end_time

            self._spoken_schedule.append(
                {"start": start_time, "end": end_time, "clause": clause}
            )

            self._send_audio(wav, ai_text=clause)
            self._spoken_sentences.append(clause)

        def _tts_worker():
            """Consume clauses from *sentence_q* and synthesise + send."""
            pause_ms = RUNTIME_CONFIG.tempo.inter_sentence_pause_ms
            prev_clause: str | None = None
            while True:
                clause = sentence_q.get()
                if clause is None or self._interrupted.is_set():
                    break
                # Inject a natural pause between sentences (not before the first).
                if (
                    prev_clause is not None
                    and pause_ms > 0
                    and not self._interrupted.is_set()
                ):
                    silence_lang = detect_language(clause)
                    silence_wav = self._make_silence_wav(pause_ms, silence_lang)
                    now = time.monotonic()
                    if self._audio_playhead < now:
                        self._audio_playhead = now
                    self._audio_playhead += pause_ms / 1000.0
                    self._send_audio(silence_wav)
                _synthesise_and_send(clause)
                prev_clause = clause

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        # ── Stream LLM tokens (producer) ──
        stream = self._resources.openai_client.chat.completions.create(
            model=RUNTIME_CONFIG.llm.model,
            messages=messages,
            stream=True,
            temperature=RUNTIME_CONFIG.llm.temperature,
            top_p=RUNTIME_CONFIG.llm.top_p,
            presence_penalty=RUNTIME_CONFIG.llm.presence_penalty,
            frequency_penalty=RUNTIME_CONFIG.llm.frequency_penalty,
            max_tokens=RUNTIME_CONFIG.llm.max_tokens,
        )

        buffer = ""  # accumulates tokens until a clause boundary
        full_reply = ""  # everything the LLM produced (token-stripped)
        lesson_end_requested = False

        try:
            for chunk in stream:
                if self._interrupted.is_set():
                    break

                delta = chunk.choices[0].delta
                if not delta.content:
                    continue

                token = delta.content
                buffer += token
                full_reply += token

                # T08: lesson-end marker must never be spoken aloud.
                if LESSON_END_TOKEN in full_reply or LESSON_END_TOKEN in buffer:
                    lesson_end_requested = True
                    full_reply = full_reply.replace(LESSON_END_TOKEN, "")
                    buffer = buffer.replace(LESSON_END_TOKEN, "")

                # Emit as soon as we have a complete trailing clause so
                # first audio starts earlier without waiting for extra tokens.
                if has_trailing_clause_boundary(buffer):
                    for cl in split_clauses(buffer):
                        if self._interrupted.is_set():
                            break
                        sentence_q.put(cl)
                    buffer = ""
                    continue

                # Try to extract complete clauses from the buffer
                clauses = split_clauses(buffer)

                if len(clauses) > 1:
                    for cl in clauses[:-1]:
                        if self._interrupted.is_set():
                            break
                        sentence_q.put(cl)

                    if self._interrupted.is_set():
                        break

                    # Keep the incomplete tail
                    buffer = clauses[-1]
        finally:
            # Close the LLM stream immediately on interrupt to
            # stop wasting tokens / network.
            try:
                stream.close()
            except Exception:
                pass

        # ── Flush remaining buffer (last clause) ──
        if buffer.strip() and not self._interrupted.is_set():
            sentence_q.put(buffer.strip())

        # Poison-pill to shut down the worker
        sentence_q.put(None)
        worker.join(timeout=30)

        if lesson_end_requested:
            self._check_lesson_end()

        self._current_ai_text = full_reply

        # Return the text that was *actually spoken* to the student.
        # When interrupted, this is only the spoken prefix.
        spoken_text = " ".join(self._spoken_sentences)
        return spoken_text if self._interrupted.is_set() else full_reply

    def _stream_llm_and_speak(self, extra_system_msg: str = "") -> str:
        """Prepare the current turn’s message list and stream LLM + TTS."""
        if extra_system_msg:
            self.messages.append({"role": "system", "content": extra_system_msg})
        return self._stream_and_speak(self.messages)

    # ────────────────────────────────────────────────────────────
    # Opening greeting
    # ────────────────────────────────────────────────────────────

    def _send_opening_greeting(self):
        if self._processing:
            return
        self._processing = True
        try:
            greeting_messages = [
                {"role": "system", "content": self.context},
                {"role": "user", "content": OPENING_GREETING_INSTRUCTION},
            ]
            spoken = self._stream_and_speak(greeting_messages)
            logger.info("Opening greeting: %s", spoken[:200])
            self.messages.append({"role": "assistant", "content": spoken})
            self._reset_silence_timer(calculate_proactive_delay(spoken))
        except Exception as e:
            logger.error("Error sending opening greeting: %s", e, exc_info=True)
        finally:
            self._processing = False
            # Check if there's a pending utterance from an interrupt
            self._process_pending_utterance()

    # ────────────────────────────────────────────────────────────
    # Turn handling
    # ────────────────────────────────────────────────────────────

    def _on_partial_utterance_detected(self, text: str):
        """Called frequently during student speech with partial STT results.
        If actual valid words are detected while the AI is talking, immediately
        interrupt it (STT-driven barge-in).
        """
        if not self._processing or self._interrupted.is_set():
            return

        with self._processing_lock:
            if not self._processing or self._interrupted.is_set():
                return

            if is_low_confidence_transcript(text):
                return
            if is_filler_utterance(text):
                return

            logger.info(
                "Valid words detected mid-utterance (%r) — triggering interrupt",
                text[:80],
            )
            # Queue it so it's captured immediately when AI turn aborts
            self._pending_utterance = text
            self.handle_interrupt()

    def _on_utterance_detected(self, text: str, elapsed_sec: float | None = None):
        self._cancel_silence_timer()
        with self._processing_lock:
            if self._processing:
                if is_low_confidence_transcript(text):
                    logger.info(
                        "Low-confidence/empty STT noise detected during AI turn — suppressing interrupt: %r",
                        text[:80],
                    )
                    return

                # T04: Filler words (e.g. "okay", "mm", "ага") should NOT
                # interrupt the agent's current thought — silently absorb them.
                if is_filler_utterance(text):
                    logger.info(
                        "Filler utterance detected during AI turn — suppressing interrupt: %r",
                        text[:80],
                    )
                    return
                # User spoke while AI is still responding — trigger interrupt
                # and queue this utterance so it's processed once the current
                # turn finishes.
                logger.info(
                    "User spoke during AI turn — triggering interrupt, "
                    "queuing utterance: %s",
                    text[:120] if text else "(empty)",
                )
                self._pending_utterance = text
                self.handle_interrupt()
                return
            self._processing = True

        threading.Thread(
            target=self._handle_turn,
            args=(text,),
            kwargs={"elapsed_sec": elapsed_sec},
            daemon=True,
        ).start()

    def _process_pending_utterance(self):
        """If a student utterance arrived during the previous AI turn,
        process it now."""
        text = self._pending_utterance
        self._pending_utterance = None
        if text is not None:
            logger.info("Processing pending interrupt utterance: %s", text[:120])
            self._on_utterance_detected(text)

    def _handle_turn(self, text: str, elapsed_sec: float | None = None):
        spoken_reply = None
        try:
            text = text.strip()
            if not text:
                return
            logger.info("Processing speech turn — text: %s", text[:200])

            if text.lower() in ("quit", "exit", "stop"):
                return

            if is_low_confidence_transcript(text):
                logger.info("Low-confidence STT transcript detected, asking to repeat")
                spoken = self._speak_direct_text(STT_CLARIFY_FALLBACK)
                self.messages.append({"role": "assistant", "content": spoken})
                return

            if is_lesson_end_request(text):
                logger.info("Student requested to finish lesson — closing session")
                closing_text = "Great work today. We can finish the lesson here."
                spoken = self._speak_direct_text(closing_text)
                self.messages.append({"role": "assistant", "content": spoken})
                self._check_lesson_end()
                return

            # ── 0. Track utterance for tempo adaptation ──
            now = time.monotonic()
            tempo_cfg = RUNTIME_CONFIG.tempo
            if self._last_student_turn_monotonic is None:
                turn_duration_sec = tempo_cfg.neutral_turn_duration_sec
            else:
                raw_delta = max(0.0, now - self._last_student_turn_monotonic)
                turn_duration_sec = max(
                    tempo_cfg.min_turn_duration_sec,
                    min(tempo_cfg.max_turn_duration_sec, raw_delta),
                )
            self._last_student_turn_monotonic = now

            self._student_utterances.append(text)
            self._student_turn_durations_sec.append(turn_duration_sec)
            if len(self._student_utterances) > 10:
                self._student_utterances = self._student_utterances[-10:]
            if len(self._student_turn_durations_sec) > 10:
                self._student_turn_durations_sec = self._student_turn_durations_sec[
                    -10:
                ]
            if RUNTIME_CONFIG.tempo.adapt_enabled:
                recent_utterances = self._student_utterances[-5:]
                self._current_tempo_hint = classify_tempo(
                    recent_utterances,
                    self._student_turn_durations_sec[-len(recent_utterances) :],
                )
                logger.debug("Tempo hint updated: %s", self._current_tempo_hint)

            # ── 1. Build interrupt context if the student interrupted ──
            interrupt_note = ""
            if self._interrupted.is_set() and self._spoken_sentences:
                spoken_so_far = " ".join(self._spoken_sentences)
                interrupt_note = INTERRUPT_CONTEXT_TEMPLATE.format(
                    spoken_text=spoken_so_far,
                    full_text=self._current_ai_text,
                )
                logger.info(
                    "Injecting interrupt context (%d chars)", len(interrupt_note)
                )
                self._interrupted.clear()

            # ── 2. RAG retrieval ──
            rag_context = ""
            if self._lesson_ready:
                rag_context = self.rag.build_retrieval_context(text)
                if rag_context:
                    logger.info("RAG injected %d chars of context", len(rag_context))

            # ── 3. Build messages & stream LLM + TTS ──
            if rag_context:
                self.messages.append({"role": "system", "content": rag_context})
            if interrupt_note:
                self.messages.append({"role": "system", "content": interrupt_note})

            self.messages.append({"role": "user", "content": text})

            spoken_reply = self._stream_llm_and_speak()
            logger.info("AI Reply (spoken): %s", spoken_reply[:200])

            # Only store the portion the student actually heard.
            # If interrupted, spoken_reply is the truncated prefix.
            self.messages.append({"role": "assistant", "content": spoken_reply})

            # ── Compress older history if it's getting long ──
            self._compress_history()

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
        finally:
            with self._processing_lock:
                self._processing = False
            # If the student spoke while we were busy, handle it now
            self._process_pending_utterance()
            self._reset_silence_timer(calculate_proactive_delay(spoken_reply))
