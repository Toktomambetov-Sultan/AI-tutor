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
    VOICE_GENDER_SYSTEM_MSG,
    VOICE_STYLE_GUARDRAIL,
)
from core.protocol import MessageType, QueueMessage
from core.rag import LessonRAG
from core.resources import SharedResources, build_default_shared_resources
from core.session_state import SessionState
from core.tempo import NORMAL
from core.turn_policy import TimingAwareTurnGate
from core.utils import (
    detect_language,
    extract_text_from_lesson_context,
)

from core.agent import (
    AgentHistoryMixin,
    AgentTTSMixin,
    AgentTurnMixin,
    AgentSilenceMixin,
)

logger = logging.getLogger(__name__)

# Suppress Vosk's own logging (we log results ourselves)
SetLogLevel(-1)


class ConversationalAgent(
    AgentHistoryMixin, AgentTTSMixin, AgentTurnMixin, AgentSilenceMixin
):
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
