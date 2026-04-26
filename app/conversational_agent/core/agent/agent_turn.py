import logging
import threading
import time

from core.config import RUNTIME_CONFIG
from core.prompts import (
    INTERRUPT_CONTEXT_TEMPLATE,
    OPENING_GREETING_INSTRUCTION,
    STT_CLARIFY_FALLBACK,
)
from core.tempo import classify_tempo
from core.turn_policy import calculate_proactive_delay
from core.utils import (
    is_filler_utterance,
    is_lesson_end_request,
    is_low_confidence_transcript,
)

logger = logging.getLogger(__name__)


class AgentTurnMixin:
    """Manages the conversation turn loop and interruptions."""

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
            self._process_pending_utterance()

    def _on_partial_utterance_detected(self, text: str):
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

                if is_filler_utterance(text):
                    logger.info(
                        "Filler utterance detected during AI turn — suppressing interrupt: %r",
                        text[:80],
                    )
                    return

                logger.info(
                    "User spoke during AI turn — triggering interrupt, queuing utterance: %s",
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

            now = time.monotonic()
            tempo_cfg = RUNTIME_CONFIG.tempo
            if getattr(self, "_last_student_turn_monotonic", None) is None:
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
                logger.debug(
                    "Tempo hint updated: %s",
                    getattr(self, "_current_tempo_hint", "NORMAL"),
                )

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

            rag_context = ""
            if self._lesson_ready:
                rag_context = self.rag.build_retrieval_context(text)
                if rag_context:
                    logger.info("RAG injected %d chars of context", len(rag_context))

            if rag_context:
                self.messages.append({"role": "system", "content": rag_context})
            if interrupt_note:
                self.messages.append({"role": "system", "content": interrupt_note})

            self.messages.append({"role": "user", "content": text})

            spoken_reply = self._stream_llm_and_speak()
            logger.info("AI Reply (spoken): %s", spoken_reply[:200])

            self.messages.append({"role": "assistant", "content": spoken_reply})

            self._compress_history()

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
        finally:
            with self._processing_lock:
                self._processing = False
            self._process_pending_utterance()
            self._reset_silence_timer(
                calculate_proactive_delay(spoken_reply) if spoken_reply else None
            )
