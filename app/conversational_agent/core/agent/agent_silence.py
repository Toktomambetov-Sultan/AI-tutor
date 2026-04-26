import logging
import threading

from core.config import RUNTIME_CONFIG
from core.prompts import SILENCE_CONTEXT_TEMPLATE
from core.protocol import MessageType, QueueMessage

logger = logging.getLogger(__name__)


class AgentSilenceMixin:
    """Manages silence timeouts and proactive engagement."""

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
            if getattr(self, "_silence_timer", None) is not None:
                self._silence_timer.cancel()
                self._silence_timer = None

    def _reset_silence_timer(self, delay_sec: float | None = None) -> None:
        """Schedule proactive re-engagement when the student stays silent."""
        if getattr(self, "_session_finished", False):
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
            if getattr(self, "_session_finished", False):
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
