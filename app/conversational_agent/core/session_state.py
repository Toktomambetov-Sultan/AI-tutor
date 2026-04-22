"""Per-session state management: initialization, interrupts, turn tracking.

Manages:
- Session lifecycle (mutable state, flags, locks)
- Interrupt handling and signal propagation
- Conversation turn queuing and re-processing
- Message history compression
"""

import logging
import threading

from core.protocol import MessageType, QueueMessage

logger = logging.getLogger(__name__)


class SessionState:
    """Manages mutable per-session state: interrupts, turn tracking, message history."""

    # Message history compression threshold
    _MAX_MESSAGES_BEFORE_COMPRESSION = 20

    def __init__(self, response_queue, loop):
        """
        Args:
            response_queue: Queue for sending signals to gRPC servicer
            loop: asyncio event loop for thread-safe queue operations
        """
        self.response_queue = response_queue
        self.loop = loop

        # Interrupt state
        self._interrupted = threading.Event()

        # Turn processing state
        self._processing = False
        self._processing_lock = threading.Lock()
        self._pending_utterance: str | None = None

        # Conversation state
        self._first_turn = True
        self.messages: list[dict] = []

    def get_interrupted_flag(self) -> threading.Event:
        """Return the interrupt event flag for LLMPipeline."""
        return self._interrupted

    def is_processing(self) -> bool:
        """Check if currently processing a turn."""
        with self._processing_lock:
            return self._processing

    def set_processing(self, value: bool) -> None:
        """Set the processing flag (thread-safe)."""
        with self._processing_lock:
            self._processing = value

    def is_first_turn(self) -> bool:
        """Check if this is the first turn."""
        return self._first_turn

    def mark_first_turn_done(self) -> None:
        """Mark that the first turn has been processed."""
        self._first_turn = False

    def set_pending_utterance(self, text: str | None) -> None:
        """Queue an utterance to process after the current turn finishes."""
        self._pending_utterance = text

    def get_and_clear_pending_utterance(self) -> str | None:
        """Get the pending utterance and clear it."""
        text = self._pending_utterance
        self._pending_utterance = None
        return text

    def handle_interrupt(self) -> None:
        """Called when the student starts speaking while AI is talking.

        Sets the interrupt flag, drains any queued audio so the client
        stops hearing stale speech, and sends an explicit ``interrupt``
        signal so the frontend can halt playback immediately.
        """
        self._interrupted.set()

        def _drain_and_signal():
            """Drain queued audio and send interrupt signal."""
            try:
                # Drain any pending audio from the queue
                while True:
                    try:
                        self.response_queue.get_nowait()
                    except Exception:
                        break
            except Exception as e:
                logger.warning("Error draining response queue: %s", e)

            # Send explicit interrupt signal to the frontend
            try:
                self.loop.call_soon_threadsafe(
                    self.response_queue.put_nowait,
                    QueueMessage(MessageType.INTERRUPT, ""),
                )
            except Exception as e:
                logger.warning("Error sending interrupt signal: %s", e)

        threading.Thread(target=_drain_and_signal, daemon=True).start()

    def compress_history(self) -> None:
        """Summarise older conversation turns into one system message.

        When the message list exceeds _MAX_MESSAGES_BEFORE_COMPRESSION,
        this method extracts all user/assistant exchanges (except the most
        recent one) and replaces them with a single summarization system message.

        This keeps the context manageable without losing information.
        """
        if len(self.messages) <= self._MAX_MESSAGES_BEFORE_COMPRESSION:
            return

        logger.info(
            "Compressing conversation history from %d to ~6 messages",
            len(self.messages),
        )

        # Extract system messages and recent conversation
        system_messages = [m for m in self.messages if m["role"] == "system"]
        user_assistant_pairs = [m for m in self.messages if m["role"] != "system"]

        if len(user_assistant_pairs) <= 2:
            return

        # Keep the most recent exchange, compress everything else
        recent = user_assistant_pairs[-2:]
        to_compress = user_assistant_pairs[:-2]

        # Build summary from old exchanges
        summary_text = "Previous conversation summary:\n"
        for msg in to_compress:
            role = msg["role"].upper()
            content = msg["content"][:200]  # truncate long messages
            summary_text += f"- {role}: {content}\n"

        # Rebuild messages: system messages + summary + recent exchange
        self.messages = (
            system_messages
            + [
                {"role": "system", "content": summary_text},
            ]
            + recent
        )

        logger.info(
            "Compressed history: %d → %d messages",
            len(user_assistant_pairs) + len(system_messages),
            len(self.messages),
        )

    def process_pending_utterance(self) -> str | None:
        """Return pending utterance if any, for reprocessing."""
        return self.get_and_clear_pending_utterance()

    def reset_interrupt(self) -> None:
        """Clear the interrupt flag (called after handling interrupt in message flow)."""
        self._interrupted.clear()
