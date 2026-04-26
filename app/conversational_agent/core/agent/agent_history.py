import logging
from core.config import RUNTIME_CONFIG

logger = logging.getLogger(__name__)


class AgentHistoryMixin:
    """Provides history summarisation capabilities."""

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
