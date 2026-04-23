"""Timing-aware turn decision policy.

Determines whether the agent should reply immediately when an utterance
boundary fires, or hold back and wait for the student to continue
speaking.  If the student stays silent longer than *force_reply_sec*
after a held partial utterance, the reply is triggered regardless.

Design notes
------------
- ``decide_turn`` is a pure function — easy to unit-test.
- ``TimingAwareTurnGate`` wraps the real ``on_utterance`` callback;
  upstream code (audio processor) keeps the same interface.
- Thread-safe: uses a single ``threading.Lock`` and ``threading.Timer``.
- Backward-compatible: callers that pass a callback function see no
  change in the public API.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

from core.config import RUNTIME_CONFIG
from core.utils import classify_utterance_ending_quality

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _ends_with_sentence_boundary(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text.strip()))


def decide_turn(
    text: str,
    silence_after_end_sec: float | None = None,
    recent_silence_sec: float | None = None,
) -> str:
    """Return ``"respond"`` or ``"wait"`` for the given utterance text.

    ``"wait"`` is returned only when the utterance looks like a partial
    thought (too few words **and** no terminal punctuation).  All other
    utterances trigger an immediate reply.
    """
    cfg = RUNTIME_CONFIG.turn_policy
    ending_quality = classify_utterance_ending_quality(text)

    if _ends_with_sentence_boundary(text):
        if (
            silence_after_end_sec is not None
            and silence_after_end_sec < cfg.min_strong_boundary_silence_sec
        ):
            return "wait"
        return "respond"

    if ending_quality == "partial":
        if silence_after_end_sec is not None and (
            silence_after_end_sec >= cfg.partial_respond_silence_sec
        ):
            return "respond"
        if recent_silence_sec is not None and (
            recent_silence_sec >= cfg.partial_respond_silence_sec
        ):
            return "respond"
        return "wait"

    if _word_count(text) >= cfg.min_words_for_complete:
        if (
            recent_silence_sec is not None
            and recent_silence_sec < cfg.min_recent_silence_sec
        ):
            return "wait"
        return "respond"

    return "wait"


# ──────────────────────────────────────────────────────────────────────
# Turn gate
# ──────────────────────────────────────────────────────────────────────

class TimingAwareTurnGate:
    """Wraps *on_utterance* with wait-and-merge / force-reply logic.

    When a (possibly partial) utterance arrives via ``feed()``:

    - decision == ``"respond"`` → forward to *on_utterance* immediately.
    - decision == ``"wait"``    → start a ``force_reply_sec`` timer.

      * If another utterance arrives before the timer fires: cancel the
        timer, merge the texts, and re-evaluate.
      * If the timer fires first: call *on_utterance* with the
        accumulated text (force reply on silence).
    """

    def __init__(self, on_utterance: Callable[..., None]) -> None:
        self._on_utterance = on_utterance
        self._lock = threading.Lock()
        self._pending_text: str | None = None
        self._timer: threading.Timer | None = None
        self._last_feed_monotonic: float | None = None
        # Timestamp when pending text accumulation started (for elapsed-silence calc).
        self._pending_since_monotonic: float | None = None

    # ── public ──────────────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """Process an incoming utterance boundary from the audio processor."""
        text = text.strip()
        if not text:
            return

        now = time.monotonic()

        with self._lock:
            recent_silence_sec = (
                None
                if self._last_feed_monotonic is None
                else max(0.0, now - self._last_feed_monotonic)
            )
            self._last_feed_monotonic = now

            # Cancel any running force-reply timer
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            # Merge with accumulated partial if one exists
            merged_from_pending = bool(self._pending_text)
            if self._pending_text:
                text = (self._pending_text + " " + text).strip()

            decision_recent_silence = None if merged_from_pending else recent_silence_sec

            if decide_turn(
                text,
                silence_after_end_sec=recent_silence_sec,
                recent_silence_sec=decision_recent_silence,
            ) == "respond":
                self._pending_text = None
                # Dispatch outside the lock to avoid holding it during the callback.
                # elapsed_sec=None for user-triggered responses (no silence gap to report).
                dispatch_text = text
                self._pending_since_monotonic = None

            else:
                # Hold the utterance and start force-reply timer
                self._pending_text = text
                if self._pending_since_monotonic is None:
                    self._pending_since_monotonic = now
                force_sec = RUNTIME_CONFIG.turn_policy.force_reply_sec
                logger.debug(
                    "Turn gate: holding partial %r — force-reply in %.1fs",
                    text[:80],
                    force_sec,
                )
                self._timer = threading.Timer(force_sec, self._force_reply)
                self._timer.daemon = True
                self._timer.start()
                return  # do not dispatch yet

        self._dispatch(dispatch_text)

    def _elapsed_since_pending(self) -> float | None:
        """Return seconds since the pending utterance first accumulated, or None."""
        if self._pending_since_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._pending_since_monotonic)

    def close(self) -> None:
        """Cancel any pending timer (call on session teardown)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # ── internal ────────────────────────────────────────────────────

    def _force_reply(self) -> None:
        """Called by the timer thread when silence lasted too long."""
        with self._lock:
            text = self._pending_text
            self._pending_text = None
            self._timer = None

        if text:
            logger.info(
                "Turn gate: force-reply triggered after silence — text: %r",
                text[:120],
            )
            elapsed_sec = self._elapsed_since_pending()
            self._pending_since_monotonic = None
            self._dispatch(text, elapsed_sec=elapsed_sec)

    def _dispatch(self, text: str, elapsed_sec: float | None = None) -> None:
        try:
            # Backward-compatible: if callback does not accept elapsed_sec,
            # call it with text-only.
            if elapsed_sec is None:
                self._on_utterance(text)
            else:
                try:
                    self._on_utterance(text, elapsed_sec=elapsed_sec)
                except TypeError:
                    self._on_utterance(text)
        except Exception:
            logger.exception("on_utterance callback raised in TimingAwareTurnGate")
