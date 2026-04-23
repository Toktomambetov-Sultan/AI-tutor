"""Speech tempo adaptation — classify student pace and shape TTS text.

The tempo hint is derived from the student's recent utterances and used
to apply lightweight prosody shaping to the agent's TTS input text.

Client-side players MAY also receive the hint as a metadata signal
(``MessageType.TEMPO_HINT``) but are not required to honour it — the
signal is purely advisory and the system degrades gracefully to the
default tempo when ignored.

Design notes
------------
- All functions are pure / stateless — easy to unit-test.
- Shaping is text-level only (comma insertion / ellipsis stripping);
  no model or audio changes are made.
- The ``NORMAL`` constant is the safe no-op default.
"""

from __future__ import annotations

import re

from core.config import RUNTIME_CONFIG

# ── Public hint constants ─────────────────────────────────────────────
SLOW: str = "slow"
NORMAL: str = "normal"
FAST: str = "fast"


# ──────────────────────────────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────────────────────────────

def _words_per_minute(utterances: list[str], durations_sec: list[float]) -> float:
    """Estimate WPM from parallel lists of utterance text and durations."""
    total_words = sum(len(u.split()) for u in utterances)
    total_sec = sum(durations_sec)
    if total_sec < 0.1:
        return 0.0
    return total_words / total_sec * 60.0


def _sanitize_turn_durations(durations_sec: list[float]) -> tuple[list[float], bool]:
    """Return safe clamped durations and whether any unsafe input was observed."""
    cfg = RUNTIME_CONFIG.tempo
    sanitized: list[float] = []
    had_invalid = False
    for duration in durations_sec:
        if duration <= 0.0:
            had_invalid = True
            continue
        if duration > cfg.max_turn_duration_sec * 2.0:
            had_invalid = True
            continue
        sanitized.append(
            max(cfg.min_turn_duration_sec, min(cfg.max_turn_duration_sec, duration))
        )
    return sanitized, had_invalid


def classify_tempo(
    utterances: list[str],
    durations_sec: list[float] | None = None,
) -> str:
    """Return ``SLOW`` / ``NORMAL`` / ``FAST`` based on recent utterances.

    When *durations_sec* is provided and length matches *utterances*, WPM
    is used for classification.  Otherwise, average word count per
    utterance is used as a lightweight proxy.

    Always returns ``NORMAL`` when ``RUNTIME_CONFIG.tempo.adapt_enabled``
    is ``False`` or when there are no utterances to analyse.
    """
    cfg = RUNTIME_CONFIG.tempo
    if not cfg.adapt_enabled or not utterances:
        return NORMAL

    if durations_sec and len(durations_sec) == len(utterances):
        safe_durations, had_invalid_timing = _sanitize_turn_durations(durations_sec)
        # Safety-first fallback: if timing data is malformed, avoid
        # over-confident fast/slow hints.
        if had_invalid_timing or len(safe_durations) != len(utterances):
            return NORMAL

        wpm = _words_per_minute(utterances, safe_durations)
        if 0 < wpm < cfg.slow_wpm_threshold:
            return SLOW
        if wpm >= cfg.fast_wpm_threshold:
            return FAST
        return NORMAL

    # Fallback: word-count heuristic
    avg_words = sum(len(u.split()) for u in utterances) / len(utterances)
    if avg_words < 4:
        return SLOW
    if avg_words > 12:
        return FAST
    return NORMAL


# ──────────────────────────────────────────────────────────────────────
# Prosody shaping
# ──────────────────────────────────────────────────────────────────────

# Insert a breath-pause after common conjunctions that begin clauses
# when speaking slowly, so TTS paces itself more deliberately.
_SLOW_CONJUNCTION_RE = re.compile(
    r"\b(and|but|so|then|also|because)\s+",
    re.IGNORECASE,
)


def apply_tempo_shaping(text: str, hint: str) -> str:
    """Return *text* with lightweight prosody adjustments for *hint*.

    - ``SLOW``  : insert soft comma-pauses after clause-opening conjunctions.
    - ``FAST``  : strip ellipsis delays (``...`` → ``.``).
    - ``NORMAL``: return text unchanged.
    """
    if hint == SLOW:
        return _SLOW_CONJUNCTION_RE.sub(lambda m: m.group(0).rstrip() + ", ", text)
    if hint == FAST:
        return re.sub(r"\.\.\.", ".", text)
    return text
