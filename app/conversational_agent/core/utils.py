import json
import re

# ─── Text splitting ──────────────────────────────────────────────────
# _SENTENCE_RE  — canonical sentence boundaries (. ! ?)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# _CLAUSE_RE — finer-grained: splits on ; : and em-dash so that long
# sentences are broken into shorter TTS chunks, reducing first-audio-byte
# latency.  Comma is intentionally excluded: comma-split audio fragments
# carry an unnatural tonal break that sounds jarring to the listener.
_CLAUSE_RE = re.compile(r"(?<=[.!?;:\u2014])\s+")
_TRAILING_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;:\u2014][\"')\]]?\s*$")
_TRAILING_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]?\s*$")
_TRAILING_CONTINUATION_RE = re.compile(
    r"\b(and|or|but|so|then|also|because|if|when|that|to|of|for|with|about|into|from|as)\s*$",
    re.IGNORECASE,
)

# Minimum character length for a clause to be spoken on its own.
# Shorter fragments are accumulated into the next clause.
_MIN_CLAUSE_LEN = 30


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences at . ! ? boundaries."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_clauses(text: str) -> list[str]:
    """Split *text* at clause boundaries (. ! ? ; : , —).

    Very short fragments are merged with the following clause so the
    TTS model receives meaningful input.
    """
    raw = _CLAUSE_RE.split(text.strip())
    merged: list[str] = []
    buf = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buf = f"{buf} {part}".strip() if buf else part
        if len(buf) >= _MIN_CLAUSE_LEN:
            merged.append(buf)
            buf = ""
    if buf:
        # If there's leftover text, it is appended as the final clause,
        # even if it's shorter than MIN_CLAUSE_LEN.
        merged.append(buf)
    return merged


def has_trailing_clause_boundary(text: str) -> bool:
    """Return True when *text* currently ends at a clause boundary."""
    if not text or not text.strip():
        return False
    return bool(_TRAILING_CLAUSE_BOUNDARY_RE.search(text))


def classify_utterance_ending_quality(text: str) -> str:
    """Classify utterance ending quality as ``strong``, ``weak``, or ``partial``."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "partial"

    if _TRAILING_SENTENCE_BOUNDARY_RE.search(cleaned):
        return "strong"

    if cleaned.endswith("-") or _TRAILING_CONTINUATION_RE.search(cleaned):
        return "partial"

    if len(cleaned.split()) <= 2:
        return "partial"

    if has_trailing_clause_boundary(cleaned):
        return "weak"

    return "weak"


# ─── Language detection ──────────────────────────────────────────────
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# ─── Filler utterance detection ─────────────────────────────────────
# Short back-channel words that indicate acknowledgment/thinking rather
# than a real conversational turn. The agent should not interrupt its
# current speech for these.
_FILLER_WORDS_EN: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "uh",
        "um",
        "ah",
        "uhh",
        "umm",
        "understood",
        "sure",
        "yeah",
        "yep",
        "yup",
        "right",
        "mhm",
        "mm",
        "hmm",
        "hm",
        "got it",
        "i see",
    }
)
_FILLER_WORDS_RU: frozenset[str] = frozenset(
    {
        "мм",
        "ага",
        "угу",
        "хорошо",
        "понял",
        "понятно",
        "окей",
        "ок",
        "да",
        "ясно",
        "ладно",
    }
)
_ALL_FILLER_WORDS: frozenset[str] = _FILLER_WORDS_EN | _FILLER_WORDS_RU

_LESSON_END_REQUEST_RE = re.compile(
    r"\b("
    r"finish|end|complete|wrap\s*up|stop\s+the\s+lesson|end\s+the\s+lesson|"
    r"законч|заверш|останови\s+урок|закончи\s+урок|"
    r"аяктаг|сабакты\s+бүтүр"
    r")\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Return ``'ru'`` if *text* is predominantly Cyrillic, else ``'en'``."""
    if not text:
        return "en"
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return "en"
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    return "ru" if cyrillic_count > len(alpha_chars) / 2 else "en"


def is_filler_utterance(text: str) -> bool:
    """Return ``True`` when *text* is a short filler / back-channel word.

    Fillers are acknowledged (logged) but must NOT trigger a new LLM
    turn — the agent should continue its current thought.
    """
    cleaned = (text or "").strip().lower().rstrip(".!?,")
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) > 3:
        return False
    return cleaned in _ALL_FILLER_WORDS


def is_low_confidence_transcript(text: str) -> bool:
    """Heuristic detector for likely-bad STT transcriptions.

    The STT stack does not always expose confidence scores in this stage,
    so we apply conservative text-shape checks and ask the student to repeat
    when a transcript looks unusable.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return True

    # Extremely short tokens are often accidental noise picks.
    if len(cleaned) <= 1:
        return True

    # Keep normal acknowledgement tokens (e.g. "ok") out of this detector.
    if is_filler_utterance(cleaned):
        return False

    alpha = [c for c in cleaned if c.isalpha()]
    if not alpha:
        return True

    # Non-letter heavy output is usually STT artefact/noise.
    if len(alpha) / max(1, len(cleaned)) < 0.45:
        return True

    # Repeated single-letter blobs ("аааа", "mmmm") are unreliable.
    lowered = cleaned.lower()
    if len(set(lowered.replace(" ", ""))) == 1 and len(lowered.replace(" ", "")) >= 4:
        return True

    return False


def is_lesson_end_request(text: str) -> bool:
    """Return ``True`` when student explicitly asks to finish the lesson."""
    return bool(_LESSON_END_REQUEST_RE.search((text or "").strip()))


def extract_text_from_lesson_context(lesson_context: str) -> str:
    """Extract human-readable lesson content for accurate language detection."""
    try:
        ctx = json.loads(lesson_context)
    except json.JSONDecodeError:
        return lesson_context

    parts: list[str] = []
    if isinstance(ctx, dict):
        lesson_title = ctx.get("lesson_title")
        if isinstance(lesson_title, str):
            parts.append(lesson_title)

        materials = ctx.get("materials")
        if isinstance(materials, list):
            for mat in materials:
                if not isinstance(mat, dict):
                    continue
                title = mat.get("title")
                if isinstance(title, str):
                    parts.append(title)
                content = mat.get("content")
                if isinstance(content, str):
                    parts.append(content)

    return " ".join(parts).strip() or lesson_context


def estimate_ai_turn_duration_sec(text: str) -> float:
    """Estimate how long the AI's spoken turn will take to finish playing."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    # Average speaking rate ~140 words per minute ~2.3 words/sec
    words = max(1, len(cleaned.split()))
    return words * 0.42


def estimate_cognitive_load_sec(text: str) -> float:
    """Estimate the time needed by the student to comprehend the AI's turn."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    # Cognitive load varies. Add ~0.15s per word for thinking time.
    words = max(1, len(cleaned.split()))
    return min(words * 0.15, 10.0)
