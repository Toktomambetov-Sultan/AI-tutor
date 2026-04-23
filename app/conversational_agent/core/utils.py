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
_MIN_CLAUSE_LEN = 12


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
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
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
        "ok", "okay", "uh", "um", "ah", "uhh", "umm",
        "understood", "sure", "yeah", "yep", "yup", "right",
        "mhm", "mm", "hmm", "hm", "got it", "i see",
    }
)
_FILLER_WORDS_RU: frozenset[str] = frozenset(
    {
        "мм", "ага", "угу", "хорошо", "понял", "понятно",
        "окей", "ок", "да", "ясно", "ладно",
    }
)
_ALL_FILLER_WORDS: frozenset[str] = _FILLER_WORDS_EN | _FILLER_WORDS_RU


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
