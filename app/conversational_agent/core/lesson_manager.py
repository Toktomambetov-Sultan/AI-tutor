"""Per-lesson setup: RAG ingestion, language detection, STT model selection.

Manages:
- Lesson context ingestion and RAG setup
- Automatic language detection for per-lesson TTS/STT
- STT recognizer initialization based on lesson language
"""

import logging

from vosk import KaldiRecognizer, Model as VoskModel

from core.audio_processor import TARGET_SAMPLE_RATE, RealtimeAudioProcessor
from core.prompts import FALLBACK_SYSTEM_PROMPT
from core.rag import LessonRAG
from core.resources import SharedResources
from core.utils import detect_language, extract_text_from_lesson_context

logger = logging.getLogger(__name__)


class LessonManager:
    """Manages lesson context, RAG, and language-specific settings."""

    def __init__(
        self,
        resources: SharedResources,
        openai_client,
        on_utterance_callback,
    ):
        """
        Args:
            resources: SharedResources with models and clients
            openai_client: OpenAI client for RAG
            on_utterance_callback: Callback for when speech is detected
        """
        self._resources = resources
        self._openai_client = openai_client
        self._on_utterance_callback = on_utterance_callback

        # Lesson state
        self.rag: LessonRAG | None = None
        self.context: str = FALLBACK_SYSTEM_PROMPT
        self._lesson_ready = False
        self._language = "en"

        # STT processor
        self._processor: RealtimeAudioProcessor | None = None

    def is_ready(self) -> bool:
        """Check if lesson context has been loaded."""
        return self._lesson_ready

    def get_language(self) -> str:
        """Get the detected language (en or ru)."""
        return self._language

    def get_context_for_rag(self) -> str:
        """Get the system prompt/RAG context."""
        return self.context

    def init_rag(self, lesson_context: str) -> None:
        """Initialize RAG with lesson context and detect language.

        Args:
            lesson_context: JSON string containing lesson materials
        """
        self.rag = LessonRAG(self._openai_client)
        self._lesson_ready = False
        self._language = "en"

        if lesson_context:
            self._ingest_lesson(lesson_context)
        else:
            self.context = FALLBACK_SYSTEM_PROMPT

    def _ingest_lesson(self, lesson_context: str) -> None:
        """Ingest lesson context into RAG and update system prompt.

        Args:
            lesson_context: JSON string with lesson materials
        """
        try:
            self.rag.ingest(lesson_context)
            self.context = self.rag.build_system_prompt()
            self._lesson_ready = True

            # Detect language from lesson content
            extracted_text = extract_text_from_lesson_context(lesson_context)
            self._language = detect_language(extracted_text)

            logger.info(
                "Lesson RAG ready: %d chunks indexed for %r (language=%s)",
                len(self.rag.documents) if hasattr(self.rag, "documents") else 0,
                extracted_text[:50],
                self._language,
            )
        except Exception as e:
            logger.error("Error initializing lesson RAG: %s", e, exc_info=True)
            self.context = FALLBACK_SYSTEM_PROMPT
            self._lesson_ready = False

    def init_processor(
        self,
        recognizer_factory=None,
        audio_processor_factory=None,
    ) -> None:
        """Initialize STT processor with language-appropriate Vosk model.

        Args:
            recognizer_factory: Callable to create KaldiRecognizer (for DI)
            audio_processor_factory: Callable to create RealtimeAudioProcessor (for DI)
        """
        _recognizer_factory = recognizer_factory or KaldiRecognizer
        _audio_processor_factory = audio_processor_factory or RealtimeAudioProcessor

        # Pick the correct Vosk model based on detected language
        vosk_model = (
            self._resources.vosk_model_ru
            if self._language == "ru" and self._resources.vosk_model_ru
            else self._resources.vosk_model_en
        )

        recognizer = (
            _recognizer_factory(vosk_model, TARGET_SAMPLE_RATE) if vosk_model else None
        )

        logger.info(
            "Session language=%s, Vosk recognizer=%s",
            self._language,
            "ready" if recognizer else "unavailable",
        )

        self._processor = _audio_processor_factory(
            on_utterance=self._on_utterance_callback,
            recognizer=recognizer,
        )

    def process_audio_chunk(self, chunk_bytes: bytes) -> None:
        """Feed audio chunk to the STT processor.

        Args:
            chunk_bytes: Raw WebM/Opus audio data
        """
        if self._processor:
            self._processor.feed(chunk_bytes)

    def close(self) -> None:
        """Clean up STT processor and RAG resources."""
        if self._processor:
            try:
                self._processor.close()
            except Exception as e:
                logger.warning("Error closing audio processor: %s", e)
