"""LLM streaming + per-sentence TTS synthesis pipeline.

This module handles the core streaming/TTS orchestration:
- Streams LLM tokens and buffers them at clause boundaries
- Synthesizes each clause to audio in a worker thread (producer/consumer pattern)
- Manages interrupt handling and audio queueing
- Overlaps LLM generation with TTS to reduce perceived latency
"""

import io
import logging
import queue
import threading

import scipy.io.wavfile
import torch

from core.protocol import MessageType, QueueMessage
from core.resources import SharedResources
from core.utils import split_clauses, detect_language

logger = logging.getLogger(__name__)


class LLMPipeline:
    """Manages LLM streaming and real-time sentence-level TTS synthesis."""

    def __init__(
        self,
        resources: SharedResources,
        response_queue,
        loop,
        interrupted_flag: threading.Event,
    ):
        """
        Args:
            resources: SharedResources containing OpenAI client and TTS models
            response_queue: Queue for sending audio/messages to gRPC servicer
            loop: asyncio event loop for thread-safe queue operations
            interrupted_flag: threading.Event signaling student interrupt
        """
        self._resources = resources
        self.response_queue = response_queue
        self.loop = loop
        self._interrupted = interrupted_flag

        # Per-utterance state
        self._spoken_sentences: list[str] = []
        self._current_ai_text: str = ""
        self._language: str = "en"  # Language for TTS synthesis

    def set_language(self, language: str) -> None:
        """Set the output language for TTS (en or ru)."""
        self._language = language

    def synthesise_sentence(self, sentence: str, language: str | None = None) -> bytes:
        """Synthesize a single sentence to WAV bytes (English or Russian).

        Args:
            sentence: Text to synthesize
            language: Override language (defaults to self._language)

        Returns:
            WAV audio bytes
        """
        lang = language or self._language
        if lang == "ru" and self._resources.ru_tts_model is not None:
            audio_tensor = self._resources.ru_tts_model.apply_tts(
                text=sentence,
                speaker=self._resources.ru_speaker,
            )
            sample_rate = self._resources.ru_sample_rate
        else:
            audio_tensor = self._resources.tts_model.generate_audio(
                self._resources.voice_state, sentence
            )
            sample_rate = self._resources.tts_model.sample_rate

        buf = io.BytesIO()
        audio_np = (
            audio_tensor.cpu().numpy()
            if torch.is_tensor(audio_tensor)
            else audio_tensor
        )
        scipy.io.wavfile.write(buf, sample_rate, audio_np)
        return buf.getvalue()

    def send_audio(self, wav_bytes: bytes, ai_text: str = "") -> None:
        """Queue WAV audio + metadata for the gRPC servicer to stream.

        Delivers complete sentence audio to avoid cutting off mid-word.
        The interrupt flag is only checked before starting a new sentence;
        the frontend handles smooth fade-out.

        Args:
            wav_bytes: WAV audio data
            ai_text: Associated text label (shown to student)
        """
        if self._interrupted.is_set():
            return

        # Send the text label first so the servicer can attach it
        if ai_text:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AI_TEXT, ai_text),
            )

        # Send audio in chunks to avoid large queue entries
        chunk_size = 32768
        for i in range(0, len(wav_bytes), chunk_size):
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AUDIO, wav_bytes[i : i + chunk_size]),
            )
        self.loop.call_soon_threadsafe(
            self.response_queue.put_nowait, QueueMessage(MessageType.END)
        )

    def stream_and_speak(self, messages: list[dict]) -> str:
        """Stream LLM for messages, split into TTS clauses, and synthesize audio.

        Uses a **producer/consumer pattern**:
        - Main thread streams LLM tokens and pushes complete clauses into queue
        - Worker thread synthesizes and sends TTS audio in parallel
        - This overlaps LLM generation with TTS, reducing perceived latency

        Args:
            messages: Message list for OpenAI chat.completions.create

        Returns:
            The *spoken* reply text (truncated at interrupt point if barging in)
        """
        self._interrupted.clear()
        self._spoken_sentences = []
        self._current_ai_text = ""

        # ── TTS worker (consumer) ──
        sentence_q: queue.Queue[str | None] = queue.Queue()

        def _synthesise_and_send(clause: str):
            """Synthesise *clause* and queue the resulting audio."""
            logger.info("TTS clause: %s", clause)
            wav = self.synthesise_sentence(clause)
            self.send_audio(wav, ai_text=clause)
            self._spoken_sentences.append(clause)

        def _tts_worker():
            """Consume clauses from *sentence_q* and synthesise + send."""
            while True:
                clause = sentence_q.get()
                if clause is None or self._interrupted.is_set():
                    break
                _synthesise_and_send(clause)

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        # ── Stream LLM tokens (producer) ──
        stream = self._resources.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        buffer = ""  # accumulates tokens until a clause boundary
        full_reply = ""  # everything the LLM produced

        try:
            for chunk in stream:
                if self._interrupted.is_set():
                    break

                delta = chunk.choices[0].delta
                if not delta.content:
                    continue

                token = delta.content
                buffer += token
                full_reply += token

                # Try to extract complete clauses from the buffer
                clauses = split_clauses(buffer)

                if len(clauses) > 1:
                    for cl in clauses[:-1]:
                        if self._interrupted.is_set():
                            break
                        sentence_q.put(cl)

                    if self._interrupted.is_set():
                        break

                    # Keep the incomplete tail
                    buffer = clauses[-1]
        finally:
            # Close the LLM stream immediately on interrupt to
            # stop wasting tokens / network
            try:
                stream.close()
            except Exception:
                pass

        # ── Flush remaining buffer (last clause) ──
        if buffer.strip() and not self._interrupted.is_set():
            sentence_q.put(buffer.strip())

        # Poison-pill to shut down the worker
        sentence_q.put(None)
        worker.join(timeout=30)

        self._current_ai_text = full_reply

        # Return the text that was *actually spoken* to the student.
        # When interrupted, this is only the spoken prefix.
        spoken_text = " ".join(self._spoken_sentences)
        return spoken_text if self._interrupted.is_set() else full_reply

    def get_current_ai_text(self) -> str:
        """Return the full LLM reply (including unspoken portion after interrupt)."""
        return self._current_ai_text

    def get_spoken_sentences(self) -> list[str]:
        """Return list of sentences actually synthesized and sent to client."""
        return self._spoken_sentences.copy()
