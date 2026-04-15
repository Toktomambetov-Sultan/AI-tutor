"""
Per-session conversational agent.

Orchestrates the full STT → RAG → LLM (streaming) → TTS pipeline for a
single student session.

Key improvements over the original implementation:
  - **Whisper STT** via OpenAI API (replaces batch Google STT)
  - **LLM streaming** with sentence-boundary chunking
  - **Per-sentence TTS** so the first audio reaches the student sooner
  - **Barge-in support** — student can interrupt mid-sentence; the agent
    knows where in the response text it was stopped and resumes cleanly
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import threading

import scipy.io.wavfile
import torch
from openai import OpenAI
from pocket_tts import TTSModel
from speech_recognition import Recognizer

from core.audio_processor import (
    BYTES_PER_SAMPLE,
    TARGET_SAMPLE_RATE,
    RealtimeAudioProcessor,
)
from core.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    OPENING_GREETING_INSTRUCTION,
    INTERRUPT_CONTEXT_TEMPLATE,
)
from core.rag import LessonRAG

logger = logging.getLogger(__name__)

# Regex that splits on sentence-ending punctuation while keeping the
# punctuation attached to the sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences at . ! ? boundaries."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class ConversationalAgent:
    """Per-session agent.  Receives pre-loaded shared resources to avoid
    re-loading the TTS model on every call."""

    # ── class-level shared resources (loaded once) ──
    _tts_model = None
    _voice_state = None
    _recognizer = None
    _openai_client: OpenAI | None = None

    @classmethod
    def load_shared_resources(cls):
        """Call once at server startup to pre-load heavy models."""
        cls._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logger.info("OpenAI client created.")

        logger.info("Loading TTS model (one-time) ...")
        cls._tts_model = TTSModel.load_model()
        cls._voice_state = cls._tts_model.get_state_for_audio_prompt("alba")
        logger.info("TTS model loaded.")

        cls._recognizer = Recognizer()
        logger.info("All shared resources loaded.")

    # ────────────────────────────────────────────────────────────
    # Instance
    # ────────────────────────────────────────────────────────────

    def __init__(self, response_queue, loop, lesson_context: str = ""):
        self.response_queue = response_queue
        self.loop = loop

        # ── RAG ──
        self.rag = LessonRAG(self._openai_client)
        self._lesson_ready = False

        if lesson_context:
            self._init_lesson(lesson_context)
        else:
            self.context = FALLBACK_SYSTEM_PROMPT

        self.messages: list[dict] = [{"role": "system", "content": self.context}]

        # ── Interrupt / barge-in state ──
        self._interrupted = threading.Event()
        self._current_ai_text: str = ""  # full reply being spoken
        self._spoken_sentences: list[str] = []  # sentences already sent as audio

        # ── Session state ──
        self._processing = False
        self._first_turn = True
        self._processor = RealtimeAudioProcessor(
            on_utterance=self._on_utterance_detected,
        )

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def process_audio_chunk(self, chunk_bytes: bytes):
        """Called by the gRPC servicer when new WebM/Opus data arrives."""
        if self._first_turn and self._lesson_ready:
            self._first_turn = False
            threading.Thread(target=self._send_opening_greeting, daemon=True).start()
        self._processor.feed(chunk_bytes)

    def handle_interrupt(self):
        """Called when the student starts speaking while AI is talking."""
        self._interrupted.set()
        logger.info("Interrupt flag set — TTS will stop after current sentence")

    def close(self):
        """Clean up ffmpeg and RAG resources."""
        self._processor.close()
        self.rag.close()

    # ────────────────────────────────────────────────────────────
    # Conversation history compression
    # ────────────────────────────────────────────────────────────

    # When the message list exceeds this count, older turns are
    # summarised into a single system message to keep the context
    # window efficient while preserving continuity.
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
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
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

    # ────────────────────────────────────────────────────────────
    # Lesson init
    # ────────────────────────────────────────────────────────────

    def _init_lesson(self, lesson_context: str):
        try:
            self.rag.ingest(lesson_context)
            self.context = self.rag.build_system_prompt()
            self._lesson_ready = True
            logger.info(
                "Lesson RAG ready: %d chunks indexed for %r",
                self.rag.chunk_count,
                self.rag.lesson_title,
            )
        except Exception as e:
            logger.error("Failed to initialise lesson RAG: %s", e, exc_info=True)
            self.context = FALLBACK_SYSTEM_PROMPT

    # ────────────────────────────────────────────────────────────
    # STT — Whisper via OpenAI API
    # ────────────────────────────────────────────────────────────

    def _transcribe(self, pcm_data: bytes) -> str:
        """Transcribe raw 16 kHz mono PCM using OpenAI Whisper API."""
        import numpy as np

        # Convert PCM bytes → WAV in memory for the API
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, TARGET_SAMPLE_RATE, samples)
        buf.seek(0)
        buf.name = "speech.wav"  # API needs a filename hint

        try:
            result = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
            )
            text = result.text.strip()
            logger.info("Whisper STT: %s", text)
            return text
        except Exception as e:
            logger.warning("Whisper STT failed: %s", e)
            return ""

    # ────────────────────────────────────────────────────────────
    # TTS helper (unchanged engine, but per-sentence now)
    # ────────────────────────────────────────────────────────────

    def _synthesise_sentence(self, sentence: str) -> bytes:
        """Return WAV bytes for a single sentence."""
        audio_tensor = self._tts_model.generate_audio(self._voice_state, sentence)
        buf = io.BytesIO()
        audio_np = (
            audio_tensor.cpu().numpy()
            if torch.is_tensor(audio_tensor)
            else audio_tensor
        )
        scipy.io.wavfile.write(buf, self._tts_model.sample_rate, audio_np)
        return buf.getvalue()

    def _send_audio(self, wav_bytes: bytes, ai_text: str = ""):
        """Queue WAV audio + metadata for the gRPC servicer to stream."""
        # Send the text label first so the servicer can attach it
        if ai_text:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait, ("ai_text", ai_text)
            )

        chunk_size = 32768
        for i in range(0, len(wav_bytes), chunk_size):
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                ("audio", wav_bytes[i : i + chunk_size]),
            )
        self.loop.call_soon_threadsafe(self.response_queue.put_nowait, ("end", None))

    # ────────────────────────────────────────────────────────────
    # LLM streaming + per-sentence TTS
    # ────────────────────────────────────────────────────────────

    def _stream_llm_and_speak(self, extra_system_msg: str = ""):
        """Stream the LLM response, split by sentence, and TTS each one.

        Returns the full reply text (may be truncated if interrupted).
        """
        if extra_system_msg:
            self.messages.append({"role": "system", "content": extra_system_msg})

        self._interrupted.clear()
        self._spoken_sentences = []
        self._current_ai_text = ""

        # ── Stream LLM tokens ──
        stream = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            stream=True,
        )

        buffer = ""  # accumulates tokens until a sentence boundary
        full_reply = ""  # everything the LLM produced

        for chunk in stream:
            delta = chunk.choices[0].delta
            if not delta.content:
                continue

            token = delta.content
            buffer += token
            full_reply += token

            # Try to extract complete sentences from the buffer
            sentences = _split_sentences(buffer)

            if len(sentences) > 1:
                # All but the last are complete sentences
                for sent in sentences[:-1]:
                    if self._interrupted.is_set():
                        logger.info("Interrupted — stopping TTS mid-stream")
                        break

                    logger.info("TTS sentence: %s", sent)
                    wav = self._synthesise_sentence(sent)
                    self._send_audio(wav, ai_text=sent)
                    self._spoken_sentences.append(sent)

                if self._interrupted.is_set():
                    break

                # Keep the incomplete tail
                buffer = sentences[-1]
            # else: not enough for a split yet — keep accumulating

        # ── Flush remaining buffer (last sentence) ──
        if buffer.strip() and not self._interrupted.is_set():
            logger.info("TTS sentence (final): %s", buffer.strip())
            wav = self._synthesise_sentence(buffer.strip())
            self._send_audio(wav, ai_text=buffer.strip())
            self._spoken_sentences.append(buffer.strip())

        self._current_ai_text = full_reply
        return full_reply

    # ────────────────────────────────────────────────────────────
    # Opening greeting
    # ────────────────────────────────────────────────────────────

    def _send_opening_greeting(self):
        if self._processing:
            return
        self._processing = True
        try:
            greeting_messages = [
                {"role": "system", "content": self.context},
                {"role": "user", "content": OPENING_GREETING_INSTRUCTION},
            ]
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=greeting_messages
            )
            greeting = response.choices[0].message.content
            logger.info("Opening greeting: %s", greeting)

            self.messages.append({"role": "assistant", "content": greeting})

            # Speak sentence by sentence
            for sent in _split_sentences(greeting):
                if self._interrupted.is_set():
                    break
                wav = self._synthesise_sentence(sent)
                self._send_audio(wav, ai_text=sent)
                self._spoken_sentences.append(sent)

        except Exception as e:
            logger.error("Error sending opening greeting: %s", e, exc_info=True)
        finally:
            self._processing = False

    # ────────────────────────────────────────────────────────────
    # Turn handling
    # ────────────────────────────────────────────────────────────

    def _on_utterance_detected(self, pcm_data: bytes):
        if self._processing:
            logger.info("Still processing previous turn, skipping utterance")
            return
        self._processing = True
        threading.Thread(
            target=self._handle_turn, args=(pcm_data,), daemon=True
        ).start()

    def _handle_turn(self, pcm_data: bytes):
        try:
            duration = len(pcm_data) / (TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
            logger.info("Processing speech turn (%.1fs of PCM) ...", duration)

            # ── 1. STT (Whisper) ──
            text = self._transcribe(pcm_data)
            if not text or text.strip().lower() in ("quit", "exit", "stop"):
                return

            # ── 1b. Build interrupt context if the student interrupted ──
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

            # ── 2. RAG retrieval ──
            rag_context = ""
            if self._lesson_ready:
                rag_context = self.rag.build_retrieval_context(text)
                if rag_context:
                    logger.info("RAG injected %d chars of context", len(rag_context))

            # ── 3. Build messages & stream LLM + TTS ──
            if rag_context:
                self.messages.append({"role": "system", "content": rag_context})
            if interrupt_note:
                self.messages.append({"role": "system", "content": interrupt_note})

            self.messages.append({"role": "user", "content": text})

            full_reply = self._stream_llm_and_speak()
            logger.info("AI Reply: %s", full_reply[:200])
            self.messages.append({"role": "assistant", "content": full_reply})

            # ── Compress older history if it's getting long ──
            self._compress_history()

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
        finally:
            self._processing = False
