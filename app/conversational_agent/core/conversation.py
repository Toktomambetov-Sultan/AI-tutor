"""
Per-session conversational agent.

Orchestrates the full STT → RAG → LLM → TTS pipeline for a single
student session.  Receives pre-loaded shared resources (TTS model,
OpenAI client) to avoid re-loading on every call.
"""

import io
import logging
import os
import threading

import scipy.io.wavfile
import torch
from openai import OpenAI
from pocket_tts import TTSModel
from speech_recognition import AudioData, Recognizer

from core.audio_processor import (
    BYTES_PER_SAMPLE,
    TARGET_SAMPLE_RATE,
    RealtimeAudioProcessor,
)
from conversational_agent.core.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    OPENING_GREETING_INSTRUCTION,
)
from conversational_agent.core.rag import LessonRAG

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Per-session agent. Receives pre-loaded shared resources to avoid
    re-loading the TTS model on every call."""

    # ── class-level shared resources (loaded once) ──
    _tts_model = None
    _voice_state = None
    _recognizer = None
    _openai_client = None

    @classmethod
    def load_shared_resources(cls):
        """Call once at server startup to pre-load heavy models."""
        # 1. OpenAI
        cls._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logger.info("OpenAI client created.")

        # 2. TTS
        logger.info("Loading TTS model (one-time) ...")
        cls._tts_model = TTSModel.load_model()
        cls._voice_state = cls._tts_model.get_state_for_audio_prompt("alba")
        logger.info("TTS model loaded.")

        # 3. STT
        cls._recognizer = Recognizer()
        logger.info("All shared resources loaded.")

    def __init__(self, response_queue, loop, lesson_context: str = ""):
        self.response_queue = response_queue
        self.loop = loop

        # ── RAG setup ──
        self.rag = LessonRAG(self._openai_client)
        self._lesson_ready = False

        if lesson_context:
            self._init_lesson(lesson_context)
        else:
            # Fallback: generic assistant (no lesson materials)
            self.context = FALLBACK_SYSTEM_PROMPT

        self.messages = [{"role": "system", "content": self.context}]

        # Per-session processor (fast — just spawns ffmpeg)
        self._processing = False
        self._first_turn = True  # Flag to generate opening greeting
        self._processor = RealtimeAudioProcessor(
            on_utterance=self._on_utterance_detected,
        )

    def process_audio_chunk(self, chunk_bytes: bytes):
        """Called by server when new WebM/Opus data arrives from the client."""
        # On first audio data, send the opening greeting if lesson is ready
        if self._first_turn and self._lesson_ready:
            self._first_turn = False
            threading.Thread(target=self._send_opening_greeting, daemon=True).start()
        self._processor.feed(chunk_bytes)

    def _init_lesson(self, lesson_context: str):
        """Ingest lesson materials and build the RAG-augmented system prompt."""
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

    def _send_opening_greeting(self):
        """Generate and send an opening greeting that introduces the lesson."""
        if self._processing:
            return
        self._processing = True
        try:
            # Ask the LLM to produce a short opening greeting
            greeting_messages = [
                {"role": "system", "content": self.context},
                {"role": "user", "content": OPENING_GREETING_INSTRUCTION},
            ]
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=greeting_messages
            )
            greeting = response.choices[0].message.content
            logger.info("Opening greeting: %s", greeting)

            # Store in conversation history
            self.messages.append({"role": "assistant", "content": greeting})

            # TTS
            audio_tensor = self._tts_model.generate_audio(self._voice_state, greeting)
            buf = io.BytesIO()
            audio_np = (
                audio_tensor.cpu().numpy()
                if torch.is_tensor(audio_tensor)
                else audio_tensor
            )
            scipy.io.wavfile.write(buf, self._tts_model.sample_rate, audio_np)
            wav_bytes = buf.getvalue()

            chunk_size = 32768
            for i in range(0, len(wav_bytes), chunk_size):
                self.loop.call_soon_threadsafe(
                    self.response_queue.put_nowait,
                    ("audio", wav_bytes[i : i + chunk_size]),
                )
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait, ("end", None)
            )
        except Exception as e:
            logger.error("Error sending opening greeting: %s", e, exc_info=True)
        finally:
            self._processing = False

    def close(self):
        """Clean up the ffmpeg process and RAG resources."""
        self._processor.close()
        self.rag.close()

    def _on_utterance_detected(self, pcm_data: bytes):
        """Called by the VAD reader thread when a complete utterance is ready."""
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

            # ── 1. STT via SpeechRecognition ──
            try:
                audio_data = AudioData(pcm_data, TARGET_SAMPLE_RATE, BYTES_PER_SAMPLE)
                text = self._recognizer.recognize_google(audio_data, language="en-US")
                logger.info("User said: %s", text)
            except Exception as e:
                logger.warning("STT failed: %s", e)
                return

            if not text or text.strip().lower() in ("quit", "exit", "stop"):
                return

            # ── 2. RAG retrieval ──
            rag_context = ""
            if self._lesson_ready:
                rag_context = self.rag.build_retrieval_context(text)
                if rag_context:
                    logger.info("RAG injected %d chars of context", len(rag_context))

            # ── 3. LLM (OpenAI) with RAG-augmented message ──
            if rag_context:
                # Inject retrieved context as a system message before the user's message
                self.messages.append({"role": "system", "content": rag_context})

            self.messages.append({"role": "user", "content": text})
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=self.messages
            )
            reply = response.choices[0].message.content
            logger.info("AI Reply: %s", reply)
            self.messages.append({"role": "assistant", "content": reply})

            # ── 4. TTS (PocketTTS) ──
            logger.info("Generating TTS audio...")
            audio_tensor = self._tts_model.generate_audio(self._voice_state, reply)

            buf = io.BytesIO()
            audio_np = (
                audio_tensor.cpu().numpy()
                if torch.is_tensor(audio_tensor)
                else audio_tensor
            )
            scipy.io.wavfile.write(buf, self._tts_model.sample_rate, audio_np)
            wav_bytes = buf.getvalue()
            logger.info(
                "TTS generated %d bytes of WAV (%.1fs)",
                len(wav_bytes),
                len(wav_bytes) / (self._tts_model.sample_rate * 2),
            )  # approx

            # ── 5. Queue for sending back (thread-safe into asyncio.Queue) ──
            chunk_size = 32768
            for i in range(0, len(wav_bytes), chunk_size):
                self.loop.call_soon_threadsafe(
                    self.response_queue.put_nowait,
                    ("audio", wav_bytes[i : i + chunk_size]),
                )
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait, ("end", None)
            )

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
        finally:
            self._processing = False
