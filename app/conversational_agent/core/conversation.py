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
import queue
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

# ─── Text splitting ──────────────────────────────────────────────────
# _SENTENCE_RE  — canonical sentence boundaries (. ! ?)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# _CLAUSE_RE — finer-grained: also splits on ; : , and em-dash so
# that long sentences are broken into shorter TTS chunks, reducing
# first-audio-byte latency.
_CLAUSE_RE = re.compile(r"(?<=[.!?;:,\u2014])\s+")

# Minimum character length for a clause to be spoken on its own.
# Shorter fragments are accumulated into the next clause.
_MIN_CLAUSE_LEN = 12


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences at . ! ? boundaries."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_clauses(text: str) -> list[str]:
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


# ─── Language detection ──────────────────────────────────────────────
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _detect_language(text: str) -> str:
    """Return ``'ru'`` if *text* is predominantly Cyrillic, else ``'en'``."""
    if not text:
        return "en"
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return "en"
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    return "ru" if cyrillic_count > len(alpha_chars) / 2 else "en"


class ConversationalAgent:
    """Per-session agent.  Receives pre-loaded shared resources to avoid
    re-loading the TTS model on every call."""

    # ── class-level shared resources (loaded once) ──
    _tts_model = None
    _voice_state = None
    _ru_tts_model = None
    _ru_speaker: str | None = None
    _ru_sample_rate: int | None = None
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

        logger.info("Loading Russian TTS model \u2026")
        try:
            ru_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v3_1_ru",
                trust_repo=True,
            )
            cls._ru_tts_model = ru_model
            cls._ru_speaker = "baya"
            cls._ru_sample_rate = 24000
            logger.info("Russian TTS model loaded.")
        except Exception as e:
            logger.warning("Failed to load Russian TTS model: %s", e)
            cls._ru_tts_model = None

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
        self._processing_lock = threading.Lock()
        self._pending_utterance: bytes | None = None  # queued when user interrupts
        self._first_turn = True

        # ── Eager STT — start transcribing while silence is still
        #    being measured so that the text is ready by the time the
        #    utterance boundary fires. ──
        self._eager_stt_result: str | None = None
        self._eager_stt_pcm_len: int = 0
        self._eager_stt_done = threading.Event()
        self._eager_stt_lock = threading.Lock()

        self._processor = RealtimeAudioProcessor(
            on_utterance=self._on_utterance_detected,
            on_speech_paused=self._on_speech_paused,
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
        """Called when the student starts speaking while AI is talking.

        Sets the interrupt flag, drains any queued audio so the client
        stops hearing stale speech, and sends an explicit ``interrupt``
        signal so the frontend can halt playback immediately.
        """
        self._interrupted.set()

        def _drain_and_signal():
            drained = 0
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                    drained += 1
                except Exception:
                    break
            self.response_queue.put_nowait(("signal", "interrupt"))
            logger.info(
                "Interrupt flag set — drained %d queued messages, sent interrupt signal",
                drained,
            )

        self.loop.call_soon_threadsafe(_drain_and_signal)

    def close(self):
        """Clean up ffmpeg and RAG resources."""
        self._processor.close()
        self.rag.close()

    # ────────────────────────────────────────────────────────────
    # Eager STT — transcribe during silence detection
    # ────────────────────────────────────────────────────────────

    def _on_speech_paused(self, pcm_data: bytes):
        """Called by VAD at the first silence frame after speech.

        Kicks off a background Whisper API call so that the
        transcription overlaps with the remaining silence-detection
        wait (~0.5-1 s).  When the full utterance boundary fires,
        ``_handle_turn`` checks whether the eager result is already
        available and skips a second Whisper call.
        """
        with self._eager_stt_lock:
            self._eager_stt_result = None
            self._eager_stt_pcm_len = len(pcm_data)
            self._eager_stt_done.clear()

        def _background():
            text = self._transcribe(pcm_data)
            with self._eager_stt_lock:
                self._eager_stt_result = text
                self._eager_stt_pcm_len = len(pcm_data)
            self._eager_stt_done.set()
            logger.info("Eager STT finished: %s", text[:120] if text else "(empty)")

        threading.Thread(target=_background, daemon=True).start()

    def _get_eager_or_transcribe(self, pcm_data: bytes) -> str:
        """Return transcription text, preferring the eager result.

        If the eager background transcription already finished (or
        finishes within a short wait) and it covered at least 70 % of
        the final utterance length, use that result directly — saving
        a full Whisper round-trip.  Otherwise fall back to a fresh
        ``_transcribe`` call.
        """
        # Wait briefly — the eager call started ~silence_sec ago, so
        # it should be nearly done (or already done).
        if self._eager_stt_done.wait(timeout=2.0):
            with self._eager_stt_lock:
                # Use if the eager audio is ≥ 70 % of the final PCM
                if self._eager_stt_pcm_len >= len(pcm_data) * 0.7:
                    text = self._eager_stt_result or ""
                    logger.info(
                        "Using eager STT result (covered %.0f%% of utterance)",
                        self._eager_stt_pcm_len / len(pcm_data) * 100,
                    )
                    return text

        logger.info("Eager STT miss — running full transcription")
        return self._transcribe(pcm_data)

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
        """Transcribe raw 16 kHz mono PCM using OpenAI Whisper API.

        Speed optimisations:
          - Trim leading/trailing silence so Whisper processes less audio.
          - Encode as OGG/Opus (~10× smaller than WAV) for faster upload.
          - Use ``response_format="text"`` to skip JSON overhead.
        """
        import numpy as np
        import subprocess as _sp

        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # ── Trim silence (threshold: ~1 % of int16 range) ──
        threshold = 500
        above = np.nonzero(np.abs(samples) > threshold)[0]
        if len(above) == 0:
            logger.info("Whisper STT: audio is silence, skipping")
            return ""
        # Keep a small margin (160 samples = 10 ms @ 16 kHz)
        margin = 160
        start = max(0, above[0] - margin)
        end = min(len(samples), above[-1] + margin)
        samples = samples[start:end]

        # ── Encode to OGG/Opus via ffmpeg (in-memory) ──
        wav_buf = io.BytesIO()
        scipy.io.wavfile.write(wav_buf, TARGET_SAMPLE_RATE, samples)
        wav_bytes = wav_buf.getvalue()

        try:
            proc = _sp.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "wav",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                input=wav_bytes,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout:
                upload_buf = io.BytesIO(proc.stdout)
                upload_buf.name = "speech.ogg"
            else:
                # Fallback to WAV if ffmpeg encoding fails
                logger.warning("OGG encoding failed, falling back to WAV")
                upload_buf = io.BytesIO(wav_bytes)
                upload_buf.name = "speech.wav"
        except Exception:
            upload_buf = io.BytesIO(wav_bytes)
            upload_buf.name = "speech.wav"

        try:
            result = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=upload_buf,
                response_format="text",
            )
            # With response_format="text", result is a plain string
            text = result.strip() if isinstance(result, str) else result.text.strip()
            logger.info("Whisper STT: %s", text)
            return text
        except Exception as e:
            logger.warning("Whisper STT failed: %s", e)
            return ""

    # ────────────────────────────────────────────────────────────
    # TTS helper (unchanged engine, but per-sentence now)
    # ────────────────────────────────────────────────────────────

    def _synthesise_sentence(self, sentence: str) -> bytes:
        """Return WAV bytes for a single sentence (English or Russian)."""
        lang = _detect_language(sentence)
        if lang == "ru" and self._ru_tts_model is not None:
            audio_tensor = self._ru_tts_model.apply_tts(
                text=sentence,
                speaker=self._ru_speaker,
                sample_rate=self._ru_sample_rate,
            )
            sample_rate = self._ru_sample_rate
        else:
            audio_tensor = self._tts_model.generate_audio(self._voice_state, sentence)
            sample_rate = self._tts_model.sample_rate

        buf = io.BytesIO()
        audio_np = (
            audio_tensor.cpu().numpy()
            if torch.is_tensor(audio_tensor)
            else audio_tensor
        )
        scipy.io.wavfile.write(buf, sample_rate, audio_np)
        return buf.getvalue()

    def _send_audio(self, wav_bytes: bytes, ai_text: str = ""):
        """Queue WAV audio + metadata for the gRPC servicer to stream.

        Delivers the complete sentence audio to avoid cutting off
        mid-word.  The interrupt flag is checked only before starting
        a new sentence — the frontend handles smooth fade-out.
        """
        if self._interrupted.is_set():
            return

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
        """Stream the LLM response, split into clauses, and TTS each one.

        Uses a **producer / consumer** pattern: the main thread streams
        LLM tokens and pushes complete clauses into a queue; a worker
        thread synthesises and sends TTS audio in parallel.  This
        overlaps LLM generation with TTS, reducing perceived latency.

        Returns the *spoken* reply text (truncated at the interrupt
        point when the student barges in).
        """
        if extra_system_msg:
            self.messages.append({"role": "system", "content": extra_system_msg})

        self._interrupted.clear()
        self._spoken_sentences = []
        self._current_ai_text = ""

        # ── TTS worker (consumer) ──
        sentence_q: queue.Queue[str | None] = queue.Queue()

        def _tts_worker():
            """Consume clauses from *sentence_q* and synthesise + send."""
            while True:
                clause = sentence_q.get()
                if clause is None:  # poison pill
                    break
                if self._interrupted.is_set():
                    break
                logger.info("TTS clause: %s", clause)
                wav = self._synthesise_sentence(clause)
                self._send_audio(wav, ai_text=clause)
                self._spoken_sentences.append(clause)

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        # ── Stream LLM tokens (producer) ──
        stream = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
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
                clauses = _split_clauses(buffer)

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
            # Close the LLM stream immediately on interrupt (P1) to
            # stop wasting tokens / network.
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

    # ────────────────────────────────────────────────────────────
    # Opening greeting
    # ────────────────────────────────────────────────────────────

    def _send_opening_greeting(self):
        if self._processing:
            return
        self._processing = True
        try:
            self._interrupted.clear()
            self._spoken_sentences = []

            greeting_messages = [
                {"role": "system", "content": self.context},
                {"role": "user", "content": OPENING_GREETING_INSTRUCTION},
            ]

            # Stream the greeting so the first clause is spoken sooner.
            stream = self._openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=greeting_messages, stream=True
            )

            buffer = ""
            full_greeting = ""

            try:
                for chunk in stream:
                    if self._interrupted.is_set():
                        break
                    delta = chunk.choices[0].delta
                    if not delta.content:
                        continue
                    token = delta.content
                    buffer += token
                    full_greeting += token

                    clauses = _split_clauses(buffer)
                    if len(clauses) > 1:
                        for cl in clauses[:-1]:
                            if self._interrupted.is_set():
                                break
                            wav = self._synthesise_sentence(cl)
                            self._send_audio(wav, ai_text=cl)
                            self._spoken_sentences.append(cl)
                        if self._interrupted.is_set():
                            break
                        buffer = clauses[-1]
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

            # Flush remaining buffer
            if buffer.strip() and not self._interrupted.is_set():
                wav = self._synthesise_sentence(buffer.strip())
                self._send_audio(wav, ai_text=buffer.strip())
                self._spoken_sentences.append(buffer.strip())

            logger.info("Opening greeting: %s", full_greeting[:200])

            # Only store the portion that was actually spoken
            spoken = (
                " ".join(self._spoken_sentences)
                if self._spoken_sentences
                else full_greeting
            )
            self.messages.append({"role": "assistant", "content": spoken})

        except Exception as e:
            logger.error("Error sending opening greeting: %s", e, exc_info=True)
        finally:
            self._processing = False
            # Check if there's a pending utterance from an interrupt
            self._process_pending_utterance()

    # ────────────────────────────────────────────────────────────
    # Turn handling
    # ────────────────────────────────────────────────────────────

    def _on_utterance_detected(self, pcm_data: bytes):
        with self._processing_lock:
            if self._processing:
                # User spoke while AI is still responding — trigger interrupt
                # and queue this utterance so it's processed once the current
                # turn finishes.
                logger.info(
                    "User spoke during AI turn — triggering interrupt, "
                    "queuing utterance (%d bytes)",
                    len(pcm_data),
                )
                self._pending_utterance = pcm_data
                self.handle_interrupt()
                return
            self._processing = True

        threading.Thread(
            target=self._handle_turn, args=(pcm_data,), daemon=True
        ).start()

    def _process_pending_utterance(self):
        """If a student utterance arrived during the previous AI turn,
        process it now."""
        pcm = self._pending_utterance
        self._pending_utterance = None
        if pcm is not None:
            logger.info("Processing pending interrupt utterance (%d bytes)", len(pcm))
            # Reset processing flag so _on_utterance_detected can acquire it
            self._on_utterance_detected(pcm)

    def _handle_turn(self, pcm_data: bytes):
        try:
            duration = len(pcm_data) / (TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE)
            logger.info("Processing speech turn (%.1fs of PCM) ...", duration)

            # ── 1. STT — try eager result first, fall back to fresh call ──
            text = self._get_eager_or_transcribe(pcm_data)
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

            spoken_reply = self._stream_llm_and_speak()
            logger.info("AI Reply (spoken): %s", spoken_reply[:200])

            # Only store the portion the student actually heard.
            # If interrupted, spoken_reply is the truncated prefix.
            self.messages.append({"role": "assistant", "content": spoken_reply})

            # ── Compress older history if it's getting long ──
            self._compress_history()

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
        finally:
            with self._processing_lock:
                self._processing = False
            # If the student spoke while we were busy, handle it now
            self._process_pending_utterance()
