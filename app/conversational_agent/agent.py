import os
import logging
import threading
import time
import subprocess
import numpy as np
import torch
import io
from speech_recognition import Recognizer, AudioData
from openai import OpenAI
from pocket_tts import TTSModel
import scipy.io.wavfile
from dotenv import load_dotenv

from rag import LessonRAG

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Constants ───
TARGET_SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM


class RealtimeAudioProcessor:
    """
    Spawns a persistent ffmpeg process that decodes a continuous WebM/Opus
    stream into 16 kHz mono PCM in real-time.  A reader thread performs
    energy-based VAD on the decoded PCM and fires a callback when an
    utterance boundary (speech → silence) is detected.
    """

    def __init__(
        self,
        on_utterance: "callable",
        sample_rate: int = TARGET_SAMPLE_RATE,
        silence_sec: float = 1.2,
        energy_threshold: float = 300,
        min_speech_sec: float = 0.4,
    ):
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate
        self.silence_sec = silence_sec
        self.energy_threshold = energy_threshold
        self.min_speech_bytes = int(min_speech_sec * sample_rate * BYTES_PER_SAMPLE)

        self._alive = True
        self._pcm_buf = bytearray()
        self._is_speaking = False
        self._silence_frames = 0  # count of consecutive silent frames (audio-time)

        # Persistent ffmpeg: reads WebM from stdin, writes raw PCM to stdout.
        # -probesize / -analyzeduration keep startup fast.
        # -flush_packets 1 forces PCM output to be flushed immediately.
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                # Low probe / analyse so first PCM comes fast
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-fflags",
                "+nobuffer+flush_packets",
                "-thread_queue_size",
                "0",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-flush_packets",
                "1",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Log ffmpeg stderr in background so we can see any errors or timing
        self._stderr_thread = threading.Thread(target=self._log_stderr, daemon=True)
        self._stderr_thread.start()

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        logger.info("RealtimeAudioProcessor started (ffmpeg pid=%d)", self._proc.pid)

    # ── public ──

    def feed(self, webm_chunk: bytes):
        """Write a WebM/Opus chunk from the browser into ffmpeg's stdin."""
        if not self._alive:
            return
        try:
            self._proc.stdin.write(webm_chunk)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        """Shut down ffmpeg and reader thread."""
        self._alive = False
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.terminate()
        logger.info("RealtimeAudioProcessor closed")

    # ── internal ──

    def _log_stderr(self):
        """Drain ffmpeg stderr so it doesn't block, and log it."""
        try:
            for line in self._proc.stderr:
                msg = line.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.info("ffmpeg: %s", msg)
        except Exception:
            pass

    def _read_loop(self):
        """Read decoded PCM from ffmpeg stdout and run energy-based VAD."""
        CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz
        CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
        CHUNK_DURATION_SEC = CHUNK_SAMPLES / self.sample_rate  # 0.1 s
        SILENCE_FRAMES_NEEDED = int(self.silence_sec / CHUNK_DURATION_SEC)
        frames_read = 0

        logger.info(
            "PCM reader loop started (silence_threshold=%d frames = %.1fs)",
            SILENCE_FRAMES_NEEDED,
            self.silence_sec,
        )

        while self._alive:
            try:
                data = self._proc.stdout.read(CHUNK_BYTES)
                if not data:
                    logger.info("ffmpeg stdout EOF after %d frames", frames_read)
                    break

                frames_read += 1

                # Energy of this 100 ms frame
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(arr**2))

                # Log periodically for diagnostics
                if frames_read % 50 == 0:
                    logger.debug(
                        "PCM frame #%d: energy=%.0f speaking=%s silence_frames=%d buf=%d",
                        frames_read,
                        energy,
                        self._is_speaking,
                        self._silence_frames,
                        len(self._pcm_buf),
                    )

                if energy > self.energy_threshold:
                    # ── speech detected ──
                    if not self._is_speaking:
                        logger.info(
                            "VAD: speech START at frame %d (energy=%.0f)",
                            frames_read,
                            energy,
                        )
                    self._is_speaking = True
                    self._silence_frames = 0
                    self._pcm_buf.extend(data)
                elif self._is_speaking:
                    # ── silence while we were speaking ──
                    self._pcm_buf.extend(data)
                    self._silence_frames += 1

                    if self._silence_frames == 1:
                        logger.info(
                            "VAD: speech->silence at frame %d (energy=%.0f)",
                            frames_read,
                            energy,
                        )

                    if self._silence_frames >= SILENCE_FRAMES_NEEDED:
                        # Utterance boundary reached (audio-time based)
                        pcm = bytes(self._pcm_buf)
                        self._pcm_buf = bytearray()
                        self._is_speaking = False
                        self._silence_frames = 0

                        if len(pcm) >= self.min_speech_bytes:
                            logger.info(
                                "VAD: utterance detected (%.1fs of PCM) at frame %d",
                                len(pcm) / (self.sample_rate * BYTES_PER_SAMPLE),
                                frames_read,
                            )
                            self.on_utterance(pcm)
                        else:
                            logger.debug("VAD: utterance too short, discarding")
                # else: silence before any speech — ignore

            except Exception as e:
                if self._alive:
                    logger.warning("PCM reader error: %s", e)
                break

        logger.info(
            "PCM reader loop exited (alive=%s, frames_read=%d)",
            self._alive,
            frames_read,
        )


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
            self.context = (
                "You are a helpful AI assistant. Keep answers concise and "
                "natural for speech. Avoid using markdown, symbols, or "
                "formatting that cannot be read aloud easily."
            )

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
            self.context = (
                "You are a helpful AI assistant. Keep answers concise and "
                "natural for speech."
            )

    def _send_opening_greeting(self):
        """Generate and send an opening greeting that introduces the lesson."""
        if self._processing:
            return
        self._processing = True
        try:
            # Ask the LLM to produce a short opening greeting
            greeting_messages = [
                {"role": "system", "content": self.context},
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM: The student has just joined the voice lesson. "
                        "Greet them warmly and introduce today's lesson topic. "
                        "Keep it to 2-3 sentences. Do NOT wait for a response yet.]"
                    ),
                },
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
