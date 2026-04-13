"""
Real-time audio processor with energy-based Voice Activity Detection (VAD).

Spawns a persistent ffmpeg process that decodes a continuous WebM/Opus
stream into 16 kHz mono PCM.  A reader thread performs energy-based VAD
on the decoded PCM and fires a callback when an utterance boundary
(speech → silence) is detected.
"""

import logging
import subprocess
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ─── Audio constants ─────────────────────────────────────────────────
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

    # ── public ──────────────────────────────────────────────────────

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

    # ── internal ────────────────────────────────────────────────────

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
