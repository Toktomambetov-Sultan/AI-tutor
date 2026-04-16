"""
Real-time audio processor with Silero VAD, adaptive silence detection,
and streaming Vosk STT.

Spawns a persistent ffmpeg process that decodes a continuous WebM/Opus
stream into 16 kHz mono PCM.  A reader thread runs Silero VAD on the
decoded PCM, feeds chunks to a Vosk recognizer for real-time
transcription, and fires a callback with the recognised text when an
utterance boundary (speech → silence) is detected.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Audio constants ─────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# ─── Silero VAD (loaded once per process) ────────────────────────────
_silero_model = None
_silero_utils = None
_silero_lock = threading.Lock()


def _ensure_silero_model():
    """Lazy-load Silero VAD from torch.hub (cached after first call)."""
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return
    with _silero_lock:
        if _silero_model is not None:
            return
        logger.info("Loading Silero VAD model …")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _silero_model = model
        _silero_utils = utils
        logger.info("Silero VAD model loaded.")


class RealtimeAudioProcessor:
    """
    Spawns a persistent ffmpeg process that decodes a continuous WebM/Opus
    stream into 16 kHz mono PCM in real-time.  A reader thread runs
    **Silero VAD** on the decoded PCM, simultaneously feeding each chunk
    to a **Vosk** recognizer for streaming speech-to-text, and fires a
    callback with the transcribed text when an utterance boundary
    (speech → silence) is detected.

    Adaptive silence detection: the processor tracks the student's
    speaking cadence and adjusts ``silence_sec`` between a floor and
    ceiling so that fast speakers get shorter pauses and slower
    speakers get more generous ones.
    """

    def __init__(
        self,
        on_utterance: "callable",
        sample_rate: int = TARGET_SAMPLE_RATE,
        silence_sec: float = 1.0,
        min_silence_sec: float = 0.5,
        max_silence_sec: float = 1.2,
        vad_threshold: float = 0.45,
        min_speech_sec: float = 0.5,
        recognizer=None,
    ):
        _ensure_silero_model()

        self.on_utterance = on_utterance
        self._recognizer = recognizer
        self.sample_rate = sample_rate

        # ── Adaptive silence parameters ──
        self.silence_sec = silence_sec  # current (adaptive) pause duration
        self._min_silence_sec = min_silence_sec
        self._max_silence_sec = max_silence_sec
        self._recent_speech_durations: list[float] = []  # last N utterances in secs
        self._ADAPT_WINDOW = 5  # look at last 5 utterances

        self.vad_threshold = vad_threshold
        self.min_speech_bytes = int(min_speech_sec * sample_rate * BYTES_PER_SAMPLE)

        self._alive = True
        self._pcm_buf = bytearray()
        self._is_speaking = False
        self._silence_frames = 0

        # Persistent ffmpeg
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
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

    def _adapt_silence_threshold(self, utterance_bytes: int):
        """Track utterance lengths and adapt the silence-to-end window.

        Fast speakers → shorter required silence → quicker turn-taking.
        Slow speakers → longer required silence → avoids cutting them off.
        """
        dur = utterance_bytes / (self.sample_rate * BYTES_PER_SAMPLE)
        self._recent_speech_durations.append(dur)
        if len(self._recent_speech_durations) > self._ADAPT_WINDOW:
            self._recent_speech_durations.pop(0)

        if len(self._recent_speech_durations) < 2:
            return  # not enough data yet

        avg_dur = sum(self._recent_speech_durations) / len(
            self._recent_speech_durations
        )

        # Heuristic: shorter average speech → shorter silence needed
        # Map avg speech [1s..8s] → silence [min..max]
        ratio = max(0.0, min(1.0, (avg_dur - 1.0) / 7.0))
        new_silence = self._min_silence_sec + ratio * (
            self._max_silence_sec - self._min_silence_sec
        )
        if abs(new_silence - self.silence_sec) > 0.05:
            logger.info(
                "Adaptive silence: avg_speech=%.1fs → silence_sec %.2f→%.2f",
                avg_dur,
                self.silence_sec,
                new_silence,
            )
            self.silence_sec = new_silence

    def _run_silero_vad(self, pcm_chunk: bytes) -> float:
        """Return Silero VAD speech probability for a 512-sample PCM chunk."""
        arr = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            prob = _silero_model(tensor, self.sample_rate).item()
        return prob

    def _read_loop(self):
        """Read decoded PCM from ffmpeg stdout and run Silero VAD."""
        CHUNK_SAMPLES = 512  # 32 ms at 16 kHz  (Silero requires exactly 512 @ 16k)
        CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
        CHUNK_DURATION_SEC = CHUNK_SAMPLES / self.sample_rate  # 0.032 s
        frames_read = 0
        read_buf = bytearray()  # buffer for partial stdout reads

        logger.info(
            "PCM reader loop started (Silero VAD, threshold=%.2f, silence=%.1fs)",
            self.vad_threshold,
            self.silence_sec,
        )

        while self._alive:
            try:
                raw = self._proc.stdout.read(CHUNK_BYTES)
                if not raw:
                    logger.info("ffmpeg stdout EOF after %d frames", frames_read)
                    break

                read_buf.extend(raw)
                # Process only complete 512-sample chunks
                if len(read_buf) < CHUNK_BYTES:
                    continue
                data = bytes(read_buf[:CHUNK_BYTES])
                del read_buf[:CHUNK_BYTES]

                frames_read += 1

                # ── Silero VAD ──
                speech_prob = self._run_silero_vad(data)

                # Adaptive: recalculate silence frames based on current silence_sec
                silence_frames_needed = int(self.silence_sec / CHUNK_DURATION_SEC)

                if frames_read % 50 == 0:
                    logger.debug(
                        "frame #%d: prob=%.3f speaking=%s sil_frames=%d/%d buf=%d",
                        frames_read,
                        speech_prob,
                        self._is_speaking,
                        self._silence_frames,
                        silence_frames_needed,
                        len(self._pcm_buf),
                    )

                if speech_prob >= self.vad_threshold:
                    # ── speech detected ──
                    if not self._is_speaking:
                        logger.info(
                            "VAD: speech START at frame %d (prob=%.3f)",
                            frames_read,
                            speech_prob,
                        )
                    self._is_speaking = True
                    self._silence_frames = 0
                    self._pcm_buf.extend(data)
                    if self._recognizer:
                        self._recognizer.AcceptWaveform(data)
                elif self._is_speaking:
                    # ── silence while we were speaking ──
                    self._pcm_buf.extend(data)
                    if self._recognizer:
                        self._recognizer.AcceptWaveform(data)
                    self._silence_frames += 1

                    if self._silence_frames == 1:
                        logger.info(
                            "VAD: speech→silence at frame %d (prob=%.3f)",
                            frames_read,
                            speech_prob,
                        )

                    if self._silence_frames >= silence_frames_needed:
                        pcm_len = len(self._pcm_buf)
                        self._pcm_buf = bytearray()
                        self._is_speaking = False
                        self._silence_frames = 0

                        if pcm_len >= self.min_speech_bytes:
                            duration = pcm_len / (self.sample_rate * BYTES_PER_SAMPLE)
                            # ── Get transcription from Vosk ──
                            if self._recognizer:
                                result = json.loads(self._recognizer.FinalResult())
                                text = result.get("text", "")
                            else:
                                text = ""

                            logger.info(
                                "VAD: utterance detected (%.1fs of PCM) "
                                "at frame %d — text: %s",
                                duration,
                                frames_read,
                                text[:120] if text else "(empty)",
                            )
                            # ── Adapt silence for next utterance ──
                            self._adapt_silence_threshold(pcm_len)
                            self.on_utterance(text)
                        else:
                            logger.debug("VAD: utterance too short, discarding")
                            # Reset recognizer for discarded utterance
                            if self._recognizer:
                                self._recognizer.FinalResult()
                # else: silence before any speech — ignore

            except Exception as e:
                if self._alive:
                    logger.warning("PCM reader error: %s", e)
                break

        # Reset Silero model state at end of session
        try:
            _silero_model.reset_states()
        except Exception:
            pass

        logger.info(
            "PCM reader loop exited (alive=%s, frames_read=%d)",
            self._alive,
            frames_read,
        )
