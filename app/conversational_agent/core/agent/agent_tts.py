import io
import logging
import queue
import re
import threading
import time

import numpy as np
import scipy.io.wavfile
import torch

from core.config import RUNTIME_CONFIG
from core.prompts import LESSON_END_TOKEN
from core.protocol import MessageType, QueueMessage
from core.tempo import apply_tempo_shaping
from core.utils import (
    detect_language,
    has_trailing_clause_boundary,
    split_clauses,
)

logger = logging.getLogger(__name__)

_POSITIVE_TTS_HINT_RE = re.compile(
    r"\b(great|excellent|well done|nice work|good job|awesome|perfect)\b",
    re.IGNORECASE,
)


class AgentTTSMixin:
    """Provides TTS, audio streaming, and LLM text generation/speaking components
    for the ConversationalAgent."""

    def _get_tts_sample_rate(self, lang: str) -> int:
        """Return the sample rate for the TTS engine used for *lang*."""
        if lang == "ru" and self._resources.ru_tts_model is not None:
            return self._resources.ru_sample_rate
        return self._resources.tts_model.sample_rate

    def _make_silence_wav(self, duration_ms: int, lang: str = "en") -> bytes:
        """Return WAV bytes containing *duration_ms* of silence."""
        sample_rate = self._get_tts_sample_rate(lang)
        n_samples = int(sample_rate * duration_ms / 1000)
        silence = np.zeros(n_samples, dtype=np.float32)
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sample_rate, silence)
        return buf.getvalue()

    def _emotion_style_tts_text(self, sentence: str, lang: str) -> str:
        """Apply lightweight prosody hints for English speech output only."""
        if lang != "en":
            return sentence

        text = sentence.strip()
        if not text:
            return sentence

        # Apply tempo prosody shaping before emotion markup
        if RUNTIME_CONFIG.tempo.adapt_enabled:
            text = apply_tempo_shaping(
                text, getattr(self, "_current_tempo_hint", "NORMAL")
            )

        if not RUNTIME_CONFIG.tts.enable_emotion:
            return text

        strength = RUNTIME_CONFIG.tts.emotion_strength
        extra_exclamations = 1 + int(strength * 2)

        if _POSITIVE_TTS_HINT_RE.search(text) and text.endswith("."):
            return text[:-1] + ("!" * extra_exclamations)

        if text.endswith("!"):
            return re.sub(r"!+$", "!" * extra_exclamations, text)

        return text

    def _synthesise_sentence(self, sentence: str) -> bytes:
        """Return WAV bytes for a single sentence (English or Russian)."""
        lang = detect_language(sentence)
        if lang == "ru" and self._resources.ru_tts_model is not None:
            audio_tensor = self._resources.ru_tts_model.apply_tts(
                text=sentence,
                speaker=self._resources.ru_speaker,
                sample_rate=self._resources.ru_sample_rate,
            )
            sample_rate = self._resources.ru_sample_rate
        else:
            tts_text = self._emotion_style_tts_text(sentence, lang)
            audio_tensor = self._resources.tts_model.generate_audio(
                self._resources.voice_state, tts_text
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

    def _send_audio(self, wav_bytes: bytes, ai_text: str = ""):
        """Queue WAV audio + metadata for the gRPC servicer to stream."""
        if self._interrupted.is_set():
            return

        # Send the text label first so the servicer can attach it
        if ai_text:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AI_TEXT, ai_text),
            )

        chunk_size = 32768
        for i in range(0, len(wav_bytes), chunk_size):
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(MessageType.AUDIO, wav_bytes[i : i + chunk_size]),
            )
        self.loop.call_soon_threadsafe(
            self.response_queue.put_nowait, QueueMessage(MessageType.END)
        )

    def _speak_direct_text(self, text: str) -> str:
        """Speak a deterministic text response without a new LLM call."""
        self._interrupted.clear()
        self._spoken_sentences = []
        for clause in split_clauses(text):
            if self._interrupted.is_set():
                break
            wav = self._synthesise_sentence(clause)
            self._send_audio(wav, ai_text=clause)
            self._spoken_sentences.append(clause)
        spoken = " ".join(self._spoken_sentences) if self._spoken_sentences else text
        self._current_ai_text = spoken
        return spoken

    def _stream_and_speak(self, messages: list[dict]) -> str:
        """Stream the LLM for *messages*, split into TTS clauses, return spoken text."""
        self._interrupted.clear()
        self._spoken_sentences = []
        self._current_ai_text = ""
        self._spoken_schedule = []
        self._audio_playhead = 0.0

        if RUNTIME_CONFIG.tempo.adapt_enabled:
            self.loop.call_soon_threadsafe(
                self.response_queue.put_nowait,
                QueueMessage(
                    MessageType.TEMPO_HINT,
                    getattr(self, "_current_tempo_hint", "NORMAL"),
                ),
            )

        # ── TTS worker (consumer) ──
        sentence_q: queue.Queue[str | None] = queue.Queue()

        def _synthesise_and_send(clause: str):
            logger.info("TTS clause: %s", clause)
            wav = self._synthesise_sentence(clause)

            now = time.monotonic()
            if self._audio_playhead < now:
                self._audio_playhead = now

            start_time = self._audio_playhead
            duration_sec = 0.0
            try:
                import io, wave

                with wave.open(io.BytesIO(wav), "rb") as w:
                    duration_sec = w.getnframes() / float(w.getframerate())
            except Exception:
                duration_sec = max(0.5, len(clause.split()) / 2.5)

            end_time = start_time + duration_sec
            self._audio_playhead = end_time

            self._spoken_schedule.append(
                {"start": start_time, "end": end_time, "clause": clause}
            )

            self._send_audio(wav, ai_text=clause)
            self._spoken_sentences.append(clause)

        def _tts_worker():
            pause_ms = RUNTIME_CONFIG.tempo.inter_sentence_pause_ms
            prev_clause: str | None = None
            while True:
                clause = sentence_q.get()
                if clause is None or self._interrupted.is_set():
                    break
                if (
                    prev_clause is not None
                    and pause_ms > 0
                    and not self._interrupted.is_set()
                ):
                    silence_lang = detect_language(clause)
                    silence_wav = self._make_silence_wav(pause_ms, silence_lang)
                    now = time.monotonic()
                    if self._audio_playhead < now:
                        self._audio_playhead = now
                    self._audio_playhead += pause_ms / 1000.0
                    self._send_audio(silence_wav)
                _synthesise_and_send(clause)
                prev_clause = clause

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        # ── Stream LLM tokens (producer) ──
        stream = self._resources.openai_client.chat.completions.create(
            model=RUNTIME_CONFIG.llm.model,
            messages=messages,
            stream=True,
            temperature=RUNTIME_CONFIG.llm.temperature,
            top_p=RUNTIME_CONFIG.llm.top_p,
            presence_penalty=RUNTIME_CONFIG.llm.presence_penalty,
            frequency_penalty=RUNTIME_CONFIG.llm.frequency_penalty,
            max_tokens=RUNTIME_CONFIG.llm.max_tokens,
        )

        buffer = ""
        full_reply = ""
        lesson_end_requested = False

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

                if LESSON_END_TOKEN in full_reply or LESSON_END_TOKEN in buffer:
                    lesson_end_requested = True
                    full_reply = full_reply.replace(LESSON_END_TOKEN, "")
                    buffer = buffer.replace(LESSON_END_TOKEN, "")

                if has_trailing_clause_boundary(buffer):
                    for cl in split_clauses(buffer):
                        if self._interrupted.is_set():
                            break
                        sentence_q.put(cl)
                    buffer = ""
                    continue

                clauses = split_clauses(buffer)
                if len(clauses) > 1:
                    for cl in clauses[:-1]:
                        if self._interrupted.is_set():
                            break
                        sentence_q.put(cl)
                    if self._interrupted.is_set():
                        break
                    buffer = clauses[-1]
        finally:
            try:
                stream.close()
            except Exception:
                pass

        if buffer.strip() and not self._interrupted.is_set():
            sentence_q.put(buffer.strip())

        sentence_q.put(None)
        worker.join(timeout=30)

        if lesson_end_requested:
            self._check_lesson_end()

        self._current_ai_text = full_reply
        spoken_text = " ".join(self._spoken_sentences)
        return spoken_text if self._interrupted.is_set() else full_reply

    def _stream_llm_and_speak(self, extra_system_msg: str = "") -> str:
        """Prepare the current turn’s message list and stream LLM + TTS."""
        if extra_system_msg:
            self.messages.append({"role": "system", "content": extra_system_msg})
        return self._stream_and_speak(self.messages)
