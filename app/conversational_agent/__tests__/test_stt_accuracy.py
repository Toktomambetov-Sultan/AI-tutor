"""STT accuracy regression test using fixture audio and reference transcript.

This test is integration-like: it loads a real Vosk model and decodes a real
fixture audio file. It skips automatically when dependencies/models are missing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_overlap_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_count = Counter(ref_tokens)
    hyp_count = Counter(hyp_tokens)
    overlap = sum((ref_count & hyp_count).values())

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@pytest.mark.integration
def test_fixture_audio_transcription_quality():
    root = Path(__file__).resolve().parent
    audio_path = root / "test_audio.mp3"
    transcript_path = root / "audio_text.txt"

    if not audio_path.exists() or not transcript_path.exists():
        pytest.skip("Audio fixtures are missing in __tests__")

    try:
        from pydub import AudioSegment
        from vosk import KaldiRecognizer, Model
    except Exception:
        pytest.skip("pydub/vosk unavailable in test environment")

    model_candidates = [
        Path("/app/models/vosk-model-en-us-0.22-lgraph"),
        Path("/app/models/vosk-model-en-us-0.22"),
        Path("/app/models/vosk-model-small-en-us-0.15"),
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        pytest.skip("No Vosk English model found in expected paths")

    expected_text = transcript_path.read_text(encoding="utf-8").strip()
    if not expected_text:
        pytest.skip("Reference transcript is empty")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pcm_bytes = audio.raw_data

    recognizer = KaldiRecognizer(Model(str(model_path)), 16000)
    if hasattr(recognizer, "SetWords"):
        recognizer.SetWords(True)

    parts: list[str] = []
    chunk_size = 4000
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i : i + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            res = json.loads(recognizer.Result())
            text = (res.get("text") or "").strip()
            if text:
                parts.append(text)

    final_res = json.loads(recognizer.FinalResult())
    final_text = (final_res.get("text") or "").strip()
    if final_text:
        parts.append(final_text)

    predicted_text = " ".join(parts).strip()

    # Non-empty transcript is a hard requirement.
    assert predicted_text, "STT returned empty transcription for fixture audio"

    score = _token_overlap_f1(expected_text, predicted_text)
    # Conservative threshold to avoid flaky failures across CPU/model variants.
    assert score >= 0.45, (
        f"Low STT token-overlap F1: {score:.3f}\n"
        f"EXPECTED: {expected_text}\n"
        f"PREDICTED: {predicted_text}"
    )
