"""
test_streaming.py — Unit tests for the streaming conversation pipeline.

Tests sentence splitting, interrupt handling, TTS queueing, and the
LLM-streaming + sentence-chunked TTS flow WITHOUT requiring actual
models / gRPC / audio hardware.

Run:
    cd /path/to/conversational_agent
    OPENAI_API_KEY=sk-... pytest __tests__/test_streaming.py -v -p no:ethereum
"""

import asyncio
import io
import sys
import threading
import time
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ─────────────────────────────────────────────────────────────────────
# Stub out heavy C-extension / Docker-only dependencies so the module
# can be imported on a dev machine without pocket_tts, grpc stubs, etc.
# ─────────────────────────────────────────────────────────────────────

try:
    import pocket_tts  # noqa: F401
except ImportError:
    _pocket_tts = ModuleType("pocket_tts")
    _pocket_tts.TTSModel = MagicMock()
    sys.modules["pocket_tts"] = _pocket_tts

try:
    import torch  # noqa: F401
except ImportError:
    _torch = ModuleType("torch")
    _torch.is_tensor = lambda x: False
    sys.modules["torch"] = _torch

# ── scipy: only stub if it's truly missing ──
try:
    import scipy.io.wavfile  # noqa: F401
except ImportError:
    _scipy = ModuleType("scipy")
    _scipy_io = ModuleType("scipy.io")
    _scipy_wavfile = ModuleType("scipy.io.wavfile")
    _scipy_wavfile.write = MagicMock()
    _scipy.io = _scipy_io
    _scipy_io.wavfile = _scipy_wavfile
    sys.modules["scipy"] = _scipy
    sys.modules["scipy.io"] = _scipy_io
    sys.modules["scipy.io.wavfile"] = _scipy_wavfile

try:
    import vosk  # noqa: F401
except ImportError:
    _vosk = ModuleType("vosk")
    _vosk.Model = MagicMock
    _vosk.KaldiRecognizer = MagicMock
    _vosk.SetLogLevel = MagicMock()
    sys.modules["vosk"] = _vosk

# ── openai: ensure the OpenAI class exists (old pip version lacks it) ──
try:
    from openai import OpenAI  # noqa: F401
except (ImportError, AttributeError):
    import openai as _openai_mod

    _openai_mod.OpenAI = MagicMock()

# ── torch.hub: stub torch.hub.load for Silero VAD ──
_torch_mod = sys.modules.get("torch") or sys.modules.setdefault(
    "torch", ModuleType("torch")
)
if not hasattr(_torch_mod, "hub"):
    _torch_hub = ModuleType("torch.hub")
    _torch_hub.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    _torch_mod.hub = _torch_hub
    sys.modules["torch.hub"] = _torch_hub
if not hasattr(_torch_mod, "FloatTensor"):
    _torch_mod.FloatTensor = MagicMock()
if not hasattr(_torch_mod, "no_grad"):
    _torch_mod.no_grad = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )

# ── chromadb: stub only if truly missing ──
try:
    import chromadb  # noqa: F401
except ImportError:
    _chromadb = ModuleType("chromadb")
    _chromadb.Client = MagicMock()
    sys.modules["chromadb"] = _chromadb




@pytest.mark.skipif(
    not hasattr(torch, "hub") or isinstance(torch.hub, MagicMock),
    reason="Real torch required (run in Docker)",
)
class TestRussianTTSIntegration:
    """Integration tests for Silero Russian TTS.

    These tests load the actual model and generate audio, then verify
    the output is valid WAV.  Skipped outside Docker.
    """

    @pytest.fixture(autouse=True)
    def _load_model(self):
        """Load the Russian TTS model once for the class."""
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v3_1_ru",
                trust_repo=True,
            )
            self.speaker = "baya"
            self.sample_rate = 24000
        except Exception:
            pytest.skip("Silero Russian TTS model not available")

    def test_generates_audio_tensor(self):
        audio = self.model.apply_tts(
            text="Привет мир",
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        assert torch.is_tensor(audio)
        assert audio.ndim == 1
        assert audio.shape[0] > 0

    def test_audio_produces_valid_wav(self):
        audio = self.model.apply_tts(
            text="Добрый день, как ваши дела?",
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        buf = io.BytesIO()
        audio_np = audio.cpu().numpy()
        scipy.io.wavfile.write(buf, self.sample_rate, audio_np)
        wav_bytes = buf.getvalue()
        # WAV header starts with RIFF
        assert wav_bytes[:4] == b"RIFF"
        assert len(wav_bytes) > 1000  # non-trivial audio

    def test_different_speakers_produce_audio(self):
        """Verify multiple speakers work."""
        for speaker in ["baya", "xenia"]:
            audio = self.model.apply_tts(
                text="Тест",
                speaker=speaker,
                sample_rate=self.sample_rate,
            )
            assert audio.shape[0] > 0

    def test_long_russian_text(self):
        text = (
            "Фотосинтез — это процесс, при котором растения "
            "используют солнечный свет для превращения углекислого "
            "газа и воды в глюкозу и кислород."
        )
        audio = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=self.sample_rate,
        )
        # Should generate at least ~2 seconds of audio at 24kHz
        assert audio.shape[0] > self.sample_rate * 2


