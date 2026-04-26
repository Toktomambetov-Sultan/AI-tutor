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
    not hasattr(sys.modules.get("vosk", None), "__file__"),
    reason="Real vosk required (run in Docker)",
)
class TestVoskIntegration:
    """Integration tests for Vosk streaming STT.

    These tests load the actual Vosk model and run recognition on
    synthetic PCM.  Skipped outside Docker.
    """

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from vosk import Model as _VoskModel, KaldiRecognizer as _KaldiRec
        import json as _json

        model_path = "/app/models/vosk-model-small-en-us-0.15"
        try:
            self.model = _VoskModel(model_path)
        except Exception:
            pytest.skip(f"Vosk model not found at {model_path}")
        self.sample_rate = 16000
        self._json = _json
        self._KaldiRec = _KaldiRec

    def test_recognizer_accepts_silence(self):
        """Recognizer should return empty text for silence."""
        import numpy as np

        rec = self._KaldiRec(self.model, self.sample_rate)
        silence = np.zeros(self.sample_rate, dtype=np.int16).tobytes()
        rec.AcceptWaveform(silence)
        result = self._json.loads(rec.FinalResult())
        assert result.get("text", "") == ""

    def test_recognizer_returns_text_for_speech(self):
        """Recognizer should produce a result for a tone
        (may not be meaningful, but shouldn't crash)."""
        import numpy as np

        rec = self._KaldiRec(self.model, self.sample_rate)
        t = np.linspace(0, 1, self.sample_rate, dtype=np.float64)
        pcm = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).tobytes()
        rec.AcceptWaveform(pcm)
        result = self._json.loads(rec.FinalResult())
        assert "text" in result

    def test_russian_model_loads(self):
        """Verify the Russian Vosk model can be loaded."""
        from vosk import Model as _VoskModel
        import numpy as np

        ru_model_path = "/app/models/vosk-model-small-ru-0.22"
        try:
            ru_model = _VoskModel(ru_model_path)
        except Exception:
            pytest.skip(f"Russian Vosk model not found at {ru_model_path}")
        rec = self._KaldiRec(ru_model, self.sample_rate)
        silence = np.zeros(self.sample_rate, dtype=np.int16).tobytes()
        rec.AcceptWaveform(silence)
        result = self._json.loads(rec.FinalResult())
        assert "text" in result


