import logging
import os
from dataclasses import dataclass

import torch
from openai import OpenAI
from pocket_tts import TTSModel
from vosk import Model as VoskModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedResources:
    """Process-wide heavy dependencies loaded once and reused by sessions."""

    openai_client: OpenAI
    tts_model: TTSModel
    voice_state: object
    ru_tts_model: object | None
    ru_speaker: str | None
    ru_sample_rate: int | None
    vosk_model_en: VoskModel | None
    vosk_model_ru: VoskModel | None


_ALBA_SAFETENSORS_PATH = os.environ.get(
    "VOICE_STATE_PATH", "/app/models/voice_states/alba.safetensors"
)


def load_en_tts() -> tuple[TTSModel, object]:
    """Load the English PocketTTS model and voice state.

    If a pre-exported safetensors voice state exists (baked into the Docker
    image at build time) it is loaded directly — this is much faster than
    re-processing the audio prompt on every container start.
    """
    logger.info("Loading English TTS model ...")
    tts_model = TTSModel.load_model()
    if os.path.exists(_ALBA_SAFETENSORS_PATH):
        logger.info(
            "Loading pre-cached voice state from %s", _ALBA_SAFETENSORS_PATH
        )
        voice_state = tts_model.get_state_for_audio_prompt(_ALBA_SAFETENSORS_PATH)
    else:
        logger.info("Pre-cached voice state not found; deriving from audio prompt.")
        voice_state = tts_model.get_state_for_audio_prompt("alba")
    logger.info("English TTS model loaded.")
    return tts_model, voice_state


def load_ru_tts() -> tuple[object | None, str | None, int | None]:
    """Load the Silero Russian TTS model if available."""
    logger.info("Loading Russian TTS model ...")
    try:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v3_1_ru",
            trust_repo=True,
        )
        logger.info("Russian TTS model loaded.")
        return model, "baya", 24000
    except Exception as e:
        logger.warning("Failed to load Russian TTS model: %s", e)
        return None, None, None


def load_vosk_models(
    model_en_path: str | None = None,
    model_ru_path: str | None = None,
) -> tuple[VoskModel | None, VoskModel | None]:
    """Load Vosk EN and RU STT models, returning None for unavailable models."""
    en_path = model_en_path or os.environ.get(
        "VOSK_MODEL_EN", "/app/models/vosk-model-small-en-us-0.15"
    )
    ru_path = model_ru_path or os.environ.get(
        "VOSK_MODEL_RU", "/app/models/vosk-model-small-ru-0.22"
    )

    def _load(path: str, label: str) -> VoskModel | None:
        logger.info("Loading Vosk %s model from %s ...", label, path)
        try:
            model = VoskModel(path)
            logger.info("Vosk %s model loaded.", label)
            return model
        except Exception as e:
            logger.warning("Failed to load Vosk %s model: %s", label, e)
            return None

    return _load(en_path, "EN"), _load(ru_path, "RU")


def build_default_shared_resources(
    openai_api_key: str | None = None,
    model_en_path: str | None = None,
    model_ru_path: str | None = None,
) -> SharedResources:
    """Build process-wide shared resources from environment and model hubs."""
    openai_client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
    logger.info("OpenAI client created.")

    tts_model, voice_state = load_en_tts()
    ru_tts_model, ru_speaker, ru_sample_rate = load_ru_tts()
    vosk_model_en, vosk_model_ru = load_vosk_models(model_en_path, model_ru_path)

    return SharedResources(
        openai_client=openai_client,
        tts_model=tts_model,
        voice_state=voice_state,
        ru_tts_model=ru_tts_model,
        ru_speaker=ru_speaker,
        ru_sample_rate=ru_sample_rate,
        vosk_model_en=vosk_model_en,
        vosk_model_ru=vosk_model_ru,
    )
