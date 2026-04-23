# syntax=docker/dockerfile:1

# ===========================================================
#  Stage 1 — Install deps with uv (10-100× faster than pip)
#  Uses glibc-based slim for full wheel compatibility.
#  BuildKit cache mount → uv's cache survives across builds,
#  so even when requirements.txt changes only the diff is
#  downloaded — unchanged packages are instant.
# ===========================================================
FROM python:3.12-slim AS builder

# Grab the static uv binary (ultra-fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY requirements.txt .

# • --mount=type=cache  → persists uv download + wheel cache across
#     every build; only new / changed packages are fetched.
# • --extra-index-url …/cpu  → pulls CPU-only PyTorch (~300 MB vs ~2 GB).
# • --index-strategy unsafe-best-match → pick best version across indexes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# Download Vosk speech-recognition models (EN + RU)
RUN python -c "import urllib.request, zipfile, os; os.makedirs('/app/models', exist_ok=True); [(print(f'Downloading {n}...'), urllib.request.urlretrieve(u, f'/tmp/{n}.zip'), zipfile.ZipFile(f'/tmp/{n}.zip').extractall('/app/models'), os.remove(f'/tmp/{n}.zip')) for u, n in [('https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip','vosk-model-small-en-us-0.15'),('https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip','vosk-model-small-ru-0.22')]]; print('Done.')"

# ── Pre-download pocket_tts (English TTS) model and export the
#    default voice state to safetensors for fast runtime loading.
#    HF_HOME is pinned so the cache lands inside /app/models/hf_cache
#    and survives into the final image layer.
ENV HF_HOME=/app/models/hf_cache
RUN python -c "\
import os; \
os.makedirs('/app/models/voice_states', exist_ok=True); \
from pocket_tts import TTSModel, export_model_state; \
print('Downloading pocket_tts model ...'); \
m = TTSModel.load_model(); \
print('Exporting alba voice state to safetensors ...'); \
state = m.get_state_for_audio_prompt('alba'); \
export_model_state(state, '/app/models/voice_states/alba.safetensors'); \
print('pocket_tts model cached.'); \
"

# ── Pre-download Silero TTS (Russian) via torch.hub.
#    TORCH_HOME pins the hub cache inside /app/models/torch_hub so it
#    is copied into the final image and never re-downloaded at runtime.
ENV TORCH_HOME=/app/models/torch_hub
RUN python -c "\
import torch; \
print('Downloading Silero TTS RU model ...'); \
torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v3_1_ru', trust_repo=True); \
print('Silero TTS RU model cached.'); \
"

# Generate gRPC Python stubs (grpc_tools needs native libs)
COPY proto/ proto/
RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    proto/audio.proto


# ===========================================================
#  Stage 2 — Lean runtime (no build tooling, no uv, no gcc)
#  NOTE: Alpine is incompatible with PyTorch (libtorch links
#  glibc symbols like fcntl64 that musl/gcompat cannot shim).
#  slim is the thinnest image that works with torch.
# ===========================================================
FROM python:3.12-slim

# ffmpeg decodes WebM/Opus audio from the browser
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the installed packages + binaries (no gcc, uv, headers …)
COPY --from=builder /usr/local/lib/python3.12/site-packages/ \
    /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy pre-generated gRPC stubs
COPY --from=builder /app/proto/ proto/

# Copy all pre-cached models: Vosk, pocket_tts (HuggingFace), Silero (torch.hub), voice states
COPY --from=builder /app/models/ models/

# Point HuggingFace and torch.hub to the pre-cached directories so
# pocket_tts / torch.hub never attempt a network download at runtime.
ENV HF_HOME=/app/models/hf_cache
ENV TORCH_HOME=/app/models/torch_hub

# Application code (single layer — code changes don't bust the
# large site-packages cache layer above)
COPY core/ core/
COPY server.py ./
COPY pyproject.toml ./
COPY __tests__/ __tests__/

RUN mkdir -p /data/audio

EXPOSE 50051

CMD ["python", "server.py"]
