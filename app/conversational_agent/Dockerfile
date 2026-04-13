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

# Application code (single layer — code changes don't bust the
# large site-packages cache layer above)
COPY core/ core/
COPY prompts.py server.py rag.py ./

RUN mkdir -p /data/audio

EXPOSE 50051

CMD ["python", "server.py"]
