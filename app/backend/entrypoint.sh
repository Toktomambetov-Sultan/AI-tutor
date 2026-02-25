#!/bin/bash
set -e

# Re-generate gRPC stubs (volume mount may overwrite the build-time copy)
if [ -f /app/proto/audio.proto ]; then
    python -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/app \
        --grpc_python_out=/app/app \
        /app/proto/audio.proto

    # Fix absolute import in generated grpc stub so it works inside the app package
    sed -i 's/^import audio_pb2 as audio__pb2$/from app import audio_pb2 as audio__pb2/' /app/app/audio_pb2_grpc.py

    echo "gRPC stubs generated and patched."
fi

exec "$@"
