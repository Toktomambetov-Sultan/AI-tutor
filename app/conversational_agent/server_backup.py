"""
Audio gRPC microservice.
Receives audio chunks via bidirectional streaming,
saves each chunk to local disk, and echoes the same audio back.
"""

import os
import time
import logging
from concurrent import futures

import grpc
from proto import audio_pb2
from proto import audio_pb2_grpc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

AUDIO_SAVE_DIR = os.environ.get("AUDIO_SAVE_DIR", "/data/audio")


class AudioServicer(audio_pb2_grpc.AudioServiceServicer):
    """Implements the AudioService gRPC interface."""

    def StreamAudio(self, request_iterator, context):
        """
        Bidirectional streaming RPC.
        For every incoming AudioChunk:
          1. Save raw bytes to disk under  <AUDIO_SAVE_DIR>/<session_id>/<timestamp>.raw
          2. Yield the same chunk back (echo).
        """
        peer = context.peer()
        logger.info("StreamAudio started  peer=%s", peer)
        chunk_count = 0

        for chunk in request_iterator:
            chunk_count += 1

            # Ensure session directory exists
            session_dir = os.path.join(AUDIO_SAVE_DIR, chunk.session_id)
            os.makedirs(session_dir, exist_ok=True)

            # Write raw audio bytes to disk
            filename = f"{chunk.timestamp_ms}.raw"
            filepath = os.path.join(session_dir, filename)
            with open(filepath, "wb") as f:
                f.write(chunk.data)

            # Echo the same chunk back to the caller
            yield audio_pb2.AudioChunk(
                data=chunk.data,
                session_id=chunk.session_id,
                timestamp_ms=chunk.timestamp_ms,
            )

        logger.info(
            "StreamAudio finished  peer=%s  chunks=%d  session=%s",
            peer,
            chunk_count,
            chunk.session_id if chunk_count else "n/a",
        )


def serve():
    port = os.environ.get("GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_pb2_grpc.add_AudioServiceServicer_to_server(AudioServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Audio service listening on port %s", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

