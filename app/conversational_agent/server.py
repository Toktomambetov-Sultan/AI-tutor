"""
Entry point for the conversational-agent gRPC server.

Usage:
    python server.py
"""

import asyncio
import logging
import os

import grpc
from dotenv import load_dotenv
from proto import audio_pb2_grpc

from core.conversation import ConversationalAgent
from core.grpc_servicer import AudioServicer

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


async def serve():
    port = os.environ.get("GRPC_PORT", "50051")

    # ── Pre-load heavy models ONCE at server startup ──
    logger.info("Pre-loading shared resources (TTS model, etc.) ...")
    ConversationalAgent.load_shared_resources()
    logger.info("Shared resources ready.")

    # Create Async gRPC server
    server = grpc.aio.server()

    audio_pb2_grpc.add_AudioServiceServicer_to_server(AudioServicer(), server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info("Audio service listening on port %s", port)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
