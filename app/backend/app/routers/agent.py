"""
AI Agent router — WebSocket audio streaming with gRPC echo microservice.

Flow:
  1.  Client opens WebSocket to /api/v1/ws/audio/{lesson_id}?token=<session_token>
  2.  Client sends binary audio chunks (from MediaRecorder, webm/opus).
  3.  Backend forwards each chunk to the audio_service via gRPC bidirectional streaming.
  4.  audio_service saves to disk and echoes back the same chunk.
  5.  Backend relays the echoed bytes to the client over WebSocket.
  6.  Client plays back the received audio.
"""

import asyncio
import json
import os
import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

import grpc
import grpc.aio

from app.database import async_session_factory
from app.models.user import User, UserRole
from app.redis import redis_client

logger = logging.getLogger(__name__)
router = APIRouter()

AUDIO_SERVICE_HOST = os.environ.get("AUDIO_SERVICE_HOST", "audio_service:50051")

# ---------------------------------------------------------------------------
# Lazy import of generated gRPC stubs — they only exist after Docker build
# runs protoc.  We import them at call-time so the module can still be loaded
# in environments without the compiled proto (e.g. IDE, tests).
# ---------------------------------------------------------------------------
_audio_pb2 = None
_audio_pb2_grpc = None


def _ensure_grpc_stubs():
    global _audio_pb2, _audio_pb2_grpc
    if _audio_pb2 is None:
        from app import audio_pb2, audio_pb2_grpc  # type: ignore

        _audio_pb2 = audio_pb2
        _audio_pb2_grpc = audio_pb2_grpc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _authenticate_ws(token: str) -> Optional[User]:
    """Validate a session token from Redis, return the User or None."""
    session_data = await redis_client.get(f"session:{token}")
    if not session_data:
        return None

    data = json.loads(session_data)
    user_id = uuid.UUID(data["user_id"])

    async with async_session_factory() as db:
        result = await db.execute(
            select(User).where(User.id == user_id, User.deleted_at.is_(None))
        )
        user = result.scalar_one_or_none()
        if user and user.is_active:
            return user
    return None


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/audio/{lesson_id}")
async def audio_websocket(websocket: WebSocket, lesson_id: uuid.UUID):
    # --- authenticate via query-param token ---
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    user = await _authenticate_ws(token)
    if not user or user.role != UserRole.student:
        await websocket.close(code=4003, reason="Unauthorized")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(
        "Audio WS opened  user=%s  lesson=%s  session=%s",
        user.id,
        lesson_id,
        session_id,
    )

    # --- open gRPC channel to audio_service ---
    _ensure_grpc_stubs()
    channel = grpc.aio.insecure_channel(AUDIO_SERVICE_HOST)
    stub = _audio_pb2_grpc.AudioServiceStub(channel)

    # Queue that bridges the WS receive loop → gRPC request generator
    send_queue: asyncio.Queue = asyncio.Queue()

    async def _grpc_request_iter():
        """Yields AudioChunks until a sentinel (None) is enqueued."""
        while True:
            item = await send_queue.get()
            if item is None:
                return
            yield item

    # Start the bidirectional gRPC stream
    response_stream = stub.StreamAudio(_grpc_request_iter())

    # --- task: read from client WS → feed into gRPC ---
    async def _ws_to_grpc():
        try:
            while True:
                data = await websocket.receive_bytes()
                await send_queue.put(
                    _audio_pb2.AudioChunk(
                        data=data,
                        session_id=session_id,
                        timestamp_ms=int(time.time() * 1000),
                    )
                )
        except WebSocketDisconnect:
            logger.info("Client disconnected  session=%s", session_id)
        except Exception as exc:
            logger.warning("WS receive error: %s", exc)
        finally:
            await send_queue.put(None)  # signal gRPC iterator to stop

    # --- task: read echoed chunks from gRPC → push to client WS ---
    async def _grpc_to_ws():
        try:
            async for response in response_stream:
                await websocket.send_bytes(response.data)
        except Exception as exc:
            logger.warning("gRPC response stream error: %s", exc)

    # Run both directions concurrently
    try:
        await asyncio.gather(_ws_to_grpc(), _grpc_to_ws())
    except Exception as exc:
        logger.error("Audio session error: %s", exc)
    finally:
        await channel.close()
        logger.info("Audio session closed  session=%s", session_id)
