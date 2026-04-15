"""
AI Agent router — WebSocket audio streaming with gRPC echo microservice.

Flow:
  1.  Client opens WebSocket to /api/v1/ws/audio/{lesson_id}?token=<session_token>
  2.  Backend fetches lesson materials from the database and builds a lesson context.
  3.  The first gRPC message carries the lesson context (JSON) for RAG setup.
  4.  Client sends binary audio chunks (from MediaRecorder, webm/opus).
  5.  Backend forwards each chunk to the audio_service via gRPC bidirectional streaming.
  6.  audio_service uses RAG-augmented AI to respond with lesson-aware audio.
  7.  Backend relays the AI audio back to the client over WebSocket.
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
from app.models.lesson import Lesson
from app.models.material import Material
from app.models.class_ import Class
from app.models.course import Course
from app.redis import redis_client
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

AUDIO_SERVICE_HOST = os.environ.get("AUDIO_SERVICE_HOST", "10.129.0.158:50051")
settings = get_settings()

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


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file on disk. Returns empty string on failure."""
    full_path = os.path.join(settings.UPLOAD_DIR, file_path)
    if not os.path.exists(full_path):
        logger.warning("PDF file not found: %s", full_path)
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(full_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n".join(text_parts)
        logger.info("Extracted %d chars from PDF %s", len(text), file_path)
        return text
    except Exception as e:
        logger.error("Failed to extract PDF text from %s: %s", file_path, e)
        return ""


async def _build_lesson_context(lesson_id: uuid.UUID) -> str:
    """Fetch lesson, its class/course hierarchy, and materials from the DB.
    Returns a JSON string suitable for the RAG system."""
    async with async_session_factory() as db:
        # Fetch lesson
        result = await db.execute(
            select(Lesson).where(Lesson.id == lesson_id, Lesson.deleted_at.is_(None))
        )
        lesson = result.scalar_one_or_none()
        if not lesson:
            logger.warning("Lesson %s not found for RAG context", lesson_id)
            return ""

        # Fetch class and course for hierarchy titles
        class_title = ""
        course_title = ""
        if lesson.class_id:
            cls_result = await db.execute(
                select(Class).where(Class.id == lesson.class_id)
            )
            cls = cls_result.scalar_one_or_none()
            if cls:
                class_title = cls.title
                if cls.course_id:
                    course_result = await db.execute(
                        select(Course).where(Course.id == cls.course_id)
                    )
                    course = course_result.scalar_one_or_none()
                    if course:
                        course_title = course.title

        # Fetch materials
        mat_result = await db.execute(
            select(Material).where(
                Material.lesson_id == lesson_id,
                Material.deleted_at.is_(None),
            )
        )
        materials = mat_result.scalars().all()

        material_list = []
        for m in materials:
            content = ""
            if m.type.value == "text":
                content = m.content or ""
            elif m.type.value == "pdf" and m.file_path:
                content = _extract_pdf_text(m.file_path)

            if content.strip():
                material_list.append(
                    {
                        "type": m.type.value,
                        "content": content,
                        "title": f"{m.type.value.upper()} material",
                    }
                )

        context = {
            "lesson_title": lesson.title,
            "class_title": class_title,
            "course_title": course_title,
            "materials": material_list,
        }

        context_json = json.dumps(context)
        logger.info(
            "Built lesson context for %r: %d materials, %d chars",
            lesson.title,
            len(material_list),
            len(context_json),
        )
        return context_json


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

    # --- fetch lesson materials for RAG ---
    lesson_context = await _build_lesson_context(lesson_id)

    # --- open gRPC channel to audio_service ---
    _ensure_grpc_stubs()
    channel = grpc.aio.insecure_channel(AUDIO_SERVICE_HOST)
    stub = _audio_pb2_grpc.AudioServiceStub(channel)

    # Queue that bridges the WS receive loop → gRPC request generator
    send_queue: asyncio.Queue = asyncio.Queue()

    # Send lesson context as the FIRST gRPC message
    await send_queue.put(
        _audio_pb2.AudioChunk(
            data=b"",
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
            lesson_context=lesson_context,
        )
    )

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
                raw = await websocket.receive()
                # Support both binary audio frames and text control messages
                if "bytes" in raw and raw["bytes"]:
                    await send_queue.put(
                        _audio_pb2.AudioChunk(
                            data=raw["bytes"],
                            session_id=session_id,
                            timestamp_ms=int(time.time() * 1000),
                        )
                    )
                elif "text" in raw and raw["text"]:
                    import json as _json

                    try:
                        msg = _json.loads(raw["text"])
                        if msg.get("signal") == "interrupt":
                            logger.info(
                                "Client interrupt signal  session=%s", session_id
                            )
                            await send_queue.put(
                                _audio_pb2.AudioChunk(
                                    session_id=session_id,
                                    timestamp_ms=int(time.time() * 1000),
                                    client_signal="interrupt",
                                )
                            )
                    except Exception:
                        pass
        except WebSocketDisconnect:
            logger.info("Client disconnected  session=%s", session_id)
        except Exception as exc:
            logger.warning("WS receive error: %s", exc)
        finally:
            await send_queue.put(None)  # signal gRPC iterator to stop

    # --- task: read responses from gRPC → push to client WS ---
    async def _grpc_to_ws():
        try:
            async for response in response_stream:
                # Control signals are sent as JSON text frames
                if response.signal:
                    ctrl = {"signal": response.signal}
                    if response.ai_text:
                        ctrl["ai_text"] = response.ai_text
                    await websocket.send_json(ctrl)
                elif response.data:
                    # Audio data — optionally attach ai_text as a preceding text frame
                    if response.ai_text:
                        await websocket.send_json({"ai_text": response.ai_text})
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
