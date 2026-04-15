"""
gRPC AudioService implementation.

Handles bidirectional audio streaming between the browser client
and the ConversationalAgent.  Supports:
  - "ready" signal after RAG initialisation completes
  - Barge-in / interrupt handling
  - Sentence-level streaming TTS
"""

import asyncio
import logging

import grpc
from proto import audio_pb2

from core.conversation import ConversationalAgent

logger = logging.getLogger(__name__)


class AudioServicer:
    """
    Implements the AudioService gRPC interface using AsyncIO.
    Integrates the ConversationalAgent with RAG-powered lesson awareness.
    """

    async def StreamAudio(self, request_iterator, context):
        """
        Bidirectional streaming RPC.
        1. The FIRST message carries lesson_context (JSON) for RAG setup.
        2. Server sends a "ready" signal once RAG is initialised.
        3. Subsequent messages carry audio chunks.
        4. Messages with client_signal="interrupt" trigger barge-in.
        5. AI audio is streamed sentence-by-sentence with ai_text metadata.
        """
        peer = context.peer()
        logger.info("StreamAudio started  peer=%s", peer)

        response_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        agent = None
        first_message = True

        async def request_consumer():
            """Consumes messages from the client and feeds them to the agent."""
            nonlocal agent, first_message
            try:
                async for chunk in request_iterator:
                    # ── Handle client-side interrupt signal ──
                    if chunk.client_signal == "interrupt" and agent is not None:
                        logger.info("Client interrupt received — cancelling TTS")
                        agent.handle_interrupt()
                        continue

                    if first_message:
                        lesson_context = chunk.lesson_context or ""
                        if lesson_context:
                            logger.info(
                                "Received lesson context (%d chars) — initialising RAG",
                                len(lesson_context),
                            )
                        else:
                            logger.info("No lesson context provided — generic mode")

                        agent = ConversationalAgent(
                            response_queue, loop, lesson_context=lesson_context
                        )
                        first_message = False

                        # ── Send "ready" signal to the client ──
                        await response_queue.put(("signal", "ready"))

                        if chunk.data:
                            agent.process_audio_chunk(chunk.data)
                    else:
                        if agent is not None:
                            agent.process_audio_chunk(chunk.data)
            except grpc.RpcError as e:
                logger.info(f"Client stream closed: {e.code()}")
            except Exception as e:
                logger.error(f"Error in request consumer: {e}")

        consumer_task = asyncio.create_task(request_consumer())

        try:
            audio_buffer = bytearray()
            current_ai_text = ""

            while True:
                if consumer_task.done():
                    if audio_buffer:
                        yield audio_pb2.AudioChunk(
                            data=bytes(audio_buffer), ai_text=current_ai_text
                        )
                        audio_buffer = bytearray()
                    if response_queue.empty():
                        break

                try:
                    msg_type, data = await asyncio.wait_for(
                        response_queue.get(), timeout=0.1
                    )

                    if msg_type == "signal":
                        # Control signal — send immediately with no audio
                        yield audio_pb2.AudioChunk(signal=data)

                    elif msg_type == "audio":
                        audio_buffer.extend(data)

                    elif msg_type == "ai_text":
                        # Text of the sentence whose audio follows
                        current_ai_text = data

                    elif msg_type == "end":
                        if audio_buffer:
                            yield audio_pb2.AudioChunk(
                                data=bytes(audio_buffer),
                                ai_text=current_ai_text,
                            )
                            audio_buffer = bytearray()
                            current_ai_text = ""

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            consumer_task.cancel()
            if agent is not None:
                agent.close()
            logger.info("StreamAudio finished  peer=%s", peer)
