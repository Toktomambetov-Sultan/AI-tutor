"""
Real-time gRPC test — generates speech via PocketTTS (using the model
already cached on disk), encodes as WebM/Opus, sends through gRPC
while keeping the stream open.  Verifies agent responds MID-STREAM.

Usage (inside the container):
    python test_grpc.py
"""

import asyncio
import subprocess
import struct
import math
import io
import grpc
from proto import audio_pb2, audio_pb2_grpc

GRPC_HOST = "localhost:50051"


def generate_speech_wav() -> bytes:
    """Generate a speech WAV using PocketTTS (model already cached from server startup)."""
    import torch
    import scipy.io.wavfile
    from pocket_tts import TTSModel

    print("[TEST] Loading PocketTTS (cached) ...")
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt("alba")
    text = "Hello, what is the capital of France?"
    print(f"[TEST] Generating speech: '{text}'")
    t = model.generate_audio(state, text)
    a = t.cpu().numpy() if torch.is_tensor(t) else t
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, model.sample_rate, a)
    return buf.getvalue()


async def run_test():
    # 1. Generate real speech + silence padding
    print("[TEST] Generating speech WAV ...")
    wav = generate_speech_wav()
    print(f"[TEST] Speech WAV size: {len(wav)} bytes")

    # 2. Encode to WebM/Opus with 3s silence pad
    print("[TEST] Encoding to WebM/Opus with 3s silence pad ...")
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            "pipe:0",
            "-af",
            "apad=pad_dur=3",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-ar",
            "48000",
            "-f",
            "webm",
            "pipe:1",
        ],
        input=wav,
        capture_output=True,
        timeout=30,
    )
    assert r.returncode == 0, f"ffmpeg failed: {r.stderr.decode()[:300]}"
    full_webm = r.stdout
    print(f"[TEST] WebM size: {len(full_webm)} bytes")

    # 3. Split into small chunks (like MediaRecorder would)
    CHUNK = 2048
    chunks = [full_webm[i : i + CHUNK] for i in range(0, len(full_webm), CHUNK)]
    print(f"[TEST] Split into {len(chunks)} chunks of ~{CHUNK}B\n")

    # 4. gRPC stream
    print(f"[TEST] Connecting to {GRPC_HOST} ...")
    channel = grpc.aio.insecure_channel(GRPC_HOST)
    stub = audio_pb2_grpc.AudioServiceStub(channel)

    t0 = asyncio.get_event_loop().time()

    async def send_chunks():
        """Send all chunks at pace, then keep the stream open for 20s more."""
        for i, c in enumerate(chunks):
            yield audio_pb2.AudioChunk(
                data=c, session_id="vad-test", timestamp_ms=i * 100
            )
            await asyncio.sleep(0.05)  # fast but paced
        t_sent = asyncio.get_event_loop().time()
        print(
            f"[TEST] All {len(chunks)} chunks sent at t={t_sent - t0:.1f}s.  "
            f"Keeping stream open 20s for mid-stream response ..."
        )
        await asyncio.sleep(20)
        print("[TEST] Closing sender.")

    stream = stub.StreamAudio(send_chunks())

    received = 0
    n = 0

    try:
        async for resp in stream:
            n += 1
            received += len(resp.data)
            elapsed = asyncio.get_event_loop().time() - t0
            print(
                f"[TEST] <- response chunk #{n}: {len(resp.data)}B  "
                f"total={received}B  @ {elapsed:.1f}s"
            )
    except grpc.aio.AioRpcError as e:
        print(f"[TEST] gRPC error: {e.code()} — {e.details()}")

    elapsed = asyncio.get_event_loop().time() - t0
    print(f"\n{'='*60}")
    print(f"[TEST] Finished — received {received}B in {n} chunks over {elapsed:.1f}s")

    if received > 0 and elapsed < 18:
        print("[TEST] ✅ SUCCESS — response arrived mid-stream (real-time VAD works!)")
    elif received > 0:
        print(
            "[TEST] ⚠️ Response arrived but only after stream ended (VAD not real-time)"
        )
    else:
        print("[TEST] ❌ No response received")
        print("[TEST]    Check agent logs: docker logs app-conversational_agent-1")
    print(f"{'='*60}")

    await channel.close()


if __name__ == "__main__":
    asyncio.run(run_test())
