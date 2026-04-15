"""Core pipeline modules for the conversational agent.

Heavy dependencies (pocket_tts, torch) are imported lazily so that
lightweight modules like ``core.rag`` and ``core.prompts`` can be used
without pulling in the full TTS / audio stack.
"""


def __getattr__(name: str):
    """Lazy-load submodules on first access to avoid import-time side effects."""
    if name == "RealtimeAudioProcessor":
        from core.audio_processor import RealtimeAudioProcessor

        return RealtimeAudioProcessor
    if name == "ConversationalAgent":
        from core.conversation import ConversationalAgent

        return ConversationalAgent
    if name == "AudioServicer":
        from core.grpc_servicer import AudioServicer

        return AudioServicer
    raise AttributeError(f"module 'core' has no attribute {name!r}")


__all__ = ["RealtimeAudioProcessor", "ConversationalAgent", "AudioServicer"]
