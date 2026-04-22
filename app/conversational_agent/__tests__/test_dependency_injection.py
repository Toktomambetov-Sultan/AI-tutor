import asyncio
from unittest.mock import MagicMock, patch

from proto import audio_pb2

from core.conversation import ConversationalAgent
from core.resources import SharedResources
from core.grpc_servicer import AudioServicer


def test_load_shared_resources_uses_injected_loader():
    original_resources = ConversationalAgent._default_resources

    injected = SharedResources(
        openai_client=MagicMock(name="openai"),
        tts_model=MagicMock(name="tts"),
        voice_state=MagicMock(name="voice_state"),
        ru_tts_model=MagicMock(name="ru_tts"),
        ru_speaker="baya",
        ru_sample_rate=24000,
        vosk_model_en=MagicMock(name="vosk_en"),
        vosk_model_ru=MagicMock(name="vosk_ru"),
    )

    try:
        ConversationalAgent.load_shared_resources(resource_loader=lambda: injected)
        assert ConversationalAgent._default_resources is injected
    finally:
        ConversationalAgent._default_resources = original_resources


def test_agent_uses_injected_factories():
    response_queue = asyncio.Queue()
    loop = asyncio.new_event_loop()

    fake_openai = MagicMock(name="openai_client")
    fake_rag = MagicMock(name="rag")
    rag_factory = MagicMock(return_value=fake_rag)

    fake_recognizer = MagicMock(name="recognizer")
    recognizer_factory = MagicMock(return_value=fake_recognizer)

    fake_processor = MagicMock(name="audio_processor")
    audio_processor_factory = MagicMock(return_value=fake_processor)

    resources = SharedResources(
        openai_client=fake_openai,
        tts_model=MagicMock(name="tts"),
        voice_state=MagicMock(name="voice_state"),
        ru_tts_model=None,
        ru_speaker=None,
        ru_sample_rate=None,
        vosk_model_en=MagicMock(name="vosk_en"),
        vosk_model_ru=None,
    )
    agent = ConversationalAgent(
        response_queue,
        loop,
        resources=resources,
        rag_factory=rag_factory,
        recognizer_factory=recognizer_factory,
        audio_processor_factory=audio_processor_factory,
    )

    try:
        assert agent.rag is fake_rag
        rag_factory.assert_called_once_with(fake_openai)
        recognizer_factory.assert_called_once()
        audio_processor_factory.assert_called_once()
    finally:
        loop.close()


def test_agent_uses_default_kaldi_recognizer_when_factory_not_injected():
    response_queue = asyncio.Queue()
    loop = asyncio.new_event_loop()

    fake_openai = MagicMock(name="openai_client")
    fake_rag = MagicMock(name="rag")
    rag_factory = MagicMock(return_value=fake_rag)

    fake_recognizer = MagicMock(name="recognizer")
    default_recognizer = MagicMock(return_value=fake_recognizer)

    fake_processor = MagicMock(name="audio_processor")
    audio_processor_factory = MagicMock(return_value=fake_processor)

    resources = SharedResources(
        openai_client=fake_openai,
        tts_model=MagicMock(name="tts"),
        voice_state=MagicMock(name="voice_state"),
        ru_tts_model=None,
        ru_speaker=None,
        ru_sample_rate=None,
        vosk_model_en=MagicMock(name="vosk_en"),
        vosk_model_ru=None,
    )

    with patch("core.conversation.KaldiRecognizer", default_recognizer):
        agent = ConversationalAgent(
            response_queue,
            loop,
            resources=resources,
            rag_factory=rag_factory,
            audio_processor_factory=audio_processor_factory,
        )

    try:
        assert agent.rag is fake_rag
        default_recognizer.assert_called_once_with(resources.vosk_model_en, 16000)
        audio_processor_factory.assert_called_once()
    finally:
        loop.close()


def test_audio_servicer_uses_injected_agent_factory():
    fake_agent = MagicMock()
    factory = MagicMock(return_value=fake_agent)
    servicer = AudioServicer(agent_factory=factory)

    async def _run_test():
        async def request_iterator():
            yield audio_pb2.AudioChunk(
                data=b"audio-bytes",
                lesson_context='{"lesson_title": "Intro"}',
            )

        class _FakeContext:
            def peer(self):
                return "test-peer"

        responses = []
        async for response in servicer.StreamAudio(request_iterator(), _FakeContext()):
            responses.append(response)
        return responses

    responses = asyncio.run(_run_test())

    assert responses
    assert responses[0].signal == "ready"
    factory.assert_called_once()
    fake_agent.process_audio_chunk.assert_called_once_with(b"audio-bytes")
    fake_agent.close.assert_called_once()
