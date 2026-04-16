"""Shared fixtures for conversational-agent tests."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_kaldi_recognizer():
    """Prevent real KaldiRecognizer from being instantiated in unit tests.

    The C-extension constructor requires a genuine VoskModel pointer;
    MagicMock objects crash it.  Patching at the module level in
    conversation.py lets every test that creates a ConversationalAgent
    skip the real Vosk call without per-test patching.
    """
    with patch("core.conversation.KaldiRecognizer"):
        yield
