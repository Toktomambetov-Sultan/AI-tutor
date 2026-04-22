"""Typed message protocol for the response queue shared between
ConversationalAgent and AudioServicer.

Using a NamedTuple keeps backward-compatible tuple unpacking
(``msg_type, data = queue.get()``) while adding named-field access.
Because MessageType extends str, comparisons against legacy string
literals (e.g. ``msg_type == "signal"``) also continue to work.
"""

from enum import Enum
from typing import NamedTuple


class MessageType(str, Enum):
    """Message types exchanged via the asyncio response queue."""

    SIGNAL = "signal"
    AUDIO = "audio"
    AI_TEXT = "ai_text"
    END = "end"


class QueueMessage(NamedTuple):
    """A typed message on the response queue."""

    type: MessageType
    data: object = None
