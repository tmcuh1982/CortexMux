"""Core enumerations shared by providers and schemas."""

from enum import StrEnum


class TaskType(StrEnum):
    """Tasks understood by CortexMux."""

    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    DATA_ANALYSIS = "data_analysis"


class MessageRole(StrEnum):
    """Normalized chat roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
