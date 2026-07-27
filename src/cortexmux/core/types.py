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


class ProgressStage(StrEnum):
    """Normalized stages emitted while a provider executes work."""

    SUBMITTING = "submitting"
    QUEUED = "queued"
    STATUS = "status"
    EXECUTION_STARTED = "execution_started"
    NODE_STARTED = "node_started"
    NODE_PROGRESS = "node_progress"
    NODE_COMPLETED = "node_completed"
    EXECUTION_CACHED = "execution_cached"
    EXECUTION_COMPLETED = "execution_completed"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    ERROR = "error"
