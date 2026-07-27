"""Safe, structured exception hierarchy."""

from __future__ import annotations


class CortexMuxError(Exception):
    """Base exception carrying non-sensitive diagnostic context."""

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = {key: value for key, value in context.items() if value is not None}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic representation."""
        return {"error": type(self).__name__, "message": self.message, "context": self.context}


class ConfigurationError(CortexMuxError):
    """Configuration is invalid."""


class ProviderRegistrationError(CortexMuxError):
    """A provider cannot be registered."""


class ProviderNotFoundError(CortexMuxError):
    """A provider is not registered."""


class ProviderUnavailableError(CortexMuxError):
    """A provider endpoint is unavailable."""


class ProviderTimeoutError(CortexMuxError):
    """A provider request timed out."""


class ProviderResponseError(CortexMuxError):
    """A provider returned an invalid or failed response."""


class ModelNotFoundError(CortexMuxError):
    """A requested model is unavailable."""


class ModelSelectionError(CortexMuxError):
    """Routing cannot select one unambiguous model."""


class UnsupportedTaskError(CortexMuxError):
    """A provider does not support the requested task."""


class InvalidRequestError(CortexMuxError):
    """A normalized request is invalid."""


class StructuredOutputValidationError(CortexMuxError):
    """Structured model output failed validation."""


class WorkflowValidationError(CortexMuxError):
    """A ComfyUI workflow or binding is invalid."""


class WorkflowExecutionError(CortexMuxError):
    """A ComfyUI workflow failed during execution."""


class DataSourceError(CortexMuxError):
    """A data source is invalid or unsafe."""


class DataAnalysisError(CortexMuxError):
    """A deterministic data operation failed."""


class OptionalDependencyError(CortexMuxError):
    """A requested feature needs an optional dependency."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"Optional dependency '{package}' is required; install cortexmux[{extra}].",
            package=package,
            extra=extra,
        )


class SecurityError(CortexMuxError):
    """A security policy rejected an operation."""


class RemoteHostNotAllowedError(SecurityError):
    """A remote endpoint is blocked by the local-first policy."""


class OutputPathError(SecurityError):
    """An output path escapes its configured root or would overwrite a file."""
