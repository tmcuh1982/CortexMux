"""Qwen Cloud provider for OpenAI-compatible chat completions."""

from cortexmux.providers.qwencloud.client import QwenCloudClient
from cortexmux.providers.qwencloud.provider import QwenCloudProvider

__all__ = ["QwenCloudClient", "QwenCloudProvider"]
