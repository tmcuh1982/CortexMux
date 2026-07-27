"""CortexMux public API."""

from cortexmux.facade import CortexMux
from cortexmux.version import __version__
from cortexmux.web import WebPage, WebTable

__all__ = ["CortexMux", "WebPage", "WebTable", "__version__"]
