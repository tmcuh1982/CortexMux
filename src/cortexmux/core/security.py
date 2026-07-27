"""Local-first network and filesystem safeguards."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from cortexmux.core.exceptions import OutputPathError, RemoteHostNotAllowedError


def validate_provider_url(
    url: str,
    *,
    allow_remote_hosts: bool = False,
    approved_hosts: set[str] | None = None,
) -> str:
    """Validate an HTTP provider URL before any request is sent."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RemoteHostNotAllowedError("Provider URL must be an absolute HTTP(S) URL.", url=url)
    host = parsed.hostname.lower().rstrip(".")
    if allow_remote_hosts or host in (approved_hosts or set()) or host == "localhost":
        return url.rstrip("/")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, parsed.port)}
    except socket.gaierror as exc:
        raise RemoteHostNotAllowedError(
            "Provider host could not be resolved safely.", host=host
        ) from exc
    if addresses and all(ipaddress.ip_address(address).is_loopback for address in addresses):
        return url.rstrip("/")
    raise RemoteHostNotAllowedError(
        "Remote provider hosts are disabled. Enable allow_remote_hosts explicitly.",
        host=host,
    )


def safe_output_path(root: Path, filename: str, *, overwrite: bool = False) -> Path:
    """Resolve a new file beneath *root* and prevent traversal or overwrites."""
    resolved_root = root.expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    candidate = (resolved_root / filename).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise OutputPathError("Output path escapes the configured directory.", path=str(candidate))
    if candidate.exists() and not overwrite:
        raise OutputPathError("Output file already exists.", path=str(candidate))
    return candidate
