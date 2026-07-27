"""Local-first network and filesystem safeguards."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urldefrag, urlparse

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


def validate_web_url(
    url: str,
    *,
    allowed_hosts: set[str] | None = None,
    allowed_ports: set[int] | None = None,
    allow_private_hosts: bool = False,
) -> str:
    """Validate one web-page URL, including its resolved IP addresses."""
    normalized, _fragment = urldefrag(url)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RemoteHostNotAllowedError(
            "Web URL must be an absolute HTTP(S) URL.",
            url=url,
        )
    if parsed.username is not None or parsed.password is not None:
        raise RemoteHostNotAllowedError("Web URLs must not contain credentials.")
    host = parsed.hostname.lower().rstrip(".")
    approved = {item.lower().rstrip(".") for item in allowed_hosts or set()}
    if approved and host not in approved:
        raise RemoteHostNotAllowedError(
            "Web host is not in the configured allowlist.",
            host=host,
        )
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise RemoteHostNotAllowedError("Web URL contains an invalid port.", host=host) from exc
    permitted_ports = {80, 443} if allowed_ports is None else allowed_ports
    if port not in permitted_ports:
        raise RemoteHostNotAllowedError(
            "Web URL port is not allowed.",
            host=host,
            port=port,
        )
    try:
        literal = ipaddress.ip_address(host)
        addresses = {literal}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, port)
            }
        except (socket.gaierror, ValueError) as exc:
            raise RemoteHostNotAllowedError(
                "Web host could not be resolved safely.",
                host=host,
            ) from exc
    if not addresses:
        raise RemoteHostNotAllowedError("Web host resolved to no addresses.", host=host)
    if not allow_private_hosts and any(not address.is_global for address in addresses):
        raise RemoteHostNotAllowedError(
            "Web URL resolves to a private or non-public address.",
            host=host,
        )
    return normalized


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
