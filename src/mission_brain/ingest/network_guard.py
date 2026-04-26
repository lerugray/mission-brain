"""Network egress guard (extracted for mission-brain without the full privacy package).

`assert_local(url, allowlist=None)` raises `NetworkEgressViolation`
if the URL's host is not one of the loopback identifiers
(`127.0.0.1`, `localhost`, `::1`) or explicitly allowlisted.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

__all__ = ["NetworkEgressViolation", "assert_local"]


_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


class NetworkEgressViolation(Exception):
    """A URL points at a non-local host and is not in the allowlist."""


def _is_loopback_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_local(url: str, allowlist: list[str] | None = None) -> None:
    """Raise `NetworkEgressViolation` if `url` is not a local address.

    A URL clears the guard when either its textual host is in the
    loopback set or the allowlist, OR its host resolves to a
    loopback IP. Hostnames that fail DNS resolution are rejected
    rather than silently accepted.
    """

    allowed = frozenset(allowlist or ())

    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        raise NetworkEgressViolation(f"URL has no host component: {url!r}")

    host_lower = host.lower()
    if host_lower in _LOOPBACK_HOSTS or host_lower in allowed:
        return

    if _is_loopback_ip(host):
        return

    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False

    if is_ip_literal:
        raise NetworkEgressViolation(
            f"URL host {host!r} is not loopback and not in allowlist: {url!r}"
        )

    try:
        resolved = socket.gethostbyname(host)
    except OSError as exc:
        raise NetworkEgressViolation(
            f"URL host {host!r} failed to resolve: {exc}"
        ) from exc

    if _is_loopback_ip(resolved):
        return

    raise NetworkEgressViolation(
        f"URL host {host!r} resolved to {resolved!r}, which is not loopback "
        f"and not in allowlist: {url!r}"
    )
