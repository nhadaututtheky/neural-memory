"""Shared dependencies for API routes."""

from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from neural_memory.core.brain import Brain
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


@lru_cache(maxsize=1)
def _parse_trusted_networks(
    networks: tuple[str, ...],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse and cache CIDR network strings into ip_network objects."""
    parsed: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for net in networks:
        if not net:
            continue
        try:
            parsed.append(ipaddress.ip_network(net, strict=False))
        except ValueError:
            logger.warning("Invalid trusted network CIDR: %s (skipped)", net)
    return tuple(parsed)


def is_trusted_host(host: str) -> bool:
    """Check if a host is trusted (localhost or in configured trusted networks).

    Args:
        host: Client IP address or hostname.

    Returns:
        True if the host is localhost or within a trusted network CIDR.
    """
    if host in _LOCALHOST_HOSTS:
        return True

    from neural_memory.utils.config import get_config

    config = get_config()
    if not config.trusted_networks:
        return False

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False

    parsed = _parse_trusted_networks(tuple(config.trusted_networks))
    return any(addr in net for net in parsed)


async def require_local_request(request: Request) -> None:
    """Reject requests from untrusted sources.

    Allows localhost and any IP within NEURAL_MEMORY_TRUSTED_NETWORKS CIDRs.
    """
    if request.client is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_trusted_host(request.client.host):
        raise HTTPException(status_code=403, detail="Forbidden")


async def get_storage() -> NeuralStorage:
    """
    Dependency to get storage instance.

    This is overridden by the application at startup.
    """
    raise NotImplementedError("Storage not configured")


async def get_brain(
    brain_id: Annotated[str, Header(alias="X-Brain-ID")],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> Brain:
    """Dependency to get and validate brain from header."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        # Fallback: brain_id might be a name, not a UUID
        brain = await storage.find_brain_by_name(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    # Set brain context using the actual brain ID
    storage.set_brain(brain.id)
    return brain
