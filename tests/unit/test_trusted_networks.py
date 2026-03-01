"""Tests for trusted networks (Docker/container deployment support)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from neural_memory.server.dependencies import (
    _parse_trusted_networks,
    is_trusted_host,
    require_local_request,
)
from neural_memory.utils.config import Config, reset_config


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset config and LRU cache between tests."""
    reset_config()
    _parse_trusted_networks.cache_clear()


class TestIsTrustedHost:
    """Tests for is_trusted_host()."""

    def test_localhost_ipv4_trusted(self) -> None:
        assert is_trusted_host("127.0.0.1") is True

    def test_localhost_ipv6_trusted(self) -> None:
        assert is_trusted_host("::1") is True

    def test_localhost_name_trusted(self) -> None:
        assert is_trusted_host("localhost") is True

    def test_testclient_trusted(self) -> None:
        assert is_trusted_host("testclient") is True

    def test_external_ip_rejected_no_config(self) -> None:
        assert is_trusted_host("192.168.1.50") is False

    def test_external_ip_rejected_empty_config(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=[]),
        ):
            assert is_trusted_host("192.168.1.50") is False

    def test_docker_ip_trusted_with_cidr(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12"]),
        ):
            assert is_trusted_host("172.17.0.5") is True

    def test_docker_ip_outside_cidr_rejected(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12"]),
        ):
            assert is_trusted_host("10.0.0.5") is False

    def test_private_range_192_168(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["192.168.0.0/16"]),
        ):
            assert is_trusted_host("192.168.1.100") is True

    def test_multiple_cidrs(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12", "192.168.0.0/16", "10.0.0.0/8"]),
        ):
            assert is_trusted_host("10.255.0.1") is True
            assert is_trusted_host("172.31.255.255") is True
            assert is_trusted_host("192.168.99.1") is True
            assert is_trusted_host("8.8.8.8") is False

    def test_single_ip_as_cidr(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["192.168.1.50/32"]),
        ):
            assert is_trusted_host("192.168.1.50") is True
            assert is_trusted_host("192.168.1.51") is False

    def test_invalid_cidr_skipped(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["not-a-cidr", "172.16.0.0/12"]),
        ):
            # Invalid CIDR logged and skipped, valid one still works
            assert is_trusted_host("172.17.0.5") is True

    def test_invalid_client_host(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12"]),
        ):
            assert is_trusted_host("not-an-ip") is False

    def test_public_ip_rejected_even_with_private_ranges(self) -> None:
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12", "192.168.0.0/16"]),
        ):
            assert is_trusted_host("203.0.113.50") is False


class TestParseTrustedNetworks:
    """Tests for _parse_trusted_networks cache."""

    def test_empty_tuple(self) -> None:
        result = _parse_trusted_networks(())
        assert result == ()

    def test_valid_cidrs(self) -> None:
        result = _parse_trusted_networks(("10.0.0.0/8", "172.16.0.0/12"))
        assert len(result) == 2

    def test_empty_strings_skipped(self) -> None:
        result = _parse_trusted_networks(("", "10.0.0.0/8", ""))
        assert len(result) == 1

    def test_invalid_skipped_with_warning(self) -> None:
        result = _parse_trusted_networks(("garbage", "10.0.0.0/8"))
        assert len(result) == 1


class TestRequireLocalRequestWithTrustedNetworks:
    """Integration tests for require_local_request with trusted networks."""

    async def test_localhost_allowed(self) -> None:
        request = AsyncMock()
        request.client.host = "127.0.0.1"
        await require_local_request(request)  # Should not raise

    async def test_no_client_rejected(self) -> None:
        """Requests with no client info are rejected (security: unknown source)."""
        request = AsyncMock()
        request.client = None
        with pytest.raises(HTTPException) as exc_info:
            await require_local_request(request)
        assert exc_info.value.status_code == 403

    async def test_external_ip_forbidden_default(self) -> None:
        request = AsyncMock()
        request.client.host = "192.168.1.50"
        with pytest.raises(HTTPException) as exc_info:
            await require_local_request(request)
        assert exc_info.value.status_code == 403

    async def test_docker_ip_allowed_with_trusted_network(self) -> None:
        request = AsyncMock()
        request.client.host = "172.17.0.5"
        with patch(
            "neural_memory.utils.config.get_config",
            return_value=Config(trusted_networks=["172.16.0.0/12"]),
        ):
            await require_local_request(request)  # Should not raise

    async def test_docker_ip_forbidden_without_trusted_network(self) -> None:
        request = AsyncMock()
        request.client.host = "172.17.0.5"
        with pytest.raises(HTTPException) as exc_info:
            await require_local_request(request)
        assert exc_info.value.status_code == 403
