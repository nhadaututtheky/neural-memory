"""Tests for brain mode configuration."""

import pytest

from neural_memory.core.brain_mode import (
    BrainMode,
    BrainModeConfig,
    HybridConfig,
    SharedConfig,
    SyncStrategy,
)


class TestBrainMode:
    """Tests for BrainMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert BrainMode.LOCAL == "local"
        assert BrainMode.SHARED == "shared"
        assert BrainMode.HYBRID == "hybrid"

    def test_mode_from_string(self) -> None:
        """Test creating mode from string."""
        assert BrainMode("local") == BrainMode.LOCAL
        assert BrainMode("shared") == BrainMode.SHARED
        assert BrainMode("hybrid") == BrainMode.HYBRID


class TestSyncStrategy:
    """Tests for SyncStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test strategy enum values."""
        assert SyncStrategy.PUSH_ONLY == "push_only"
        assert SyncStrategy.PULL_ONLY == "pull_only"
        assert SyncStrategy.BIDIRECTIONAL == "bidirectional"
        assert SyncStrategy.ON_DEMAND == "on_demand"


class TestSharedConfig:
    """Tests for SharedConfig dataclass."""

    def test_create_shared_config(self) -> None:
        """Test creating shared config."""
        config = SharedConfig(
            server_url="http://localhost:8000",
            api_key="test-key",
            timeout=60.0,
        )

        assert config.server_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.retry_count == 3
        assert config.retry_delay == 1.0

    def test_shared_config_defaults(self) -> None:
        """Test shared config default values."""
        config = SharedConfig(server_url="http://localhost:8000")

        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.retry_count == 3
        assert config.retry_delay == 1.0

    def test_shared_config_immutable(self) -> None:
        """Test that shared config is immutable."""
        config = SharedConfig(server_url="http://localhost:8000")

        with pytest.raises(AttributeError):
            config.server_url = "http://other:8000"  # type: ignore

    def test_with_server_url(self) -> None:
        """Test creating new config with different URL."""
        config = SharedConfig(server_url="http://localhost:8000", api_key="key")
        new_config = config.with_server_url("http://other:9000")

        assert new_config.server_url == "http://other:9000"
        assert new_config.api_key == "key"
        assert config.server_url == "http://localhost:8000"

    def test_with_api_key(self) -> None:
        """Test creating new config with different API key."""
        config = SharedConfig(server_url="http://localhost:8000")
        new_config = config.with_api_key("new-key")

        assert new_config.api_key == "new-key"
        assert config.api_key is None


class TestHybridConfig:
    """Tests for HybridConfig dataclass."""

    def test_create_hybrid_config(self) -> None:
        """Test creating hybrid config."""
        config = HybridConfig(
            local_path="./brain.db",
            server_url="http://localhost:8000",
            api_key="test-key",
            sync_strategy=SyncStrategy.PUSH_ONLY,
            sync_interval_seconds=120,
        )

        assert config.local_path == "./brain.db"
        assert config.server_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.sync_strategy == SyncStrategy.PUSH_ONLY
        assert config.sync_interval_seconds == 120

    def test_hybrid_config_defaults(self) -> None:
        """Test hybrid config default values."""
        config = HybridConfig(
            local_path="./brain.db",
            server_url="http://localhost:8000",
        )

        assert config.api_key is None
        assert config.sync_strategy == SyncStrategy.BIDIRECTIONAL
        assert config.sync_interval_seconds == 60
        assert config.auto_sync_on_encode is True
        assert config.auto_sync_on_query is False
        assert config.conflict_resolution == "prefer_local"


class TestBrainModeConfig:
    """Tests for BrainModeConfig dataclass."""

    def test_create_local_config(self) -> None:
        """Test creating local-only config."""
        config = BrainModeConfig(mode=BrainMode.LOCAL)

        assert config.mode == BrainMode.LOCAL
        assert config.shared is None
        assert config.hybrid is None

    def test_local_factory(self) -> None:
        """Test local factory method."""
        config = BrainModeConfig.local()

        assert config.mode == BrainMode.LOCAL
        assert config.is_local() is True
        assert config.is_shared() is False
        assert config.is_hybrid() is False

    def test_shared_mode_factory(self) -> None:
        """Test shared_mode factory method."""
        config = BrainModeConfig.shared_mode(
            server_url="http://localhost:8000",
            api_key="test-key",
            timeout=45.0,
        )

        assert config.mode == BrainMode.SHARED
        assert config.shared is not None
        assert config.shared.server_url == "http://localhost:8000"
        assert config.shared.api_key == "test-key"
        assert config.shared.timeout == 45.0
        assert config.is_shared() is True

    def test_hybrid_mode_factory(self) -> None:
        """Test hybrid_mode factory method."""
        config = BrainModeConfig.hybrid_mode(
            local_path="./brain.db",
            server_url="http://localhost:8000",
            api_key="test-key",
            sync_interval=300,
            strategy=SyncStrategy.PUSH_ONLY,
        )

        assert config.mode == BrainMode.HYBRID
        assert config.hybrid is not None
        assert config.hybrid.local_path == "./brain.db"
        assert config.hybrid.server_url == "http://localhost:8000"
        assert config.hybrid.sync_interval_seconds == 300
        assert config.hybrid.sync_strategy == SyncStrategy.PUSH_ONLY
        assert config.is_hybrid() is True

    def test_to_dict_local(self) -> None:
        """Test serialization of local config."""
        config = BrainModeConfig.local()
        data = config.to_dict()

        assert data == {"mode": "local"}

    def test_to_dict_shared(self) -> None:
        """Test serialization of shared config."""
        config = BrainModeConfig.shared_mode(
            server_url="http://localhost:8000",
            api_key="key",
        )
        data = config.to_dict()

        assert data["mode"] == "shared"
        assert data["shared"]["server_url"] == "http://localhost:8000"
        assert data["shared"]["api_key"] == "key"

    def test_to_dict_hybrid(self) -> None:
        """Test serialization of hybrid config."""
        config = BrainModeConfig.hybrid_mode(
            local_path="./brain.db",
            server_url="http://localhost:8000",
        )
        data = config.to_dict()

        assert data["mode"] == "hybrid"
        assert data["hybrid"]["local_path"] == "./brain.db"
        assert data["hybrid"]["server_url"] == "http://localhost:8000"

    def test_from_dict_local(self) -> None:
        """Test deserialization of local config."""
        data = {"mode": "local"}
        config = BrainModeConfig.from_dict(data)

        assert config.mode == BrainMode.LOCAL
        assert config.shared is None

    def test_from_dict_shared(self) -> None:
        """Test deserialization of shared config."""
        data = {
            "mode": "shared",
            "shared": {
                "server_url": "http://localhost:8000",
                "api_key": "key",
                "timeout": 45.0,
            },
        }
        config = BrainModeConfig.from_dict(data)

        assert config.mode == BrainMode.SHARED
        assert config.shared is not None
        assert config.shared.server_url == "http://localhost:8000"
        assert config.shared.api_key == "key"
        assert config.shared.timeout == 45.0

    def test_from_dict_hybrid(self) -> None:
        """Test deserialization of hybrid config."""
        data = {
            "mode": "hybrid",
            "hybrid": {
                "local_path": "./brain.db",
                "server_url": "http://localhost:8000",
                "sync_strategy": "push_only",
                "sync_interval_seconds": 120,
            },
        }
        config = BrainModeConfig.from_dict(data)

        assert config.mode == BrainMode.HYBRID
        assert config.hybrid is not None
        assert config.hybrid.local_path == "./brain.db"
        assert config.hybrid.sync_strategy == SyncStrategy.PUSH_ONLY
        assert config.hybrid.sync_interval_seconds == 120

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization/deserialization roundtrips correctly."""
        original = BrainModeConfig.hybrid_mode(
            local_path="./test.db",
            server_url="http://server:8000",
            api_key="secret",
            sync_interval=180,
            strategy=SyncStrategy.BIDIRECTIONAL,
        )

        data = original.to_dict()
        restored = BrainModeConfig.from_dict(data)

        assert restored.mode == original.mode
        assert restored.hybrid is not None
        assert restored.hybrid.local_path == original.hybrid.local_path
        assert restored.hybrid.server_url == original.hybrid.server_url
        assert restored.hybrid.api_key == original.hybrid.api_key
        assert restored.hybrid.sync_interval_seconds == original.hybrid.sync_interval_seconds
