"""Tests for brain access control enforcement (Phase 2).

Covers:
- check_read_access / check_write_access with enforce=False (default)
- Owner access
- Shared user access
- Public brain access
- Anonymous access
- Legacy unowned brain access
- AccessDeniedError exception
- require_read / require_write helpers
- Config enforce_brain_acl flag
"""

from __future__ import annotations

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.engine.brain_acl import (
    AccessDeniedError,
    check_read_access,
    check_write_access,
    require_read,
    require_write,
)
from neural_memory.utils.timeutils import utcnow


def _make_brain(
    owner_id: str | None = "owner-1",
    is_public: bool = False,
    shared_with: list[str] | None = None,
) -> Brain:
    now = utcnow()
    return Brain(
        id="brain-test",
        name="Test Brain",
        config=BrainConfig(),
        owner_id=owner_id,
        is_public=is_public,
        shared_with=shared_with or [],
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# enforce=False (default) — everything allowed
# ---------------------------------------------------------------------------


class TestACLDisabled:
    def test_read_always_allowed(self) -> None:
        brain = _make_brain()
        assert check_read_access(brain, None, enforce=False) is True
        assert check_read_access(brain, "stranger", enforce=False) is True

    def test_write_always_allowed(self) -> None:
        brain = _make_brain()
        assert check_write_access(brain, None, enforce=False) is True
        assert check_write_access(brain, "stranger", enforce=False) is True


# ---------------------------------------------------------------------------
# enforce=True — read access
# ---------------------------------------------------------------------------


class TestReadAccessEnforced:
    def test_owner_can_read(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_read_access(brain, "alice", enforce=True) is True

    def test_shared_user_can_read(self) -> None:
        brain = _make_brain(owner_id="alice", shared_with=["bob", "charlie"])
        assert check_read_access(brain, "bob", enforce=True) is True

    def test_public_brain_anyone_can_read(self) -> None:
        brain = _make_brain(owner_id="alice", is_public=True)
        assert check_read_access(brain, "stranger", enforce=True) is True
        assert check_read_access(brain, None, enforce=True) is True

    def test_stranger_denied(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_read_access(brain, "stranger", enforce=True) is False

    def test_anonymous_denied_on_owned_brain(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_read_access(brain, None, enforce=True) is False

    def test_legacy_unowned_brain_anyone_can_read(self) -> None:
        """Brains with no owner_id (pre-ACL) are accessible by all."""
        brain = _make_brain(owner_id=None)
        assert check_read_access(brain, None, enforce=True) is True
        assert check_read_access(brain, "anyone", enforce=True) is True


# ---------------------------------------------------------------------------
# enforce=True — write access
# ---------------------------------------------------------------------------


class TestWriteAccessEnforced:
    def test_owner_can_write(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_write_access(brain, "alice", enforce=True) is True

    def test_shared_user_cannot_write(self) -> None:
        """Shared users have read access only, not write."""
        brain = _make_brain(owner_id="alice", shared_with=["bob"])
        assert check_write_access(brain, "bob", enforce=True) is False

    def test_stranger_cannot_write(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_write_access(brain, "stranger", enforce=True) is False

    def test_anonymous_cannot_write(self) -> None:
        brain = _make_brain(owner_id="alice")
        assert check_write_access(brain, None, enforce=True) is False

    def test_legacy_unowned_anyone_can_write(self) -> None:
        brain = _make_brain(owner_id=None)
        assert check_write_access(brain, None, enforce=True) is True
        assert check_write_access(brain, "anyone", enforce=True) is True

    def test_public_brain_only_owner_writes(self) -> None:
        """Public brains are readable by all but writable only by owner."""
        brain = _make_brain(owner_id="alice", is_public=True)
        assert check_write_access(brain, "stranger", enforce=True) is False
        assert check_write_access(brain, "alice", enforce=True) is True


# ---------------------------------------------------------------------------
# require_read / require_write helpers
# ---------------------------------------------------------------------------


class TestRequireHelpers:
    def test_require_read_passes(self) -> None:
        brain = _make_brain(owner_id="alice")
        require_read(brain, "alice", enforce=True)  # Should not raise

    def test_require_read_raises(self) -> None:
        brain = _make_brain(owner_id="alice")
        with pytest.raises(AccessDeniedError) as exc_info:
            require_read(brain, "stranger", enforce=True)
        assert exc_info.value.action == "read"
        assert exc_info.value.user_id == "stranger"

    def test_require_write_raises(self) -> None:
        brain = _make_brain(owner_id="alice", shared_with=["bob"])
        with pytest.raises(AccessDeniedError) as exc_info:
            require_write(brain, "bob", enforce=True)
        assert exc_info.value.action == "write"

    def test_require_noop_when_disabled(self) -> None:
        brain = _make_brain(owner_id="alice")
        require_read(brain, None, enforce=False)  # No raise
        require_write(brain, None, enforce=False)  # No raise


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------


class TestConfigFlag:
    def test_default_acl_disabled(self) -> None:
        from neural_memory.utils.config import Config

        config = Config()
        assert config.enforce_brain_acl is False

    def test_env_var_enables_acl(self) -> None:
        import os

        from neural_memory.utils.config import Config

        os.environ["NEURAL_MEMORY_ENFORCE_ACL"] = "true"
        try:
            config = Config.from_env()
            assert config.enforce_brain_acl is True
        finally:
            del os.environ["NEURAL_MEMORY_ENFORCE_ACL"]
