"""Brain access control enforcement.

Opt-in ACL layer that checks owner_id, shared_with, and is_public
on Brain objects before allowing read/write operations.

When enforce_brain_acl is False (default), all checks pass — zero
behavior change for single-user deployments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.brain import Brain

logger = logging.getLogger(__name__)


class AccessDeniedError(Exception):
    """Raised when a user lacks permission to access a brain."""

    def __init__(self, user_id: str | None, brain_id: str, action: str = "access") -> None:
        self.user_id = user_id
        self.brain_id = brain_id
        self.action = action
        super().__init__(f"User {user_id!r} cannot {action} brain {brain_id[:12]}")


def check_read_access(
    brain: Brain,
    user_id: str | None,
    enforce: bool = False,
) -> bool:
    """Check if user can read from this brain.

    Args:
        brain: The brain to check access for.
        user_id: Requesting user/agent ID (None = anonymous).
        enforce: Whether ACL is active. When False, always returns True.

    Returns:
        True if access is allowed.
    """
    if not enforce:
        return True

    # Public brains are readable by everyone
    if brain.is_public:
        return True

    # Unowned brains (legacy) are accessible by all
    if brain.owner_id is None:
        return True

    # Owner always has access
    if user_id is not None and brain.owner_id == user_id:
        return True

    # Shared users have read access
    if user_id is not None and user_id in brain.shared_with:
        return True

    # Anonymous user on non-public, owned brain → denied
    return False


def check_write_access(
    brain: Brain,
    user_id: str | None,
    enforce: bool = False,
) -> bool:
    """Check if user can write to this brain.

    Args:
        brain: The brain to check access for.
        user_id: Requesting user/agent ID (None = anonymous).
        enforce: Whether ACL is active. When False, always returns True.

    Returns:
        True if write access is allowed.
    """
    if not enforce:
        return True

    # Unowned brains (legacy) are writable by all
    if brain.owner_id is None:
        return True

    # Only owner can write
    if user_id is not None and brain.owner_id == user_id:
        return True

    return False


def require_read(
    brain: Brain,
    user_id: str | None,
    enforce: bool = False,
) -> None:
    """Raise AccessDeniedError if user cannot read this brain."""
    if not check_read_access(brain, user_id, enforce):
        raise AccessDeniedError(user_id, brain.id, "read")


def require_write(
    brain: Brain,
    user_id: str | None,
    enforce: bool = False,
) -> None:
    """Raise AccessDeniedError if user cannot write to this brain."""
    if not check_write_access(brain, user_id, enforce):
        raise AccessDeniedError(user_id, brain.id, "write")
