"""Safety utilities for Neural Memory.

This module provides tools for:
- Sensitive content detection
- Memory freshness evaluation
- Privacy protection
"""

from neural_memory.safety.freshness import (
    FreshnessLevel,
    evaluate_freshness,
    get_freshness_warning,
)
from neural_memory.safety.sensitive import (
    SensitiveMatch,
    SensitivePattern,
    check_sensitive_content,
    filter_sensitive_content,
    get_default_patterns,
)
from neural_memory.safety.encryption import (
    EncryptionResult,
    MemoryEncryptor,
)

__all__ = [
    "EncryptionResult",
    "FreshnessLevel",
    "MemoryEncryptor",
    "SensitiveMatch",
    "SensitivePattern",
    "check_sensitive_content",
    "evaluate_freshness",
    "filter_sensitive_content",
    "get_default_patterns",
    "get_freshness_warning",
]
