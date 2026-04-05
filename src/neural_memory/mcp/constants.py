"""Shared constants for MCP handlers."""

# Maximum content length per field — prevents memory exhaustion.
MAX_CONTENT_LENGTH = 100_000

# Maximum token budget for recall/context assembly.
MAX_TOKEN_BUDGET = 100_000

# Batch remember limits.
MAX_BATCH_SIZE = 20
MAX_BATCH_TOTAL_CHARS = 500_000

# Maximum HOT-tier memories injected into context.
MAX_HOT_CONTEXT_MEMORIES = 50

# Maximum topic/tag length for cognitive tools.
MAX_TOPIC_LENGTH = 500
MAX_TAG_LENGTH = 100

# Default and ceiling for list queries.
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100

# Content preview truncation lengths.
PREVIEW_SHORT = 120
PREVIEW_MEDIUM = 200
