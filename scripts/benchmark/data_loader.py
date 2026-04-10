"""Dataset loader for LongMemEval benchmark.

Downloads from HuggingFace if not cached locally, then parses into typed
dataclasses for use by the benchmark pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# HuggingFace dataset repo
_HF_REPO = "xiaowu0162/longmemeval-cleaned"
_HF_REPO_TYPE = "dataset"

# Filename mapping per variant
_VARIANT_FILES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Date format used in the dataset: "2023/04/10 (Mon) 17:50"
_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    has_answer: bool = False


@dataclass
class Session:
    """A conversation session (list of turns)."""

    session_id: str
    timestamp: datetime  # naive UTC parsed from haystack_dates
    turns: list[Turn] = field(default_factory=list)


@dataclass
class LMEInstance:
    """One LongMemEval evaluation instance."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str  # raw string, e.g. "2023/04/10 (Mon) 23:07"
    sessions: list[Session] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> datetime:
    """Parse a LongMemEval date string into a naive UTC datetime.

    Format: "2023/04/10 (Mon) 17:50"
    Falls back to epoch on parse failure.
    """
    try:
        return datetime.strptime(date_str.strip(), _DATE_FORMAT)
    except ValueError:
        logger.warning("Could not parse date %r, using epoch", date_str)
        return datetime(1970, 1, 1)


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------


def _download_file(variant: str, data_dir: Path) -> Path:
    """Download dataset file from HuggingFace if not already present.

    Uses huggingface_hub.hf_hub_download (NOT the datasets library).
    """
    filename = _VARIANT_FILES[variant]
    local_path = data_dir / filename

    if local_path.exists():
        logger.info("Dataset file already exists: %s", local_path)
        return local_path

    logger.info("Downloading %s from HuggingFace repo %s ...", filename, _HF_REPO)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic dataset download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=_HF_REPO,
        filename=filename,
        repo_type=_HF_REPO_TYPE,
        local_dir=str(data_dir),
    )
    logger.info("Downloaded to %s", downloaded)
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_instance(raw: dict) -> LMEInstance:  # type: ignore[type-arg]
    """Parse one raw JSON dict into an LMEInstance."""
    question_id: str = raw["question_id"]
    question_type: str = raw["question_type"]
    question: str = raw["question"]
    answer: str = raw["answer"]
    question_date: str = raw["question_date"]

    haystack_dates: list[str] = raw.get("haystack_dates", [])
    haystack_session_ids: list[str] = raw.get("haystack_session_ids", [])
    haystack_sessions: list[list[dict]] = raw.get("haystack_sessions", [])  # type: ignore[type-arg]
    answer_session_ids: list[str] = raw.get("answer_session_ids", [])

    sessions: list[Session] = []
    for i, raw_session in enumerate(haystack_sessions):
        session_id = haystack_session_ids[i] if i < len(haystack_session_ids) else f"session_{i}"
        date_str = haystack_dates[i] if i < len(haystack_dates) else "1970/01/01 (Thu) 00:00"
        timestamp = _parse_date(date_str)

        turns: list[Turn] = []
        for turn_raw in raw_session:
            turns.append(
                Turn(
                    role=turn_raw.get("role", "user"),
                    content=turn_raw.get("content", ""),
                    has_answer=bool(turn_raw.get("has_answer", False)),
                )
            )

        sessions.append(
            Session(
                session_id=session_id,
                timestamp=timestamp,
                turns=turns,
            )
        )

    return LMEInstance(
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_date=question_date,
        sessions=sessions,
        answer_session_ids=answer_session_ids,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(variant: str, data_dir: Path) -> list[LMEInstance]:
    """Load (and download if needed) the LongMemEval dataset.

    Args:
        variant: One of "oracle", "s", "m".
        data_dir: Local directory to cache the JSON files.

    Returns:
        List of LMEInstance objects.
    """
    if variant not in _VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}. Must be one of: {list(_VARIANT_FILES)}")

    file_path = _download_file(variant, data_dir)

    logger.info("Loading dataset from %s ...", file_path)
    with file_path.open(encoding="utf-8") as f:
        raw_list: list[dict] = json.load(f)  # type: ignore[type-arg]

    instances = [_parse_instance(r) for r in raw_list]
    logger.info("Loaded %d instances (variant=%s)", len(instances), variant)
    return instances
