"""LLM-based dedup judge for Tier 3 borderline cases."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from neural_memory.engine.dedup.prompts import DEDUP_SYSTEM_PROMPT, format_dedup_prompt

logger = logging.getLogger(__name__)


class DedupVerdict(StrEnum):
    """Verdict from LLM dedup judgment."""

    DUPLICATE = "duplicate"
    DISTINCT = "distinct"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class DedupJudgment:
    """Result of an LLM dedup judgment."""

    verdict: DedupVerdict
    reason: str
    confidence: float = 0.5


class LLMDedupJudge(ABC):
    """Abstract base class for LLM dedup judges."""

    @abstractmethod
    async def judge(self, content_a: str, content_b: str) -> DedupJudgment:
        """Judge whether two content strings are duplicates.

        Args:
            content_a: First content string.
            content_b: Second content string.

        Returns:
            DedupJudgment with verdict and reason.
        """
        ...


def _parse_verdict(response: str) -> DedupJudgment:
    """Parse LLM response into a DedupJudgment."""
    lines = response.strip().split("\n", 1)
    verdict_line = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""

    # Check DISTINCT first to avoid "NOT A DUPLICATE" matching DUPLICATE
    if "DISTINCT" in verdict_line:
        return DedupJudgment(
            verdict=DedupVerdict.DISTINCT,
            reason=reason,
            confidence=0.8,
        )
    if "DUPLICATE" in verdict_line:
        return DedupJudgment(
            verdict=DedupVerdict.DUPLICATE,
            reason=reason,
            confidence=0.8,
        )
    return DedupJudgment(
        verdict=DedupVerdict.UNCERTAIN,
        reason=reason or "Could not determine",
        confidence=0.3,
    )


class OpenAIDedupJudge(LLMDedupJudge):
    """OpenAI-based dedup judge."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "") -> None:
        self._model = model
        self._api_key = api_key

    async def judge(self, content_a: str, content_b: str) -> DedupJudgment:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self._api_key or None)
            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                    {"role": "user", "content": format_dedup_prompt(content_a, content_b)},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            return _parse_verdict(text)
        except Exception as e:
            logger.warning("OpenAI dedup judge failed: %s", e)
            return DedupJudgment(
                verdict=DedupVerdict.UNCERTAIN,
                reason="LLM call failed",
                confidence=0.0,
            )


class AnthropicDedupJudge(LLMDedupJudge):
    """Anthropic-based dedup judge."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: str = "") -> None:
        self._model = model
        self._api_key = api_key

    async def judge(self, content_a: str, content_b: str) -> DedupJudgment:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self._api_key or None)
            response = await client.messages.create(
                model=self._model,
                system=DEDUP_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": format_dedup_prompt(content_a, content_b)},
                ],
                max_tokens=100,
            )
            first_block = response.content[0] if response.content else None
            text = first_block.text if first_block and hasattr(first_block, "text") else ""
            return _parse_verdict(text)
        except Exception as e:
            logger.warning("Anthropic dedup judge failed: %s", e)
            return DedupJudgment(
                verdict=DedupVerdict.UNCERTAIN,
                reason="LLM call failed",
                confidence=0.0,
            )


def create_judge(provider: str, model: str, api_key: str = "") -> LLMDedupJudge | None:
    """Factory to create LLM dedup judge from config.

    Args:
        provider: "openai" or "anthropic" or "none".
        model: Model name.
        api_key: Optional API key override.

    Returns:
        LLMDedupJudge instance or None if provider is "none".
    """
    if provider == "openai":
        return OpenAIDedupJudge(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicDedupJudge(model=model, api_key=api_key)
    return None
