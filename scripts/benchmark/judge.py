"""LLM judge for LongMemEval benchmark.

Compares a hypothesis answer against the ground truth and returns True/False.
Implements the evaluation protocol from the LongMemEval paper.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
Given a question, a reference answer, and a hypothesis answer, determine if \
the hypothesis is correct.
The hypothesis is correct if it conveys the same essential information as the \
reference, even if worded differently.
For abstention questions, the hypothesis is correct if it indicates the \
information is not available.

Question: {question}
Reference Answer: {ground_truth}
Hypothesis: {hypothesis}

Is the hypothesis correct? Answer with just "correct" or "incorrect".\
"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseJudge(ABC):
    """Abstract judge — evaluates hypothesis correctness."""

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        hypothesis: str,
        ground_truth: str,
    ) -> bool:
        """Determine if hypothesis matches ground truth.

        Args:
            question: The evaluation question.
            hypothesis: Model-generated answer to evaluate.
            ground_truth: Reference correct answer.

        Returns:
            True if hypothesis is correct, False otherwise.
        """


# ---------------------------------------------------------------------------
# Claude judge
# ---------------------------------------------------------------------------


class ClaudeJudge(BaseJudge):
    """Judge backed by Anthropic Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._model = model
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic()
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for ClaudeJudge. "
                    "Install it with: pip install anthropic"
                ) from exc
        return self._client

    async def evaluate(
        self,
        question: str,
        hypothesis: str,
        ground_truth: str,
    ) -> bool:
        import anthropic

        client = self._get_client()
        assert isinstance(client, anthropic.AsyncAnthropic)

        prompt = _JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
        )

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            content_block = response.content[0]
            verdict = getattr(content_block, "text", str(content_block)).strip().lower()
            return verdict.startswith("correct")
        except Exception:
            logger.exception(
                "ClaudeJudge failed for question: %s", question[:80]
            )
            return False


# ---------------------------------------------------------------------------
# GPT-4o judge
# ---------------------------------------------------------------------------


class GPT4oJudge(BaseJudge):
    """Judge backed by OpenAI GPT-4o."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai

                self._client = openai.AsyncOpenAI()
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for GPT4oJudge. "
                    "Install it with: pip install openai"
                ) from exc
        return self._client

    async def evaluate(
        self,
        question: str,
        hypothesis: str,
        ground_truth: str,
    ) -> bool:
        import openai

        client = self._get_client()
        assert isinstance(client, openai.AsyncOpenAI)

        prompt = _JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
        )

        try:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = (response.choices[0].message.content or "").strip().lower()
            return verdict.startswith("correct")
        except Exception:
            logger.exception(
                "GPT4oJudge failed for question: %s", question[:80]
            )
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_judge(config: object) -> BaseJudge:
    """Create a judge from BenchmarkConfig."""
    from scripts.benchmark.config import BenchmarkConfig

    assert isinstance(config, BenchmarkConfig)

    if config.judge == "claude":
        return ClaudeJudge(model=config.claude_model)
    if config.judge == "gpt4o":
        return GPT4oJudge()

    raise ValueError(f"Unknown judge: {config.judge!r}")
