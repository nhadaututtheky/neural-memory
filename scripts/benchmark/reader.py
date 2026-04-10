"""LLM reader interface for LongMemEval benchmark.

The reader takes retrieved NM context + the question and generates a hypothesis
answer. Different backend LLMs are supported.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are answering questions about a user's past conversations with an AI assistant.\n"
    "Use ONLY the provided conversation history to answer. "
    "If the information is not in the history, say \"I don't have that information.\"\n"
    "Be concise and direct. The current date is {question_date}."
)

_USER_TEMPLATE = (
    "Conversation history:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseReader(ABC):
    """Abstract reader — takes NM context + question, returns hypothesis."""

    @abstractmethod
    async def answer(self, question: str, context: str, question_date: str) -> str:
        """Generate a hypothesis answer.

        Args:
            question: The evaluation question.
            context: Retrieved conversation context from Neural Memory.
            question_date: The question date string (shown to the reader as "today").

        Returns:
            Hypothesis answer string.
        """


# ---------------------------------------------------------------------------
# Claude reader
# ---------------------------------------------------------------------------


class ClaudeReader(BaseReader):
    """Reader backed by Anthropic Claude."""

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
                    "anthropic package is required for ClaudeReader. "
                    "Install it with: pip install anthropic"
                ) from exc
        return self._client

    async def answer(self, question: str, context: str, question_date: str) -> str:
        import anthropic

        client = self._get_client()
        assert isinstance(client, anthropic.AsyncAnthropic)

        system = _SYSTEM_PROMPT.format(question_date=question_date)
        user_msg = _USER_TEMPLATE.format(context=context, question=question)

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            content_block = response.content[0]
            if hasattr(content_block, "text"):
                return content_block.text.strip()
            return str(content_block)
        except Exception:
            logger.exception("ClaudeReader failed for question: %s", question[:80])
            return "I don't have that information."


# ---------------------------------------------------------------------------
# Ollama reader (generic — works with gemma3, llama3, etc.)
# ---------------------------------------------------------------------------


class OllamaReader(BaseReader):
    """Reader backed by a local Ollama model."""

    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def answer(self, question: str, context: str, question_date: str) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for OllamaReader. "
                "Install it with: pip install httpx"
            ) from exc

        system = _SYSTEM_PROMPT.format(question_date=question_date)
        user_msg = _USER_TEMPLATE.format(context=context, question=question)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"num_predict": 512},
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"].strip()
        except Exception:
            logger.exception("OllamaReader failed for question: %s", question[:80])
            return "I don't have that information."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_reader(config: object) -> BaseReader:
    """Create a reader from BenchmarkConfig."""
    from scripts.benchmark.config import BenchmarkConfig

    assert isinstance(config, BenchmarkConfig)

    if config.reader == "claude":
        return ClaudeReader(model=config.claude_model)
    if config.reader in ("gemma4", "ollama"):
        model = config.gemma_model if config.reader == "gemma4" else config.gemma_model
        return OllamaReader(model=model, base_url=config.ollama_url)

    raise ValueError(f"Unknown reader: {config.reader!r}")
