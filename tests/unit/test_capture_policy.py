"""CapturePolicy registry: one class per language, same detection results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.mcp.auto_capture import analyze_text_for_memories
from neural_memory.mcp.capture_policy import (
    CapturePolicy,
    CaptureSpec,
    ChinesePolicy,
    LatinPolicy,
    PatternMatch,
    VietnamesePolicy,
    detect_language,
    get_policy,
    iter_policies,
    register_policy,
    unregister_policy,
)


class _ProbePolicy(CapturePolicy):
    """A stand-in language. Registration alone must make analyze see it."""

    language_code = "xx"
    min_input_length = 20
    min_capture_length = 3

    def specs(self) -> tuple[CaptureSpec, ...]:
        return (
            CaptureSpec(
                memory_type="fact",
                patterns=(r"probe-marker[:\s]+(.{3,}?)(?:\.|$)",),
                confidence=0.4,
                priority=3,
                min_match_len=3,
                prefix="",
                lowercase=True,
            ),
        )


class TestLanguageSelection:
    def test_registry_order_is_english_vietnamese_chinese(self) -> None:
        assert [policy.language_code for policy in iter_policies()] == ["en", "vi", "zh"]

    def test_detect_language(self) -> None:
        assert detect_language("We decided to use PostgreSQL for the API.") == "en"
        assert detect_language("Quyết định dùng Redis thay vì Memcached cho hệ thống.") == "vi"
        assert detect_language("我们决定改用 PostgreSQL。") == "zh"
        assert detect_language("これはテストです。日本語の文章がここにあります。") == "ja"
        assert detect_language("이것은 테스트입니다. 한국어 문장이 여기에 있습니다.") == "ko"
        assert detect_language("我们决定改用 PostgreSQL の 配置") == "zh"

    def test_get_policy_falls_back_for_unsupported_scripts(self) -> None:
        assert isinstance(get_policy("We decided to use PostgreSQL today."), LatinPolicy)
        assert isinstance(
            get_policy("Quyết định dùng Redis thay vì Memcached cho hệ thống caching."),
            VietnamesePolicy,
        )
        assert isinstance(get_policy("我们决定改用 PostgreSQL。"), ChinesePolicy)
        japanese = "これはテストです。日本語の文章がここにあります。"
        assert detect_language(japanese) == "ja"
        assert isinstance(get_policy(japanese), LatinPolicy)

    def test_input_floors(self) -> None:
        assert LatinPolicy.min_input_length == 20
        assert VietnamesePolicy.min_input_length == 20
        assert ChinesePolicy.min_input_length == 8
        assert ChinesePolicy.min_capture_length == 4
        assert VietnamesePolicy.min_capture_length == 25


class TestPolicyBehavior:
    def test_pattern_match_is_frozen(self) -> None:
        match = PatternMatch("todo", "update docs", 0.75, 5)
        with pytest.raises(FrozenInstanceError):
            match.content = "changed"  # type: ignore[misc]

    def test_vietnamese_quality_gate_penalizes_or_rejects(self) -> None:
        policy = VietnamesePolicy()
        assert policy.quality_gate("của này trong để các những") == 0.0
        assert policy.quality_gate("dùng Redis caching cho hệ thống") == pytest.approx(0.55)

    def test_chinese_policy_finds_todo_latin_policy_does_not(self) -> None:
        text = "待办：更新部署文档，并在周五前同步给团队。"
        assert ChinesePolicy().detect_patterns(text, capture_decisions=False)[0].memory_type == (
            "todo"
        )
        assert LatinPolicy().detect_patterns(text) == []

    def test_mixed_sentence_keeps_both_languages(self) -> None:
        text = (
            "We decided to switch the database from SQLite to PostgreSQL because the API "
            "needs concurrent writes.\n待办：更新部署文档，并通知团队。"
        )
        types = {str(item["type"]) for item in analyze_text_for_memories(text)}
        assert "decision" in types
        assert "todo" in types

    def test_registering_a_policy_is_enough(self) -> None:
        policy = _ProbePolicy()
        register_policy(policy)
        text = "probe-marker: keep this phrase intact for the registry."
        try:
            detected = analyze_text_for_memories(
                text,
                capture_decisions=False,
                capture_errors=False,
                capture_todos=False,
                capture_insights=False,
                capture_preferences=False,
            )
            assert any("keep this phrase" in str(item["content"]) for item in detected)
        finally:
            unregister_policy("xx")
        assert [item.language_code for item in iter_policies()] == ["en", "vi", "zh"]
        detected_after = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_errors=False,
            capture_todos=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert not any("keep this phrase" in str(item["content"]) for item in detected_after)
