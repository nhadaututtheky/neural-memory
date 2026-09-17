"""Tests for Chinese (CJK) auto-capture support.

Regression cover for the failure where Chinese text silently produced zero
detected memories: every pattern group was English + Vietnamese only, so the
extractor returned "No memorable content detected" for any non-Latin,
non-Vietnamese input.
"""

from __future__ import annotations

import pytest

from neural_memory.core.trigger_engine import TriggerType, check_triggers
from neural_memory.mcp.auto_capture import (
    _cjk_script,
    analyze_text_for_memories,
    empty_capture_hint,
)

ZH_FULL = (
    "我们决定把数据库从 SQLite 换成 PostgreSQL，因为 API 需要并发写入。"
    "迁移花了两天。待办：更新部署文档。"
    "错误：构建失败，因为没装 tsc。"
)

EN_FULL = (
    "We decided to switch the database from SQLite to PostgreSQL because the API "
    "needs concurrent writes. The migration took two days. TODO: update the "
    "deployment docs. Error: the build failed because tsc was not installed."
)


def _types(detected: list[dict[str, object]]) -> list[str]:
    return [str(item["type"]) for item in detected]


class TestChineseExtraction:
    """Chinese text must produce memories, not an empty result."""

    def test_chinese_text_is_not_silently_dropped(self) -> None:
        detected = analyze_text_for_memories(ZH_FULL)
        assert detected, "Chinese input must not silently yield zero memories"

    def test_chinese_decision_detected(self) -> None:
        assert "decision" in _types(analyze_text_for_memories(ZH_FULL))

    def test_chinese_error_detected(self) -> None:
        assert "error" in _types(analyze_text_for_memories(ZH_FULL))

    def test_chinese_todo_detected(self) -> None:
        assert "todo" in _types(analyze_text_for_memories(ZH_FULL))

    def test_chinese_fix_detected(self) -> None:
        text = "错误：构建失败，原因是没装 tsc。已修复：安装 typescript 依赖后通过。"
        contents = [str(item["content"]) for item in analyze_text_for_memories(text)]
        # The fix clause must produce its own capture — asserting only that an
        # "error" exists would still pass if the 已修复 pattern were deleted.
        assert any("安装 typescript" in c for c in contents), contents

    def test_short_chinese_capture_keeps_full_confidence(self) -> None:
        """One Han character is roughly one word — the Latin 'too short' floor
        (10 chars) must not be applied to Chinese captures."""
        detected = analyze_text_for_memories(
            ZH_FULL,
            capture_decisions=False,
            capture_errors=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        todos = [item for item in detected if item["type"] == "todo"]
        assert todos
        for item in todos:
            assert item["confidence"] == pytest.approx(0.75)

    def test_mid_sentence_verb_is_not_a_todo(self) -> None:
        """'因为 API 需要并发写入' is a statement, not an action item."""
        text = "我们决定改用 PostgreSQL，因为需要并发写入，性能更好。"
        detected = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_errors=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert detected == []

    def test_sentence_initial_todo_is_captured(self) -> None:
        text = "上次讨论的结论是先不动。待办：更新部署文档，并在周五前同步给团队。"
        detected = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_errors=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert "todo" in _types(detected)

    def test_chinese_preference_detected(self) -> None:
        text = "我们希望所有新项目都使用 PostgreSQL 而不是 MySQL 数据库。"
        detected = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_errors=False,
            capture_todos=False,
            capture_facts=False,
            capture_insights=False,
        )
        assert "preference" in _types(detected)


class TestShortChineseInput:
    """A short Chinese sentence is a complete thought. The Latin minimum-input
    floor (20 chars) used to discard it before detection even ran, so a
    10-character todo line returned zero memories."""

    def test_short_todo_is_captured(self) -> None:
        text = "待办：更新部署文档。"
        assert len(text) < 20, "fixture must stay under the Latin floor to be meaningful"
        detected = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_errors=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert "todo" in _types(detected)

    def test_short_error_is_captured(self) -> None:
        text = "错误：构建失败，原因是没装 tsc。"
        assert len(text) < 20
        detected = analyze_text_for_memories(
            text,
            capture_decisions=False,
            capture_todos=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert "error" in _types(detected)

    def test_short_decision_is_captured(self) -> None:
        text = "我们决定改用 PostgreSQL。"
        assert len(text) < 20
        detected = analyze_text_for_memories(
            text,
            capture_errors=False,
            capture_todos=False,
            capture_facts=False,
            capture_insights=False,
            capture_preferences=False,
        )
        assert "decision" in _types(detected)

    def test_english_short_input_still_rejected(self) -> None:
        """The Latin floor is unchanged by the CJK allowance."""
        assert analyze_text_for_memories("todo: fix it") == []

    def test_very_short_chinese_is_still_rejected(self) -> None:
        """Below the CJK floor (8 chars) we still skip, guarding against noise."""
        assert analyze_text_for_memories("待办：更新") == []


class TestEnglishRegression:
    """The English corpus must keep behaving exactly as before."""

    def test_english_sample_still_yields_four(self) -> None:
        detected = analyze_text_for_memories(EN_FULL)
        assert len(detected) == 4

    def test_english_sample_types_unchanged(self) -> None:
        assert sorted(_types(analyze_text_for_memories(EN_FULL))) == [
            "decision",
            "error",
            "error",
            "todo",
        ]

    def test_english_confidences_unchanged(self) -> None:
        detected = analyze_text_for_memories(EN_FULL)
        confidences = sorted(round(float(item["confidence"]), 3) for item in detected)
        assert confidences == [0.75, 0.8, 0.85, 0.85]


class TestUnsupportedLanguageHint:
    """An empty result must never be indistinguishable from a language the
    extractor cannot read."""

    def test_english_has_no_hint(self) -> None:
        assert empty_capture_hint("plain english sentence with nothing special") == ""

    def test_short_input_is_reported_as_length_not_missing_trigger(self) -> None:
        """A short todo line already contains its trigger word — blaming a
        missing trigger would send the user down a dead end."""
        hint = empty_capture_hint("待办：更新")
        assert "below" in hint
        assert "trigger word" not in hint

    def test_short_english_input_is_also_explained(self) -> None:
        hint = empty_capture_hint("todo now")
        assert "below" in hint

    def test_chinese_hint_lists_triggers(self) -> None:
        hint = empty_capture_hint("随便写点什么内容，但是没有任何触发词存在")
        assert "Chinese is supported" in hint

    def test_japanese_reports_unsupported(self) -> None:
        text = "これはテストです。日本語の文章がここにあります。"
        assert analyze_text_for_memories(text) == []
        assert "not yet supported" in empty_capture_hint(text)

    def test_korean_reports_unsupported(self) -> None:
        text = "이것은 테스트입니다. 한국어 문장이 여기에 있습니다."
        assert analyze_text_for_memories(text) == []
        assert "not yet supported" in empty_capture_hint(text)

    def test_script_classification(self) -> None:
        assert _cjk_script("我们决定改用 PostgreSQL") == "chinese"
        assert _cjk_script("これはテストです") == "japanese"
        assert _cjk_script("이것은 테스트입니다") == "korean"
        assert _cjk_script("plain english") is None


class TestChineseTriggers:
    """Auto-save triggers must fire on Chinese equivalents of the English
    phrases, without firing on neutral prose."""

    def test_decision_trigger(self) -> None:
        result = check_triggers("我们决定改用 PostgreSQL 数据库来支撑并发写入")
        assert result.triggered
        assert result.trigger_type == TriggerType.DECISION_MADE

    def test_milestone_trigger(self) -> None:
        result = check_triggers("这个功能完成了，已经合并到主干分支")
        assert result.triggered
        assert result.trigger_type == TriggerType.WORKFLOW_END

    def test_error_fixed_trigger(self) -> None:
        result = check_triggers("这个 bug 已修复，不再报错了")
        assert result.triggered
        assert result.trigger_type == TriggerType.ERROR_FIXED

    def test_user_leaving_trigger(self) -> None:
        result = check_triggers("好的先这样，我下线了，下次再聊")
        assert result.triggered
        assert result.trigger_type == TriggerType.USER_LEAVING

    def test_neutral_chinese_does_not_trigger(self) -> None:
        result = check_triggers("今天天气不错，我们去公园散步吧")
        assert not result.triggered

    def test_english_trigger_regression(self) -> None:
        result = check_triggers("We decided to use PostgreSQL for the storage layer")
        assert result.trigger_type == TriggerType.DECISION_MADE


class TestChineseOverCapture:
    """Ordinary Chinese prose must not be captured.

    An earlier revision accepted descriptive verbs (采用/确定/改用/切换到), which
    auto-saved 12 of these 15 neutral sentences at or above the passive-write
    gate of 0.75 — e.g. the sentence ending in 确定了 captured the trailing
    fragment that followed it as a decision. The patterns now require an
    explicit decision or action marker.
    """

    NEUTRAL = [
        "我们采用 PostgreSQL 作为主库，性能更好。",
        "我们确定用 Redis 做缓存层。",
        "改用 SQLite 会让部署更简单。",
        "这个方案确定了，下周开始动手。",
        "我们把存储切换到 PostgreSQL 之后稳定多了。",
        "需要连接池来扁住并发。",
        "错误处理这块还需要再想想。",
        "发现一个问题，暂时不影响使用。",
        "记得当时讨论过这个点。",
        "性能问题主要出在序列化上。",
        "原来是这样，难怪会慢。",
        "解决方案还在评估中。",
    ]

    @pytest.mark.parametrize("text", NEUTRAL)
    def test_neutral_prose_yields_nothing(self, text: str) -> None:
        assert analyze_text_for_memories(text) == []

    def test_capture_does_not_keep_trailing_particle(self) -> None:
        """The trailing particle 了 must not be stored."""
        detected = analyze_text_for_memories(
            "以后不要用同步 IO 了。",
            capture_decisions=False,
            capture_errors=False,
            capture_todos=False,
            capture_facts=False,
            capture_insights=False,
        )
        contents = [item["content"] for item in detected]
        assert contents, "a prohibition is a legitimate preference"
        assert all(not c.endswith("了") for c in contents), contents
