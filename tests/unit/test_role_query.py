"""Tests for role-aware query detection."""

from __future__ import annotations

from neural_memory.engine.role_query import RoleTarget, detect_role_target

# ---------------------------------------------------------------------------
# detect_role_target — assistant patterns
# ---------------------------------------------------------------------------


class TestDetectRoleTargetAssistant:
    """Test detection of queries targeting assistant content."""

    def test_you_said(self) -> None:
        assert detect_role_target("What did you say about Python?") == RoleTarget.ASSISTANT

    def test_you_recommended(self) -> None:
        assert detect_role_target("You recommended a framework, which one?") == RoleTarget.ASSISTANT

    def test_you_suggested(self) -> None:
        assert detect_role_target("You suggested I try Docker") == RoleTarget.ASSISTANT

    def test_your_recommendation(self) -> None:
        assert detect_role_target("What was your recommendation?") == RoleTarget.ASSISTANT

    def test_your_advice(self) -> None:
        assert (
            detect_role_target("Can you remind me of your advice on testing?")
            == RoleTarget.ASSISTANT
        )

    def test_what_did_you_tell(self) -> None:
        assert detect_role_target("What did you tell me about React?") == RoleTarget.ASSISTANT

    def test_can_you_remind_me(self) -> None:
        assert (
            detect_role_target("Can you remind me what you said about databases?")
            == RoleTarget.ASSISTANT
        )

    def test_you_mentioned(self) -> None:
        assert detect_role_target("You mentioned a tool for deployment") == RoleTarget.ASSISTANT

    def test_answer_you_gave(self) -> None:
        assert detect_role_target("The recommendation you gave about IDEs") == RoleTarget.ASSISTANT


# ---------------------------------------------------------------------------
# detect_role_target — user patterns
# ---------------------------------------------------------------------------


class TestDetectRoleTargetUser:
    """Test detection of queries targeting user content."""

    def test_i_said(self) -> None:
        assert detect_role_target("What did I say about the meeting?") == RoleTarget.USER

    def test_i_mentioned(self) -> None:
        assert detect_role_target("I mentioned a book earlier, which one?") == RoleTarget.USER

    def test_what_did_i_tell(self) -> None:
        assert detect_role_target("What did I tell you about my project?") == RoleTarget.USER

    def test_something_i_said(self) -> None:
        assert detect_role_target("Something I said about Python") == RoleTarget.USER

    def test_what_i_mentioned(self) -> None:
        assert (
            detect_role_target("Do you remember what I mentioned about Docker?") == RoleTarget.USER
        )

    def test_my_question(self) -> None:
        assert detect_role_target("My question was about databases") == RoleTarget.USER


# ---------------------------------------------------------------------------
# detect_role_target — no role signal
# ---------------------------------------------------------------------------


class TestDetectRoleTargetNone:
    """Test queries with no role signal return None."""

    def test_plain_question(self) -> None:
        assert detect_role_target("What is Python?") is None

    def test_recommendation_request(self) -> None:
        assert detect_role_target("Can you recommend a good IDE?") is None

    def test_how_to(self) -> None:
        assert detect_role_target("How do I install Docker?") is None

    def test_empty(self) -> None:
        assert detect_role_target("") is None

    def test_short(self) -> None:
        assert detect_role_target("hi") is None

    def test_generic_discussion(self) -> None:
        assert detect_role_target("Tell me about machine learning") is None


# ---------------------------------------------------------------------------
# RoleTarget enum
# ---------------------------------------------------------------------------


class TestRoleTargetEnum:
    """Test RoleTarget enum values."""

    def test_values(self) -> None:
        assert RoleTarget.ASSISTANT.value == "assistant"
        assert RoleTarget.USER.value == "user"

    def test_distinct(self) -> None:
        assert RoleTarget.ASSISTANT != RoleTarget.USER
