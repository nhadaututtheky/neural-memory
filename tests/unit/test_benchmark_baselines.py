from __future__ import annotations

from datetime import datetime

from scripts.benchmark.baselines import BASELINES, retrieve_naive
from scripts.benchmark.data_loader import LMEInstance, Session, Turn


def _session(session_id: str, text: str) -> Session:
    return Session(
        session_id=session_id,
        timestamp=datetime(2024, 1, 1),
        turns=[Turn(role="user", content=text)],
    )


def test_retrieve_naive_ranks_complete_sessions_by_query_token_overlap() -> None:
    instance = LMEInstance(
        question_id="q-1",
        question_type="single-session-user",
        question="Where did I leave the blue bicycle?",
        answer="In the garage",
        question_date="2024/01/02 (Tue) 00:00",
        sessions=[
            _session("session-c", "The bicycle needs a repair."),
            _session("session-a", "The blue bicycle is inside the garage."),
            _session("session-b", "A completely unrelated conversation."),
        ],
        answer_session_ids=["session-a"],
    )

    result = retrieve_naive(instance, top_k=3)

    assert result.session_ids == ["session-a", "session-c", "session-b"]
    assert result.extra["method"] == "naive_token_overlap"
    assert len(result.extra["scores"]) == 3
    assert "naive" in BASELINES


def test_retrieve_naive_empty_text_has_stable_id_order_and_honors_top_k() -> None:
    instance = LMEInstance(
        question_id="q-empty",
        question_type="single-session-user",
        question="",
        answer="",
        question_date="",
        sessions=[
            _session("session-z", ""),
            _session("session-a", ""),
            _session("session-m", ""),
        ],
    )

    result = retrieve_naive(instance, top_k=2)

    assert result.session_ids == ["session-a", "session-m"]
    assert result.extra["scores"] == [0, 0]
