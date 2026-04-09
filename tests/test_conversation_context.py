from __future__ import annotations

from wald_agent_reference.reasoning.conversation import ConversationContextResolver


def test_follow_up_question_is_rewritten_using_previous_topic() -> None:
    resolver = ConversationContextResolver()
    messages = [
        {"role": "user", "content": "What are the primary risks in each region?"},
        {"role": "assistant", "answer": "Europe has the highest risk score."},
    ]

    resolved = resolver.resolve("How were the risk scores for these estimated?", messages)

    assert resolved.used_history is True
    assert "Previous user question:" in resolved.question
    assert "What are the primary risks in each region?" in resolved.question
    assert "Previous assistant answer:" in resolved.question
    assert "Follow-up question: How were the risk scores for these estimated?" in resolved.question


def test_new_topic_question_is_not_rewritten() -> None:
    resolver = ConversationContextResolver()
    messages = [
        {"role": "user", "content": "What are the primary risks in each region?"},
        {"role": "assistant", "answer": "Europe has the highest risk score."},
    ]

    resolved = resolver.resolve("What are the strategic priorities for next year?", messages)

    assert resolved.used_history is False
    assert resolved.question == "What are the strategic priorities for next year?"
