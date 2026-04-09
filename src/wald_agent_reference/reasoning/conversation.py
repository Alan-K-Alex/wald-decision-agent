from __future__ import annotations

from dataclasses import dataclass

from ..utils import compact_whitespace, tokenize


@dataclass
class ResolvedQuestion:
    question: str
    used_history: bool
    reason: str = ""


class ConversationContextResolver:
    """Lightweight follow-up resolver for chat sessions.

    Referential follow-ups are expanded with the most recent user question and
    assistant answer summary. Standalone questions are returned unchanged.
    """

    _referential_terms = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "them",
        "they",
        "their",
        "former",
        "latter",
    }

    def resolve(self, question: str, messages: list[dict[str, object]]) -> ResolvedQuestion:
        cleaned = compact_whitespace(question)
        previous_question, previous_answer = self._last_turn(messages)
        if not previous_question:
            return ResolvedQuestion(question=cleaned, used_history=False)
        if not self._looks_like_follow_up(cleaned):
            return ResolvedQuestion(question=cleaned, used_history=False)

        contextualized = self._build_contextualized_question(cleaned, previous_question, previous_answer)
        return ResolvedQuestion(
            question=contextualized,
            used_history=True,
            reason="Expanded follow-up question using the latest chat turn.",
        )

    def _last_turn(self, messages: list[dict[str, object]]) -> tuple[str | None, str | None]:
        previous_answer: str | None = None
        previous_question: str | None = None
        for message in reversed(messages):
            role = message.get("role")
            if role == "assistant" and previous_answer is None:
                previous_answer = compact_whitespace(str(message.get("answer") or message.get("content") or ""))
                continue
            if role == "user":
                previous_question = compact_whitespace(str(message.get("content", "")))
                if previous_question:
                    return previous_question, previous_answer
        return None, previous_answer

    def _looks_like_follow_up(self, question: str) -> bool:
        lowered = question.lower()
        tokens = tokenize(lowered)
        significant = [token for token in tokens if token not in {"how", "what", "why", "is", "was", "were", "are", "the", "a", "an"}]
        has_reference = any(token in self._referential_terms for token in tokens)
        if has_reference:
            return True
        follow_up_starters = ("why", "how", "what about", "how about", "and ", "also ", "is it", "was it", "did you")
        return len(significant) <= 8 and lowered.startswith(follow_up_starters)

    def _build_contextualized_question(self, question: str, previous_question: str, previous_answer: str | None) -> str:
        parts = [f"Previous user question: {previous_question}"]
        if previous_answer:
            parts.append(f"Previous assistant answer: {previous_answer}")
        parts.append(f"Follow-up question: {question}")
        return "\n".join(parts)
