from __future__ import annotations

import re

from ..core.models import CalculationResult, VisualArtifact
from ..utils import tokenize


class VisualReasoner:
    _QUESTION_STOPWORDS = {
        "a",
        "all",
        "across",
        "an",
        "and",
        "any",
        "are",
        "chart",
        "figure",
        "for",
        "graph",
        "has",
        "how",
        "in",
        "is",
        "our",
        "over",
        "quarter",
        "quarterly",
        "show",
        "shows",
        "subsidiaries",
        "the",
        "trend",
        "visual",
        "what",
    }

    def answer(self, question: str, visuals: list[VisualArtifact]) -> CalculationResult | None:
        lowered = question.lower()
        if not any(term in lowered for term in ["chart", "graph", "visual", "figure", "trend", "quarterly"]):
            return None
        if not visuals:
            return None

        visual = visuals[0]
        if not self._is_grounded_visual_match(question, visual):
            return None
        extracted = visual.extracted_text
        quarters = re.findall(r"\bQ[1-4]\b", extracted, flags=re.IGNORECASE)
        numbers = [float(match) for match in re.findall(r"\b\d+(?:\.\d+)?\b", extracted)]
        if quarters and len(numbers) >= len(quarters):
            values = numbers[-len(quarters) :]
            direction = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "flat"
            answer = (
                f"The chart shows an overall {direction} trend across "
                f"{', '.join(quarters)} with labeled values {', '.join(f'{value:,.0f}' for value in values)}."
            )
            return CalculationResult(
                answer=answer,
                findings=[
                    f"Visual title: {visual.metadata.get('title', visual.source_path.stem)}.",
                    f"Detected sequence: {', '.join(f'{quarter}={value:,.0f}' for quarter, value in zip(quarters, values))}.",
                ],
                trace=[
                    f"Visual extraction backend: {visual.metadata.get('extraction_backend', 'unknown')}",
                    f"Source visual: {visual.source_path.name}",
                ],
                evidence_refs=[f"[{visual.source_path.name}]({visual.source_path})"],
                chart_data={"type": "line", "labels": quarters, "values": values, "title": visual.metadata.get("title", visual.source_path.stem)},
                numeric_value=values[-1],
            )

        return CalculationResult(
            answer=visual.summary,
            findings=[visual.extracted_text] if visual.extracted_text else [visual.summary],
            trace=[
                f"Visual extraction backend: {visual.metadata.get('extraction_backend', 'unknown')}",
                f"Source visual: {visual.source_path.name}",
            ],
            evidence_refs=[f"[{visual.source_path.name}]({visual.source_path})"],
        )

    def _is_grounded_visual_match(self, question: str, visual: VisualArtifact) -> bool:
        visual_text = " ".join(
            [
                visual.metadata.get("title", "") if isinstance(visual.metadata.get("title", ""), str) else "",
                visual.extracted_text,
                visual.summary,
                visual.source_path.stem,
            ]
        ).lower()
        visual_tokens = set(tokenize(visual_text))

        significant_question_tokens = {
            token
            for token in tokenize(question)
            if token not in self._QUESTION_STOPWORDS and not re.fullmatch(r"q[1-4]|\d+", token)
        }
        if not significant_question_tokens:
            return True
        return bool(significant_question_tokens & visual_tokens)
