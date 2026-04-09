from __future__ import annotations

import re

from ..core.models import CalculationResult, VisualArtifact


class VisualReasoner:
    def answer(self, question: str, visuals: list[VisualArtifact]) -> CalculationResult | None:
        lowered = question.lower()
        if not any(term in lowered for term in ["chart", "graph", "visual", "figure", "trend", "quarterly"]):
            return None
        if not visuals:
            return None

        visual = visuals[0]
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
                evidence_refs=[visual.source_path.name],
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
            evidence_refs=[visual.source_path.name],
        )
