from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.agent import LeadershipInsightAgent


class EvaluationRunner:
    def __init__(self, agent: LeadershipInsightAgent) -> None:
        self.agent = agent

    def run(self, docs_path: str | Path, validation_path: str | Path) -> list[dict[str, Any]]:
        with Path(validation_path).open("r", encoding="utf-8") as handle:
            cases = json.load(handle)

        results = []
        for case in cases:
            response = self.agent.ask(question=case["question"], docs_path=docs_path, generate_plot=case.get("generate_plot", False))
            markdown = response.to_markdown()
            source_refs = set(response.source_references)
            expected_sources = set(case.get("expected_sources", []))
            expected_keywords = [keyword.lower() for keyword in case.get("expected_keywords", [])]

            keyword_hits = sum(1 for keyword in expected_keywords if keyword in markdown.lower())
            source_hits = len(expected_sources & source_refs)
            numeric_score = self._numeric_match(markdown, case.get("expected_numeric"))
            results.append(
                {
                    "question": case["question"],
                    "keyword_score": keyword_hits / max(1, len(expected_keywords)),
                    "source_score": source_hits / max(1, len(expected_sources)) if expected_sources else 1.0,
                    "numeric_score": numeric_score,
                    "plot_generated": bool(response.plot_paths),
                }
            )
        return results

    @staticmethod
    def _numeric_match(markdown: str, expected_numeric: float | None) -> float:
        if expected_numeric is None:
            return 1.0
        numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", markdown)]
        if not numbers:
            return 0.0
        nearest = min(numbers, key=lambda value: abs(value - expected_numeric))
        return 1.0 if abs(nearest - expected_numeric) < 0.05 * max(1.0, abs(expected_numeric)) else 0.0
