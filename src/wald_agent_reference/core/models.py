from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DocumentChunk:
    chunk_id: str
    source_path: Path
    content: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredTable:
    table_id: str
    source_path: Path
    source_type: str
    dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_text: str = ""


@dataclass
class ExtractedDocument:
    document_id: str
    source_path: Path
    source_type: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualArtifact:
    artifact_id: str
    source_path: Path
    source_type: str
    extracted_text: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    chunks: list[DocumentChunk] = field(default_factory=list)
    tables: dict[str, StructuredTable] = field(default_factory=dict)
    documents: dict[str, ExtractedDocument] = field(default_factory=dict)
    visuals: dict[str, VisualArtifact] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass
class QueryPlan:
    primary_route: str
    route_sequence: list[str]
    reasoning: list[str]
    should_visualize: bool = False
    require_strict_grounding: bool = True
    max_sources: int = 5


@dataclass
class CalculationResult:
    answer: str
    findings: list[str]
    trace: list[str]
    evidence_refs: list[str]
    chart_data: dict[str, Any] | None = None
    numeric_value: float | None = None


@dataclass
class VisualizationResult:
    path: Path
    caption: str
    chart_type: str


@dataclass
class AgentResponse:
    question: str
    planned_approach: list[str]
    executive_summary: str
    key_findings: list[str]
    calculations: list[str]
    evidence: list[str]
    caveats: list[str]
    source_references: list[str]
    visual_insights: list[str] = field(default_factory=list)
    plot_paths: list[Path] = field(default_factory=list)

    def to_markdown(self) -> str:
        sections = [
            ("Question", self.question),
            ("Planned Approach", "\n".join(f"- {item}" for item in self.planned_approach) or "- No plan recorded."),
            ("Executive Summary", self.executive_summary),
            ("Key Findings", "\n".join(f"{idx}. {item}" for idx, item in enumerate(self.key_findings, start=1)) or "1. No findings generated."),
            ("Calculations Performed", "\n".join(f"- {item}" for item in self.calculations) or "- No explicit calculations were required."),
            ("Evidence", "\n".join(f"- {item}" for item in self.evidence) or "- No evidence retrieved."),
            ("Visual Insights", "\n".join(f"- {item}" for item in self.visual_insights) or "- No visualization generated."),
            ("Risks / Caveats", "\n".join(f"- {item}" for item in self.caveats) or "- No caveats."),
            ("Source References", "\n".join(f"- {item}" for item in self.source_references) or "- No references."),
        ]
        if self.plot_paths:
            plot_body = "\n".join(f"![plot]({path.resolve()})" for path in self.plot_paths)
            sections.append(("Plots", plot_body))
        return "\n\n".join(f"{title}\n{body}" for title, body in sections)
