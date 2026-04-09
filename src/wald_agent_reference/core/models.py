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
    base64_image: str = ""  # Base64-encoded PNG image data


@dataclass
class ChatResponse:
    """Lightweight response optimized for chat UI display"""
    question: str
    answer: str
    key_findings: list[str]
    evidence: list[str] = field(default_factory=list)
    visual_insights: list[str] = field(default_factory=list)
    plots_base64: list[str] = field(default_factory=list)
    plot_paths: list[str] = field(default_factory=list)
    source_summary: str = ""
    data_types_used: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "question": self.question,
            "answer": self.answer,
            "key_findings": self.key_findings,
            "evidence": self.evidence,
            "visual_insights": self.visual_insights,
            "plots_base64": self.plots_base64,
            "plot_paths": [str(p) for p in self.plot_paths],
            "source_summary": self.source_summary,
            "data_types_used": self.data_types_used,
        }


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
    plot_base64: str = ""  # Base64-encoded plot image for embedding

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

    def to_display(self) -> str:
        """Format a concise chat response with only user-facing sections."""
        return self.to_chat_markdown()

    def to_chat_markdown(self, plot_urls: list[str] | None = None) -> str:
        """Export a concise markdown view for chat surfaces."""
        sections = [
            ("Executive Summary", self.executive_summary),
            ("Key Findings", "\n".join(f"{idx}. {item}" for idx, item in enumerate(self.key_findings, start=1)) or "1. No findings generated."),
            ("Evidence", "\n".join(f"- {item}" for item in self.evidence) or "- No evidence retrieved."),
        ]

        rendered_plot_urls = plot_urls or [str(path.resolve()) for path in self.plot_paths]
        if rendered_plot_urls:
            plot_body = "\n".join(f"![plot]({url})" for url in rendered_plot_urls)
            sections.append(("Plots", plot_body))
        return "\n\n".join(f"{title}\n{body}" for title, body in sections)

    def to_chat_response(self) -> ChatResponse:
        """Convert to concise chat response showing only Executive Summary, Key Findings, Evidence, and Plots
        
        Returns a lightweight response optimized for UI display.
        Full details preserved in to_markdown() for reports.
        """
        # Infer data types from source references
        data_types = set()
        for ref in self.source_references:
            if ".csv" in ref or ".xlsx" in ref:
                data_types.add("tables")
            elif ".md" in ref or ".txt" in ref:
                data_types.add("text")
        if self.plot_base64 or self.plot_paths:
            data_types.add("visuals")
        
        # Prepare plot paths as strings for JSON serialization
        plot_paths_str = [str(p) for p in self.plot_paths] if self.plot_paths else []
        
        # Build concise response with only essential sections
        return ChatResponse(
            question=self.question,
            answer=self.executive_summary,
            key_findings=self.key_findings,
            evidence=self.evidence,
            visual_insights=self.visual_insights,
            plots_base64=[self.plot_base64] if self.plot_base64 else [],
            plot_paths=plot_paths_str,
            source_summary=f"Referenced {len(self.source_references)} sources",
            data_types_used=list(data_types),
        )
