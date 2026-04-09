"""
Improved response models separating chat response from detailed reports.

ChatResponse: What users see immediately in chat (concise, visual-focused)
DetailedReport: Full analysis with all steps, evidence, and methodology
"""

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
class TablePreview:
    """Schema preview for intelligent query planning"""
    table_name: str
    source_file: str
    row_count: int
    columns: dict[str, str]  # column_name -> data_type
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def column_summary(self) -> str:
        """Get a human-readable column summary"""
        cols = ", ".join([f"{name}({dtype})" for name, dtype in self.columns.items()])
        return f"{self.table_name}: {cols}"


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
class SourceReference:
    """Individual source reference with metadata"""
    file_name: str
    file_path: str
    source_type: str  # "table", "document", "visual"
    location: str | None = None  # e.g., "Sheet1, Row 5", "Page 3", "Chart 1"
    relevance_score: float = 1.0


@dataclass
class ChatResponse:
    """Response optimized for chat display - concise and visual-focused"""
    question: str
    answer: str  # CONCISE answer to the question
    key_findings: list[str]  # Top 3-5 most important findings
    visual_insights: list[str]  # Chart/visual descriptions
    plots_base64: list[str] = field(default_factory=list)  # Embedded visualizations
    source_summary: str = ""  # e.g., "Referenced 3 tables and 2 documents"
    data_types_used: list[str] = field(default_factory=list)  # ["tables", "text", "visuals"]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for API response"""
        return {
            "question": self.question,
            "answer": self.answer,
            "key_findings": self.key_findings,
            "visual_insights": self.visual_insights,
            "plots_base64": self.plots_base64,
            "source_summary": self.source_summary,
            "data_types_used": self.data_types_used,
        }


@dataclass
class DetailedReport:
    """Full analysis report with methodology, all evidence, and detailed findings"""
    question: str
    executive_summary: str
    
    # Methodology
    planned_approach: list[str]  # Steps taken to answer
    query_routing_logic: str  # Explanation of why routes were chosen
    
    # Detailed findings
    detailed_findings: list[str]  # Long-form analysis
    
    # Evidence with context
    evidence_bundles: list[EvidenceBundle] = field(default_factory=list)
    
    # Source information
    source_references: list[SourceReference] = field(default_factory=list)
    
    # Trace information
    calculations: list[str] = field(default_factory=list)
    trace_steps: list[str] = field(default_factory=list)
    
    # Limitations
    caveats: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Export as detailed markdown report"""
        sections = [
            ("Question", self.question),
            ("Executive Summary", self.executive_summary),
            ("Methodology", "\n".join(f"- {step}" for step in self.planned_approach)),
            ("Detailed Findings", "\n".join(f"{i}. {f}" for i, f in enumerate(self.detailed_findings, 1))),
            ("Evidence", self._format_evidence()),
            ("Sources", self._format_sources()),
            ("Caveats", "\n".join(f"- {c}" for c in self.caveats) if self.caveats else "None"),
        ]
        return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
    
    def _format_evidence(self) -> str:
        """Format evidence bundles"""
        parts = []
        for bundle in self.evidence_bundles:
            parts.append(f"**{bundle.source_file}** (relevance: {bundle.relevance_score:.2%})")
            parts.append(f"  {bundle.content[:200]}...")
            parts.append("")
        return "\n".join(parts) if parts else "No evidence"
    
    def _format_sources(self) -> str:
        """Format source references"""
        parts = []
        for ref in self.source_references:
            location = f" [{ref.location}]" if ref.location else ""
            parts.append(f"- {ref.file_name}{location} ({ref.source_type})")
        return "\n".join(parts) if parts else "No sources"


@dataclass
class EvidenceBundle:
    """Collection of related evidence from a source"""
    source_file: str
    source_type: str  # "table", "document", "visual"
    content: str  # Snippet or summary
    location: str | None = None  # Page, row, sheet, etc.
    relevance_score: float = 1.0
    full_chunk_id: str = ""  # Reference to original chunk


@dataclass
class AgentResponse:
    """DEPRECATED: Use ChatResponse + DetailedReport instead
    
    Kept for backward compatibility during migration.
    This combines both chat and detailed response.
    """
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
    
    def to_chat_response(self) -> ChatResponse:
        """Convert to ChatResponse for new response format"""
        # Infer data types from source references
        data_types = set()
        for ref in self.source_references:
            if ".csv" in ref or ".xlsx" in ref:
                data_types.add("tables")
            elif ".md" in ref or ".txt" in ref:
                data_types.add("text")
            if self.plot_base64 or self.plot_paths:
                data_types.add("visuals")
        
        return ChatResponse(
            question=self.question,
            answer=self.executive_summary,
            key_findings=self.key_findings[:5],  # Limit to top 5
            visual_insights=self.visual_insights,
            plots_base64=[self.plot_base64] if self.plot_base64 else [],
            source_summary=f"Referenced {len(self.source_references)} sources",
            data_types_used=list(data_types),
        )
    
    def to_detailed_report(self) -> DetailedReport:
        """Convert to DetailedReport with full details"""
        return DetailedReport(
            question=self.question,
            executive_summary=self.executive_summary,
            planned_approach=self.planned_approach,
            query_routing_logic="See planned approach for routing decisions",
            detailed_findings=self.key_findings,  # In real impl, would be longer
            evidence_bundles=[],  # Would populate from evidence
            source_references=[
                SourceReference(
                    file_name=ref,
                    file_path=ref,
                    source_type="mixed",
                )
                for ref in self.source_references
            ],
            calculations=self.calculations,
            caveats=self.caveats,
        )
