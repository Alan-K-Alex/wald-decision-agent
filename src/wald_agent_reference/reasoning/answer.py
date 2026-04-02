from __future__ import annotations

import json
from pathlib import Path
import re

from ..core.config import AppSettings
from ..core.models import AgentResponse, CalculationResult, QueryPlan, RetrievedChunk, VisualizationResult
from ..utils import compact_whitespace, tokenize


class AnswerComposer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def compose(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None = None,
        visualization: VisualizationResult | None = None,
    ) -> AgentResponse:
        response = self._fallback_response(question, plan, retrieved, calculation, visualization)
        if self.settings.enable_llm_formatting and self.settings.active_api_key:
            llm_response = self._try_llm_formatting(response)
            if llm_response is not None:
                return llm_response
        return response

    def _fallback_response(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None,
        visualization: VisualizationResult | None,
    ) -> AgentResponse:
        evidence = []
        source_refs: list[str] = []
        for item in retrieved[: plan.max_sources]:
            label = self._reference_label(item.chunk.source_path, item.chunk.metadata)
            source_refs.append(label)
            evidence.append(f"{label}: {self._best_evidence_snippet(question, item.chunk.content)}")

        if calculation:
            executive_summary = calculation.answer
            key_findings = calculation.findings
            calculations = calculation.trace
            for ref in calculation.evidence_refs:
                if ref not in source_refs:
                    source_refs.append(ref)
        elif self._has_sufficient_evidence(question, retrieved):
            summary, findings = self._grounded_summary(question, retrieved)
            executive_summary = summary
            key_findings = findings
            calculations = []
        else:
            executive_summary = "Insufficient evidence in the provided documents to answer this question reliably."
            key_findings = ["The agent did not find enough grounded support to answer without risking hallucination."]
            calculations = []

        visual_insights = [visualization.caption] if visualization else []
        caveats = []
        if not retrieved:
            caveats.append("No supporting evidence was retrieved.")
        if calculation is None and any(term in question.lower() for term in ["revenue", "growth", "underperform", "trend", "margin", "chart", "graph"]):
            caveats.append("No deterministic calculation was produced for this numeric-style query.")
        if visualization is None and any(term in question.lower() for term in ["chart", "plot", "graph", "visual"]):
            caveats.append("A visualization was requested but no compatible chart data was available.")

        return AgentResponse(
            question=question,
            planned_approach=plan.reasoning,
            executive_summary=executive_summary,
            key_findings=key_findings,
            calculations=calculations,
            evidence=evidence,
            caveats=caveats,
            source_references=source_refs,
            visual_insights=visual_insights,
            plot_paths=[visualization.path] if visualization else [],
        )

    def _try_llm_formatting(self, response: AgentResponse) -> AgentResponse | None:
        if self.settings.llm_provider == "gemini":
            return self._try_gemini_formatting(response)
        if self.settings.llm_provider == "openai":
            return self._try_openai_formatting(response)
        return self._try_gemini_formatting(response) or self._try_openai_formatting(response)

    def _try_gemini_formatting(self, response: AgentResponse) -> AgentResponse | None:
        try:
            from google import genai
        except ImportError:
            return None

        prompt = (
            "Rewrite the following grounded report as concise leadership-ready prose. "
            "Preserve factual values, citations, and caveats. Return strict JSON with keys: "
            "executive_summary, key_findings, calculations, evidence, caveats, source_references, visual_insights.\n\n"
            f"{response.to_markdown()}"
        )
        try:
            client = genai.Client(api_key=self.settings.gemini_api_key) if self.settings.gemini_api_key else genai.Client()
            completion = client.models.generate_content(model=self.settings.chat_model, contents=prompt)
            content = completion.text
            data = json.loads(content)
            return AgentResponse(
                question=response.question,
                planned_approach=response.planned_approach,
                executive_summary=data["executive_summary"],
                key_findings=list(data["key_findings"]),
                calculations=list(data["calculations"]),
                evidence=list(data["evidence"]),
                caveats=list(data["caveats"]),
                source_references=list(data["source_references"]),
                visual_insights=list(data.get("visual_insights", [])),
                plot_paths=response.plot_paths,
            )
        except Exception:
            return None

    def _grounded_summary(self, question: str, retrieved: list[RetrievedChunk]) -> tuple[str, list[str]]:
        sentences: list[str] = []
        seen: set[str] = set()
        query_tokens = self._significant_tokens(question)
        for item in retrieved:
            for sentence in self._split_sentences(item.chunk.content):
                normalized = sentence.strip()
                if not normalized or normalized in seen:
                    continue
                score = len(query_tokens & self._significant_tokens(normalized))
                if score == 0 and len(sentences) >= 2:
                    continue
                sentences.append(normalized)
                seen.add(normalized)
                if len(sentences) >= 3:
                    break
            if len(sentences) >= 3:
                break

        if not sentences:
            return (
                "Insufficient evidence in the provided documents to answer this question reliably.",
                ["No grounded supporting sentence was found in the retrieved evidence."],
            )
        executive_summary = compact_whitespace(sentences[0])
        key_findings = [compact_whitespace(sentence) for sentence in sentences[:3]]
        return executive_summary, key_findings

    def _has_sufficient_evidence(self, question: str, retrieved: list[RetrievedChunk]) -> bool:
        if not retrieved:
            return False
        query_tokens = self._significant_tokens(question)
        if not query_tokens:
            return False
        support_hits = 0
        for item in retrieved[:3]:
            overlap = query_tokens & self._significant_tokens(item.chunk.content)
            if overlap:
                support_hits += 1
        return support_hits >= 1

    def _best_evidence_snippet(self, question: str, content: str) -> str:
        query_tokens = self._significant_tokens(question)
        sentences = self._split_sentences(content)
        ranked = sorted(
            sentences,
            key=lambda sentence: len(query_tokens & self._significant_tokens(sentence)),
            reverse=True,
        )
        snippet = ranked[0] if ranked else content[:220]
        return compact_whitespace(snippet[:260])

    def _split_sentences(self, text: str) -> list[str]:
        return [compact_whitespace(part) for part in re.split(r"(?<=[.!?])\s+", text) if compact_whitespace(part)]

    def _reference_label(self, path: Path, metadata: dict[str, object]) -> str:
        label = path.name
        if metadata.get("page"):
            label += f" (p.{metadata['page']})"
        if metadata.get("sheet_name"):
            label += f" [{metadata['sheet_name']}]"
        return f"[{label}]({path.resolve()})"

    def _significant_tokens(self, text: str) -> set[str]:
        stopwords = {
            "what",
            "which",
            "does",
            "show",
            "the",
            "our",
            "all",
            "across",
            "current",
            "is",
            "are",
            "was",
            "were",
            "by",
            "of",
            "and",
            "to",
            "for",
            "in",
            "on",
            "trend",
            "chart",
            "graph",
            "visual",
            "question",
        }
        return {token for token in tokenize(text) if token not in stopwords and len(token) > 2}

    def _try_openai_formatting(self, response: AgentResponse) -> AgentResponse | None:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=self.settings.openai_api_key)
        prompt = (
            "Rewrite the following grounded report as concise leadership-ready prose. "
            "Preserve factual values, citations, and caveats. Return strict JSON with keys: "
            "executive_summary, key_findings, calculations, evidence, caveats, source_references, visual_insights.\n\n"
            f"{response.to_markdown()}"
        )
        try:
            completion = client.responses.create(model=self.settings.chat_model, input=prompt)
            content = completion.output_text
            data = json.loads(content)
            return AgentResponse(
                question=response.question,
                planned_approach=response.planned_approach,
                executive_summary=data["executive_summary"],
                key_findings=list(data["key_findings"]),
                calculations=list(data["calculations"]),
                evidence=list(data["evidence"]),
                caveats=list(data["caveats"]),
                source_references=list(data["source_references"]),
                visual_insights=list(data.get("visual_insights", [])),
                plot_paths=response.plot_paths,
            )
        except Exception:
            return None
