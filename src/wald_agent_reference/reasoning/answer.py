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
        self.docs_base_path = None  # Will be set during composition

    def compose(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None = None,
        visualization: VisualizationResult | None = None,
        docs_path: str | Path | None = None,
    ) -> AgentResponse:
        self.docs_base_path = Path(docs_path) if docs_path else None
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
        # Filter retrieved chunks by relevance to prevent low-quality content
        high_quality_retrieved = self._filter_by_relevance(question, retrieved)
        
        # Build evidence from high-quality retrieved chunks only
        evidence = []
        source_refs: list[str] = []
        for item in high_quality_retrieved[: plan.max_sources]:
            label = self._reference_label(item.chunk.source_path, item.chunk.metadata)
            source_refs.append(label)
            evidence.append(f"{label}: {self._best_evidence_snippet(question, item.chunk.content)}")

        # Intelligently select source: calculation vs retrieval vs both
        selected_source = self._select_best_source(question, calculation, high_quality_retrieved)
        
        if selected_source == "calculation" and calculation:
            # Use calculation (SQL/tables) as primary answer
            executive_summary = calculation.answer
            key_findings = calculation.findings
            calculations = calculation.trace
            for ref in calculation.evidence_refs:
                if ref not in source_refs:
                    source_refs.append(ref)
                    
        elif selected_source == "retrieval" and self._has_sufficient_evidence(question, high_quality_retrieved):
            # Use retrieval (documents) as primary answer
            summary, findings = self._grounded_summary(question, high_quality_retrieved)
            executive_summary = summary
            key_findings = findings
            calculations = []
            
        elif selected_source == "hybrid" and calculation and high_quality_retrieved:
            # Intelligently combine both sources
            executive_summary = calculation.answer
            key_findings = self._combine_findings(calculation.findings, high_quality_retrieved, question)
            calculations = calculation.trace
            for ref in calculation.evidence_refs:
                if ref not in source_refs:
                    source_refs.append(ref)
                    
        else:
            # Fallback: insufficient evidence
            executive_summary = "Insufficient evidence in the provided documents to answer this question reliably."
            key_findings = ["The agent did not find enough grounded support to answer without risking hallucination."]
            calculations = []

        visual_insights = [visualization.caption] if visualization else []
        caveats = self._generate_caveats(question, calculation, high_quality_retrieved, visualization)

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
            plot_base64=visualization.base64_image if visualization else "",
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

        # ANTI-HALLUCINATION PROMPT: Explicitly prevent fabrication
        prompt = (
            "Rewrite the following grounded research report as concise leadership-ready prose. "
            "CRITICAL RULES:"
            "1. PRESERVE ALL factual values, metrics, percentages, and dates EXACTLY as they appear"
            "2. DO NOT add new facts, statistics, or details not explicitly in the original report"
            "3. DO NOT remove citations or caveats"
            "4. DO NOT speculate or infer beyond what the report states"
            "5. If the report says 'Insufficient evidence', keep that language"
            "6. Maintain the source references as-is"
            "7. Return ONLY valid JSON with keys: executive_summary, key_findings, calculations, evidence, caveats, source_references, visual_insights\n\n"
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
        
        # First pass: collect sentences with strong token overlap
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
                if len(sentences) >= 5:
                    break
            if len(sentences) >= 5:
                break

        # Second pass: if we don't have enough with strong overlap, be more lenient
        if len(sentences) < 2:
            for item in retrieved:
                item_text = compact_whitespace(item.chunk.content[:200])  # Use first 200 chars
                if item_text not in seen:
                    sentences.append(item_text)
                    seen.add(item_text)
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
        """Check if retrieved chunks have sufficient quality to ground an answer.
        
        STRICTER version to prevent hallucination:
        - Requires at least 2 chunks with >25% token overlap with question
        - Or at least 3 chunks with >15% token overlap
        - This prevents weak answers from insufficient evidence
        """
        if not retrieved:
            return False
        
        query_tokens = self._significant_tokens(question)
        if not query_tokens:
            return False
        
        # Count strong matches (>25% overlap)
        strong_matches = 0
        # Count ok matches (>15% overlap)
        ok_matches = 0
        
        for item in retrieved[:5]:  # Check top 5 chunks
            chunk_tokens = self._significant_tokens(item.chunk.content)
            max_len = max(len(query_tokens), len(chunk_tokens))
            if max_len == 0:
                continue
                
            overlap = len(query_tokens & chunk_tokens)
            overlap_pct = overlap / max_len
            
            if overlap_pct >= 0.25:  # Strong overlap
                strong_matches += 1
            elif overlap_pct >= 0.15:  # Weak overlap
                ok_matches += 1
        
        # STRICTER: Need either 2+ strong matches OR 3+ ok matches
        sufficient = strong_matches >= 2 or ok_matches >= 3
        
        return sufficient

    def _filter_by_relevance(self, question: str, retrieved: list[RetrievedChunk], threshold: float = 0.3) -> list[RetrievedChunk]:
        """Filter retrieved chunks to keep only those with sufficient relevance to the question.
        
        Relevance is measured by token overlap between question and chunk content.
        This prevents low-relevance chunks from polluting the answer.
        """
        if not retrieved:
            return []
        
        query_tokens = self._significant_tokens(question)
        if not query_tokens:
            return retrieved  # If question has no significant tokens, keep all
        
        filtered = []
        for item in retrieved:
            chunk_tokens = self._significant_tokens(item.chunk.content)
            overlap = query_tokens & chunk_tokens
            relevance_score = len(overlap) / max(len(query_tokens), len(chunk_tokens)) if chunk_tokens else 0
            
            # Keep chunks with sufficient token overlap
            if relevance_score >= threshold or len(overlap) >= 2:  # At least 2 matching tokens
                filtered.append(item)
        
        return filtered if filtered else retrieved[:1]  # Keep at least one chunk as fallback

    def _select_best_source(self, question: str, calculation: CalculationResult | None, retrieved: list[RetrievedChunk]) -> str:
        """Intelligently select which source(s) to use for the answer: 'calculation', 'retrieval', or 'hybrid'."""
        
        question_lower = question.lower()
        
        # Detect question type
        is_numeric_question = any(term in question_lower for term in [
            "revenue", "growth", "number", "how many", "how much", "total",
            "metric", "performance", "target", "actual", "variance", "margin",
            "chart", "graph", "trend", "quarter", "ytd", "exact"
        ])
        
        is_narrative_question = any(term in question_lower for term in [
            "why", "what", "challenge", "risk", "concern", "improvement", "priority",
            "strategic", "leadership", "context", "reason", "cause", "factor", "driver",
            "sentiment", "concern", "bottleneck", "pressure"
        ])
        
        # Check source quality
        has_good_calculation = calculation is not None and calculation.answer and len(calculation.answer) > 20
        has_good_retrieval = len(retrieved) > 0 and self._has_sufficient_evidence(question, retrieved)
        
        # Selection logic
        if is_numeric_question and has_good_calculation and not has_good_retrieval:
            # Purely numeric: use calculation if available
            return "calculation"
        elif is_narrative_question and has_good_retrieval and not has_good_calculation:
            # Pure narrative: use retrieval if available
            return "retrieval"
        elif is_numeric_question and has_good_calculation and has_good_retrieval:
            # Numeric question with both sources: prefer calculation, supplement with context
            return "calculation"
        elif is_narrative_question and has_good_retrieval and has_good_calculation:
            # Narrative with both sources: prefer retrieval, supplement with metrics
            return "retrieval"
        elif has_good_calculation:
            # Default to calculation if available
            return "calculation"
        elif has_good_retrieval:
            # Fall back to retrieval
            return "retrieval"
        else:
            # No good source
            return "insufficient"

    def _combine_findings(self, calc_findings: list[str], retrieved: list[RetrievedChunk], question: str) -> list[str]:
        """Intelligently combine findings from calculation and retrieval.
        
        Prioritizes calculation findings but enriches with top retrieval finding if relevant.
        Deduplicates and avoids redundancy.
        """
        combined = list(calc_findings)  # Start with calculation findings
        
        # Add top retrieval finding if it doesn't duplicate
        if retrieved and len(combined) < 5:
            retrieval_findings = self._extract_findings_from_retrieval(retrieved, question)
            for finding in retrieval_findings:
                # Check if not already in combined (fuzzy duplicate check)
                if not any(self._is_similar(finding, cf) for cf in combined):
                    combined.append(finding)
                    if len(combined) >= 5:
                        break
        
        return combined[:5]  # Limit to 5 findings for conciseness

    def _extract_findings_from_retrieval(self, retrieved: list[RetrievedChunk], question: str) -> list[str]:
        """Extract 1-2 key findings from retrieved chunks."""
        findings = []
        query_tokens = self._significant_tokens(question)
        
        for item in retrieved[:2]:
            sentences = self._split_sentences(item.chunk.content)
            # Find sentences most relevant to question
            ranked = sorted(
                sentences,
                key=lambda s: len(query_tokens & self._significant_tokens(s)),
                reverse=True
            )
            if ranked:
                findings.append(ranked[0])
        
        return findings

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts are similar (for deduplication).
        
        Uses word overlap to detect similar statements.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return False
        
        overlap = len(tokens1 & tokens2)
        max_set = max(len(tokens1), len(tokens2))
        similarity = overlap / max_set if max_set > 0 else 0
        
        return similarity >= threshold

    def _generate_caveats(self, question: str, calculation: CalculationResult | None, 
                         retrieved: list[RetrievedChunk], visualization: VisualizationResult | None) -> list[str]:
        """Generate contextual caveats based on what data was/wasn't available."""
        caveats = []
        
        if not retrieved:
            caveats.append("No supporting evidence was retrieved from documents.")
        
        if calculation is None and any(term in question.lower() for term in 
            ["revenue", "growth", "underperform", "trend", "margin", "chart", "graph"]):
            caveats.append("No deterministic calculation was produced for this numeric-style query.")
        
        if visualization is None and any(term in question.lower() for term in 
            ["chart", "plot", "graph", "visual", "visualize"]):
            caveats.append("A visualization was requested but no compatible chart data was available.")
        
        return caveats

    def _compute_confidence_score(self, question: str, retrieved: list[RetrievedChunk], 
                                 calculation: CalculationResult | None) -> float:
        """Compute overall confidence score for the answer (0.0 to 1.0).
        
        Factors considered:
        - Number of supporting chunks
        - Relevance overlap with question
        - Calculation availability for numeric questions
        - Consistency across evidence
        """
        confidence = 0.0
        max_confidence = 0.0
        
        # Retrieval quality score
        if retrieved:
            max_confidence += 0.5
            query_tokens = self._significant_tokens(question)
            overlap_count = 0
            
            for item in retrieved[:3]:  # Check top 3
                chunk_overlap = len(query_tokens & self._significant_tokens(item.chunk.content))
                if chunk_overlap >= 2:
                    overlap_count += 1
            
            # Base 0.2 if we have any chunks, up to 0.5 for all 3 having good overlap
            retrieval_confidence = 0.2 + (0.3 * (overlap_count / 3.0))
            confidence += min(retrieval_confidence, 0.5)
        
        # Calculation quality score
        if calculation is not None and calculation.answer:
            max_confidence += 0.5
            calculation_confidence = 0.4  # Base credit for having calculation
            
            # Bonus if calculation has evidence references
            if calculation.evidence_refs:
                calculation_confidence += 0.1
            
            confidence += calculation_confidence
        
        # Normalize by what was attempted
        if max_confidence == 0:
            return 0.0  # No evidence at all
        
        return min(confidence / max_confidence, 1.0)

    def _is_answer_grounded(self, question: str, retrieved: list[RetrievedChunk], 
                           calculation: CalculationResult | None) -> tuple[bool, str]:
        """Check if answer has sufficient grounding. Returns (is_grounded, reason).
        
        Grounding requires:
        - For numeric questions: calculation with valid answer OR multiple relevant chunks
        - For narrative questions: at least 2 chunks with good token overlap
        - Minimum 1 source reference
        """
        question_lower = question.lower()
        is_numeric = any(term in question_lower for term in 
            ["revenue", "number", "how many", "how much", "metric", "percentage", "chart", "trend"])
        
        # Check calculation
        has_valid_calculation = (calculation is not None and 
                               calculation.answer and 
                               len(calculation.answer) > 10)
        
        # Check retrieval quality
        query_tokens = self._significant_tokens(question)
        good_matches = 0
        
        for item in retrieved[:3]:
            chunk_tokens = self._significant_tokens(item.chunk.content)
            overlap = len(query_tokens & chunk_tokens)
            if overlap >= 2:  # Strong overlap
                good_matches += 1
        
        # Determine if grounded
        if is_numeric:
            # Numeric questions: need either calculation or multiple good chunks
            if has_valid_calculation:
                return True, "Has valid calculation"
            elif good_matches >= 2:
                return True, "Has multiple chunks with good token overlap"
            else:
                return False, "Numeric question needs calculation or strong evidence"
        else:
            # Narrative questions: need at least 1-2 good matches
            if good_matches >= 1:
                return True, "Has chunks with good token overlap"
            elif len(retrieved) > 0:
                return False, "Retrieved chunks but with weak relevance"
            else:
                return False, "No relevant chunks retrieved"

    def _evidence_quality_score(self, evidence_list: list[str]) -> float:
        """Score quality of evidence (0.0 to 1.0) based on specificity and length.
        
        Good evidence:
        - 50+ characters (specific enough)
        - First-person/cited language
        - Concrete facts vs generic statements
        """
        if not evidence_list:
            return 0.0
        
        total_score = 0.0
        
        for evidence in evidence_list:
            score = 0.0
            
            # Length check (specific content > 0 chars)
            if len(evidence) > 50:
                score += 0.4
            elif len(evidence) > 20:
                score += 0.2
            
            # Concreteness check (has numbers, quotes, specifics)
            if any(char.isdigit() for char in evidence):
                score += 0.3
            
            # Has citations/references
            if "[" in evidence and "]" in evidence:
                score += 0.3
            
            total_score += score
        
        return min(total_score / len(evidence_list), 1.0)

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
        """Generate a formatted reference label with relative path and metadata."""
        # Build a display name with path and metadata
        parts = []
        
        # Get relative path from docs base if available
        if self.docs_base_path and path.is_relative_to(self.docs_base_path):
            rel_path = path.relative_to(self.docs_base_path)
            parts.append(str(rel_path))
        else:
            # Fallback to just filename
            parts.append(path.name)
        
        # Add sheet/section information
        if metadata.get("sheet_name"):
            parts.append(f"[{metadata['sheet_name']}]")
        
        # Add page information if available
        if metadata.get("page"):
            parts.append(f"p.{metadata['page']}")
        
        display_text = " ".join(parts)
        
        # Create a normalized relative path for the link (replace spaces with underscores in path)
        if self.docs_base_path and path.is_relative_to(self.docs_base_path):
            # Use the relative path as-is for the link
            link_path = str(path.relative_to(self.docs_base_path))
        else:
            link_path = path.name
        
        # Return markdown link format
        return f"[{display_text}]({link_path})"

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
        # ANTI-HALLUCINATION PROMPT: Explicitly prevent fabrication
        prompt = (
            "Rewrite the following grounded research report as concise leadership-ready prose. "
            "CRITICAL RULES:"
            "1. PRESERVE ALL factual values, metrics, percentages, and dates EXACTLY as they appear"
            "2. DO NOT add new facts, statistics, or details not explicitly in the original report"
            "3. DO NOT remove citations or caveats"
            "4. DO NOT speculate or infer beyond what the report states"
            "5. If the report says 'Insufficient evidence', keep that language"
            "6. Maintain the source references as-is"
            "7. Return ONLY valid JSON with keys: executive_summary, key_findings, calculations, evidence, caveats, source_references, visual_insights\n\n"
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
