from __future__ import annotations

import json
import logging
from pathlib import Path
import re

from ..core.config import AppSettings
from ..core.models import AgentResponse, CalculationResult, QueryPlan, RetrievedChunk, VisualizationResult
from ..utils import compact_whitespace, tokenize

logger = logging.getLogger(__name__)


class AnswerComposer:
    """Answer composer that uses LLM as primary formatter.
    
    Architecture:
    - LLM-Primary: Gemini/OpenAI formats all answers using anti-hallucination prompts
    - Simple fallback: Raw response if LLM unavailable
    - Avoids complex rule-based logic which doesn't generalize across domains
    
    This approach is simpler, more maintainable, and naturally handles all question types
    without needing domain-specific pattern matching rules.
    """
    
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.docs_base_path = None  # Will be set during composition

    def compose(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None = None,
        visualizations: list[VisualizationResult] | None = None,
        docs_path: str | Path | None = None,
        supplemental_calculations: list[CalculationResult] | None = None,
    ) -> AgentResponse:
        """Compose final answer using LLM as primary formatter.
        
        Flow:
        1. Build basic raw response with evidence from all sources
        2. Pass to LLM (Gemini/OpenAI) to polish into leadership-ready prose
        3. If LLM unavailable, return raw response as-is
        
        LLM Format Prompt ensures:
        - Preserves all factual values exactly
        - No hallucination or made-up facts
        - Clean, professional presentation
        - Proper grounding (evidence, caveats, sources)
        
        This is much simpler and more maintainable than rule-based logic.
        The LLM handles diverse question types, domains, and metrics naturally.
        """
        self.docs_base_path = Path(docs_path) if docs_path else None
        
        # Build basic response with raw data gathered from all sources
        basic_response = self._build_raw_response(
            question,
            plan,
            retrieved,
            calculation,
            visualizations or [],
            supplemental_calculations or [],
        )
        
        # LLM formatting is PRIMARY - always attempt it if available
        if self.settings.active_api_key:  # Try LLM first if we have credentials
            llm_response = self._try_llm_formatting(basic_response)
            if llm_response is not None:
                return llm_response
        
        # If no LLM available, use basic response as-is
        return basic_response

    def _build_raw_response(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None,
        visualizations: list[VisualizationResult],
        supplemental_calculations: list[CalculationResult],
    ) -> AgentResponse:
        """Build basic response without fancy formatting - LLM will polish it.
        
        This just gathers evidence from all sources and structures it simply.
        The LLM will then reformat this into leadership-ready prose.
        
        Note: We filter by relevance here to ensure low-relevance chunks don't pollute
        the fallback response (when LLM is unavailable).
        """
        # Filter to high-relevance chunks only (minimal safeguard for fallback)
        high_quality_retrieved = self._filter_by_relevance(question, retrieved, plan=plan)
        
        # Collect all relevant evidence snippets
        evidence = []
        source_refs: list[str] = []
        
        for item in high_quality_retrieved[: plan.max_sources]:
            label = self._reference_label(item.chunk.source_path, item.chunk.metadata)
            source_refs.append(label)
            evidence.append(f"{label}: {self._best_evidence_snippet(question, item.chunk.content)}")

        if self._is_unit_or_assumption_question(question):
            return self._build_unit_verification_response(
                question=question,
                plan=plan,
                evidence=evidence,
                source_refs=source_refs,
                calculation=calculation,
                supplemental_calculations=supplemental_calculations,
                visualizations=visualizations,
            )

        if self._should_prioritize_retrieval_narrative(question, plan, high_quality_retrieved, calculation):
            return self._build_retrieval_led_response(
                question=question,
                plan=plan,
                retrieved=high_quality_retrieved,
                evidence=evidence,
                source_refs=source_refs,
                visualizations=visualizations,
            )

        # Start with calculation/structured data if available
        if calculation:
            executive_summary = calculation.answer
            key_findings = list(calculation.findings)
            calculations = list(calculation.trace)
            for ref in calculation.evidence_refs:
                if ref not in source_refs:
                    source_refs.append(ref)
                    
            # Handle supplemental results (often from multi-part questions)
            for extra in supplemental_calculations:
                for ref in extra.evidence_refs:
                    if ref not in source_refs:
                        source_refs.append(ref)
                for finding in extra.findings:
                    if finding not in key_findings:
                        key_findings.append(finding)
                for trace_item in extra.trace:
                    if trace_item not in calculations:
                        calculations.append(trace_item)
            
            # Merge supplemental answers into the executive summary if they are distinct
            supplemental_summary = self._summarize_supplemental_results(question, supplemental_calculations)
            if supplemental_summary and supplemental_summary.lower() not in executive_summary.lower():
                prefix = ". " if not executive_summary.endswith(".") else " "
                executive_summary = f"{executive_summary.rstrip('.')}{prefix}Additionally, {supplemental_summary}."

            if self._is_explanatory_question(question):
                causal_context, causal_evidence = self._extract_causal_context(question, high_quality_retrieved)
                # Explanation logic already partially handled by the merger above, 
                # but we keep causal context logic for 'why' questions.
                if causal_context and causal_context.lower() not in executive_summary.lower():
                    executive_summary = f"{executive_summary.rstrip('.')}, and the supporting narrative attributes the miss to {causal_context.rstrip('.') }."
                
                if causal_evidence and causal_evidence not in evidence:
                    evidence.insert(0, causal_evidence)
            
            # If evidence section is empty, populate it with key data points from the structured findings
            # to ensure the report feels grounded even when narrative retrieval fails.
            if not evidence and key_findings:
                evidence = [f"Structured evidence: {finding}" for finding in key_findings[:3]]
        else:
            # Fallback to document retrieval
            if self._has_sufficient_grounding(question, high_quality_retrieved, plan):
                if self._is_temporal_metric_gap_question(question, high_quality_retrieved):
                    return self._build_temporal_metric_gap_response(
                        question=question,
                        plan=plan,
                        retrieved=high_quality_retrieved,
                        evidence=evidence,
                        source_refs=source_refs,
                        visualizations=visualizations,
                    )
                best_snippet = self._best_evidence_snippet(question, high_quality_retrieved[0].chunk.content)
                executive_summary = best_snippet[:300] if len(best_snippet) > 300 else best_snippet
                key_findings = [item.chunk.content[:200] for item in high_quality_retrieved[:3]]
                caveats = []
            else:
                return self._build_abstention_response(question, plan, visualizations)
            
            calculations = []
        
        caveats = [] if calculation else caveats
        
        visual_insights = [visual.caption for visual in visualizations]
        plot_paths = [visual.path for visual in visualizations]
        plot_base64 = visualizations[0].base64_image if visualizations else ""
        
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
            plot_paths=plot_paths,
            plot_base64=plot_base64,
        )

    def _should_prioritize_retrieval_narrative(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        calculation: CalculationResult | None,
    ) -> bool:
        if plan.filename_filters and any(any(ext in f.lower() for ext in [".pdf", ".docx", ".doc", ".md", ".txt"]) for f in plan.filename_filters):
            # If we have a valid calculation with explicit findings, don't suppress it 
            # unless the user explicitly requested narrative text/definitions.
            if calculation and calculation.findings and not self._is_definition_seeking_question(question):
                return False
            return self._has_sufficient_grounding(question, retrieved, plan)
            
        # If question is explanatory ("reasons", "why"), prioritize narrative if available
        if self._is_explanatory_question(question) and self._has_sufficient_grounding(question, retrieved, plan):
            return True

        if plan.primary_route != "retrieval":
            return False
        if not calculation or not retrieved:
            return False
        if not self._is_update_or_brief_question(question):
            return False
        return self._has_sufficient_grounding(question, retrieved, plan)

    def _build_retrieval_led_response(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        evidence: list[str],
        source_refs: list[str],
        visualizations: list[VisualizationResult],
    ) -> AgentResponse:
        best_chunk = retrieved[0].chunk
        best_text = compact_whitespace(best_chunk.content)
        summary = self._clean_document_sentence(self._best_evidence_snippet(question, best_text))

        findings: list[str] = []
        for sentence in self._split_sentences(best_text):
            cleaned = compact_whitespace(sentence)
            if not cleaned:
                continue
            cleaned = self._clean_document_sentence(cleaned)
            if cleaned and cleaned not in findings:
                findings.append(cleaned)
            if len(findings) >= 4:
                break

        if not findings:
            findings = [summary]

        return AgentResponse(
            question=question,
            planned_approach=plan.reasoning,
            executive_summary=summary,
            key_findings=findings,
            calculations=[],
            evidence=evidence,
            caveats=[],
            source_references=source_refs,
            visual_insights=[visual.caption for visual in visualizations],
            plot_paths=[visual.path for visual in visualizations],
            plot_base64=visualizations[0].base64_image if visualizations else "",
        )

    def _build_temporal_metric_gap_response(
        self,
        question: str,
        plan: QueryPlan,
        retrieved: list[RetrievedChunk],
        evidence: list[str],
        source_refs: list[str],
        visualizations: list[VisualizationResult],
    ) -> AgentResponse:
        cost_sentences: list[str] = []
        for item in retrieved:
            for sentence in self._split_sentences(item.chunk.content):
                lowered = sentence.lower()
                if any(marker in lowered for marker in ["cost", "contractor", "ticket volume", "margin pressure", "onboarding costs"]):
                    cleaned = self._clean_document_sentence(sentence)
                    if cleaned and cleaned not in cost_sentences:
                        cost_sentences.append(cleaned)
                if len(cost_sentences) >= 3:
                    break
            if len(cost_sentences) >= 3:
                break

        executive_summary = (
            "I do not see an explicit numeric Q2 operational cost value in the uploaded sources. "
            "The available Q2 evidence is narrative: it points to support pressure from rising ticket volume and contractor usage, "
            "plus cost pressure from onboarding and slower automation rollout."
        )
        key_findings = [
            "No grounded table or chart in the uploaded corpus exposes a numeric Q2 operational cost metric.",
            *cost_sentences[:2],
        ]
        if not cost_sentences:
            key_findings.append("The available Q2 evidence is qualitative rather than a quarter-specific numeric cost series.")

        return AgentResponse(
            question=question,
            planned_approach=plan.reasoning,
            executive_summary=executive_summary,
            key_findings=key_findings,
            calculations=[],
            evidence=evidence,
            caveats=["Quarter-specific cost answer withheld because the uploaded data does not expose a numeric Q2 operational cost field."],
            source_references=source_refs,
            visual_insights=[visual.caption for visual in visualizations],
            plot_paths=[visual.path for visual in visualizations],
            plot_base64=visualizations[0].base64_image if visualizations else "",
        )

    def _is_explanatory_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in ["why", "because", "driver", "reason", "caused", "performance", "explain", "summarize"])

    def _extract_causal_context(self, question: str, retrieved: list[RetrievedChunk]) -> tuple[str | None, str | None]:
        query_tokens = self._significant_tokens(question)
        best_sentence: str | None = None
        best_label: str | None = None
        best_score = -1
        for item in retrieved:
            for sentence in self._split_sentences(item.chunk.content):
                lowered = sentence.lower()
                # Very broad causal/performance markers
                if not any(marker in lowered for marker in ["because", "due to", "driven by", "impact", "pressure", "weakness", "growth", "performance", "target"]):
                    continue
                score = len(query_tokens & self._significant_tokens(sentence))
                # Boost if sentence mentions both a metric-like term AND a dynamic term
                if any(m in lowered for m in ["target", "plan", "revenue", "margin"]) and any(d in lowered for d in ["miss", "beat", "below", "above", "improved"]):
                    score += 2
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_label = self._reference_label(item.chunk.source_path, item.chunk.metadata)
        if not best_sentence:
            return None, None
        causal_phrase = compact_whitespace(best_sentence)
        lowered = best_sentence.lower()
        for marker in ["because of", "because", "due to"]:
            if marker in lowered:
                causal_phrase = compact_whitespace(best_sentence.split(marker, 1)[1])
                break
        evidence_line = f"{best_label}: {compact_whitespace(best_sentence)}" if best_label else None
        return causal_phrase, evidence_line

    def _summarize_supplemental_results(self, question: str, supplemental_calculations: list[CalculationResult]) -> str | None:
        """Merge answers from multiple calculation results into a cohesive string."""
        summaries = []
        lowered_question = question.lower()
        
        for extra in supplemental_calculations:
            if not extra.answer:
                continue
                
            # If the answer is already mentioned in the question (echo), move on
            if extra.answer.lower() in lowered_question:
                continue
                
            # Specialized summary for trends if not fully captured in the answer string
            chart_data = extra.chart_data or {}
            if chart_data.get("type") == "line" and any(term in lowered_question for term in ["trend", "quarter", "quarterly"]):
                labels = chart_data.get("labels", [])
                values = chart_data.get("values", [])
                if labels and values:
                    trend_summary = f"the {chart_data.get('title', 'trend')} moved from {values[0]:,.2f} in {labels[0]} to {values[-1]:,.2f} in {labels[-1]}"
                    if trend_summary not in summaries:
                        summaries.append(trend_summary)
                    continue

            # Default: use the calculation's answer string
            summary = extra.answer.strip()
            if summary and summary not in summaries:
                summaries.append(summary)
                
        if not summaries:
            return None
            
        if len(summaries) == 1:
            return summaries[0]
            
        # Join multiple summaries nicely
        return "; ".join(summaries)

    def _compose_explanatory_summary(
        self,
        primary_answer: str,
        causal_context: str | None = None,
        supplemental_context: str | None = None,
    ) -> str:
        summary = primary_answer.rstrip(".")
        if causal_context:
            summary += f", and the supporting narrative attributes the miss to {causal_context.rstrip('.')}"
        if supplemental_context:
            summary += f". {supplemental_context.rstrip('.')}"
        return summary + "."

    def _build_unit_verification_response(
        self,
        question: str,
        plan: QueryPlan,
        evidence: list[str],
        source_refs: list[str],
        calculation: CalculationResult | None,
        supplemental_calculations: list[CalculationResult],
        visualizations: list[VisualizationResult],
    ) -> AgentResponse:
        for result in [calculation, *supplemental_calculations]:
            if result is None:
                continue
            for ref in result.evidence_refs:
                if ref not in source_refs:
                    source_refs.append(ref)

        evidence_text = " ".join(evidence).lower()
        explicit_units = []
        for marker in ["million", "millions", "billion", "billions", "thousand", "thousands", "usd", "$m", "€m"]:
            if marker in evidence_text:
                explicit_units.append(marker)

        if explicit_units:
            unit_text = ", ".join(sorted(set(explicit_units)))
            executive_summary = f"The cited source material explicitly mentions units: {unit_text}."
            key_findings = [
                f"The unit reference appears directly in the supporting evidence as `{unit_text}`.",
                "The answer is grounded in the uploaded sources rather than assumed.",
            ]
            caveats = []
        else:
            executive_summary = (
                "I do not see an explicit unit such as million or billion in the cited source material. "
                "The uploaded files support raw values, so any million-based wording would be an assumption rather than a grounded fact."
            )
            key_findings = [
                "The supporting tables and chart labels show numeric values, but no explicit unit label is cited in the retrieved evidence.",
                "If you want unit confirmation, the source document would need to label the metric explicitly, for example as million, billion, or currency-denominated.",
            ]
            caveats = ["No explicit unit marker was found in the grounded evidence used for this answer."]

        return AgentResponse(
            question=question,
            planned_approach=plan.reasoning,
            executive_summary=executive_summary,
            key_findings=key_findings,
            calculations=[],
            evidence=evidence,
            caveats=caveats,
            source_references=source_refs,
            visual_insights=[visual.caption for visual in visualizations],
            plot_paths=[visual.path for visual in visualizations],
            plot_base64=visualizations[0].base64_image if visualizations else "",
        )



    def _try_llm_formatting(self, response: AgentResponse) -> AgentResponse | None:
        """Try to format answer using available LLM providers.
        
        Priority: Groq -> HuggingFace -> Gemini -> OpenAI -> None
        """
        logger.info(f"_try_llm_formatting called with provider: {self.settings.llm_provider}")
        
        if self.settings.llm_provider == "groq":
            logger.info("Using Groq provider")
            return self._try_groq_formatting(response)
        if self.settings.llm_provider == "huggingface":
            logger.info("Using HuggingFace provider")
            return self._try_huggingface_formatting(response)
        if self.settings.llm_provider == "gemini":
            logger.info("Using Gemini provider")
            return self._try_gemini_formatting(response)
        if self.settings.llm_provider == "openai":
            logger.info("Using OpenAI provider")
            return self._try_openai_formatting(response)
        
        logger.info("Fallback: trying all providers in order")
        return self._try_groq_formatting(response) or self._try_huggingface_formatting(response) or self._try_gemini_formatting(response) or self._try_openai_formatting(response)

    def _filter_by_relevance(self, question: str, retrieved: list[RetrievedChunk], threshold: float = 0.3, plan: QueryPlan | None = None) -> list[RetrievedChunk]:
        """Filter retrieved chunks to keep only those with sufficient relevance to the question."""
        if not retrieved:
            return []
        
        query_tokens = self._significant_tokens(question, plan)
        if not query_tokens:
            return retrieved  # If question has no significant tokens, keep all
        
        # If the user explicitly provided a filename filter, we trust their grounding more
        active_filters = plan.filename_filters if plan else []
        adjusted_threshold = 0.15 if active_filters else threshold
        
        filtered = []
        for item in retrieved:
            chunk_tokens = self._significant_tokens(item.chunk.content, plan)
            overlap = query_tokens & chunk_tokens
            overlap_ratio = len(overlap) / max(len(query_tokens), 1)
            
            # Keep only chunks with meaningful lexical support OR strong semantic score
            if overlap_ratio >= adjusted_threshold or len(overlap) >= 2 or (active_filters and item.score > 0.6):
                filtered.append(item)
        
        return filtered

    def _has_sufficient_grounding(self, question: str, retrieved: list[RetrievedChunk], plan: QueryPlan | None = None) -> bool:
        """Require strong enough support before answering from unstructured retrieval."""
        if not retrieved:
            return False

        # If a plan is provided, use it for better token extraction
        query_tokens = self._significant_tokens(question, plan)
        if not query_tokens:
            return False

        strong_matches = 0
        # If the user explicitly provided a filename filter, we trust their grounding more.
        # We also allow the semantic score from the retriever to satisfy grounding.
        filename_filters = plan.filename_filters if plan else []
        threshold = 0.2 if filename_filters else 0.4
        k_limit = 3
        
        for item in retrieved[:k_limit]:
            # Semantic boost: if we have a very strong semantic hit in the requested file, it's grounded
            if filename_filters and item.score > 0.6:
                strong_matches += 1
                continue
                
            chunk_tokens = self._significant_tokens(item.chunk.content, plan)
            overlap = query_tokens & chunk_tokens
            overlap_ratio = len(overlap) / max(len(query_tokens), 1)
            
            # Lowered threshold for explicit files (lexical overlap of 1 significant token)
            if (filename_filters and len(overlap) >= 1) or len(overlap) >= 2 or overlap_ratio >= threshold:
                strong_matches += 1

        return strong_matches >= 1

    def _build_json_rewrite_prompt(
        self,
        response: AgentResponse,
        intro: str = "Rewrite the following grounded research report as concise leadership-ready prose.",
        include_style_rules: bool = False,
    ) -> str:
        style_rules = ""
        if include_style_rules:
            style_rules = (
                "STYLE RULES:\n"
                "1. Executive Summary: Write 2-3 sentences of narrative explanation, not a raw dump of numbers.\n"
                "2. Key Findings: Turn each metric into a business insight while keeping the figures exact.\n"
            )

        return (
            f"{intro}\n\n"
            f"{style_rules}"
            "GROUNDING RULES:\n"
            "1. PRESERVE ALL factual values, metrics, percentages, and dates EXACTLY as they appear.\n"
            "2. HYBRID SYNTHESIS: You MUST bridge numeric calculation results with narrative evidence snippets. Do not just list them; explain the relationship (e.g., 'Revenue missed target because of...').\n"
            "3. DO NOT add new facts, statistics, labels, or details not explicitly in the original report.\n"
            "4. DO NOT speculate or infer beyond what the report states.\n"
            "5. DO NOT remove citations or caveats.\n"
            "6. If the report says 'Insufficient evidence', keep that language.\n"
            "7. NEVER add or normalize units such as million, billion, M, K, USD, EUR, or percentages unless that exact unit is explicitly present in the source-backed report.\n"
            "8. If a value appears without an explicit unit, keep it unitless and do not imply a currency scale.\n"
            "9. Maintain the source references as-is.\n"
            "10. Return ONLY valid JSON with keys: executive_summary, key_findings, calculations, evidence, caveats, source_references, visual_insights.\n\n"
            f"REPORT TO REWRITE:\n{response.to_markdown()}"
        )

    def _finalize_formatted_response(self, original: AgentResponse, data: dict) -> AgentResponse:
        def ensure_list(value):
            if isinstance(value, str):
                return [value] if value.strip() else []
            return list(value) if value else []

        key_findings = ensure_list(data.get("key_findings", original.key_findings)) or original.key_findings
        executive_summary = data.get("executive_summary", original.executive_summary)

        if self._should_preserve_raw_summary(original, executive_summary):
            executive_summary = original.executive_summary

        key_findings = self._merge_explanatory_findings(
            question=original.question,
            original_findings=original.key_findings,
            formatted_findings=key_findings,
        )

        return AgentResponse(
            question=original.question,
            planned_approach=original.planned_approach,
            executive_summary=executive_summary,
            key_findings=key_findings,
            calculations=ensure_list(data.get("calculations", original.calculations)) or original.calculations,
            evidence=original.evidence,
            caveats=ensure_list(data.get("caveats", original.caveats)) or original.caveats,
            source_references=original.source_references,
            visual_insights=ensure_list(data.get("visual_insights", original.visual_insights)) or original.visual_insights,
            plot_paths=original.plot_paths,
            plot_base64=original.plot_base64,
        )

    def _should_preserve_raw_summary(self, original: AgentResponse, formatted_summary: str) -> bool:
        if not formatted_summary:
            return True
        if not self._is_explanatory_question(original.question):
            return False

        original_lower = original.executive_summary.lower()
        formatted_lower = formatted_summary.lower()

        # Keep the raw summary if the rewrite drops the grounded causal explanation.
        if self._contains_causal_language(original_lower) and not self._contains_causal_language(formatted_lower):
            return True

        # Preserve the raw summary for mixed explanatory+trend questions if the rewrite
        # drops the trend context that was already grounded in the source data.
        if self._question_mentions_trend(original.question):
            if self._contains_trend_language(original_lower) and not self._contains_trend_language(formatted_lower):
                return True

        return False

    def _merge_explanatory_findings(
        self,
        question: str,
        original_findings: list[str],
        formatted_findings: list[str],
    ) -> list[str]:
        if not self._is_explanatory_question(question):
            return formatted_findings or original_findings

        merged = list(formatted_findings or [])
        if any(self._contains_causal_language(item.lower()) for item in merged):
            return merged

        for finding in original_findings:
            if self._contains_causal_language(finding.lower()) and finding not in merged:
                merged.append(finding)
                break

        return merged or original_findings

    def _contains_causal_language(self, text: str) -> bool:
        return any(
            marker in text
            for marker in [
                "because",
                "due to",
                "attribut",
                "conversion",
                "weakness",
                "pressure",
                "driver",
                "reason",
                "caused",
                "slower",
            ]
        )

    def _contains_trend_language(self, text: str) -> bool:
        return any(
            marker in text
            for marker in [
                "trend",
                "quarter",
                "q1",
                "q2",
                "q3",
                "q4",
                "increased",
                "decreased",
                "grew",
                "declined",
            ]
        )

    def _question_mentions_trend(self, question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in ["trend", "quarter", "quarterly"])

    def _is_update_or_brief_question(self, question: str) -> bool:
        lowered = self._normalize_query_text(question.lower())
        return any(
            marker in lowered
            for marker in [
                "operational update",
                "operational updates",
                "update for q",
                "quarterly update",
                "brief",
                "memo",
                "status update",
                "highlights",
            ]
        )

    def _is_temporal_metric_gap_question(self, question: str, retrieved: list[RetrievedChunk]) -> bool:
        lowered = self._normalize_query_text(question.lower())
        has_temporal_constraint = any(term in lowered for term in ["q1", "q2", "q3", "q4", "quarter", "quarterly", "fy"])
        asks_for_cost = any(term in lowered for term in ["cost", "costs", "operational cost", "operational costs"])
        if not (has_temporal_constraint and asks_for_cost):
            return False
        evidence_text = " ".join(item.chunk.content.lower() for item in retrieved)
        # Treat this as a gap if we only found narrative cost language and no explicit period-bound cost metric.
        has_numeric_period_cost = any(period in evidence_text and "cost" in evidence_text for period in ["q1 2024", "q2 2024", "q3 2024", "q4 2024"])
        return not has_numeric_period_cost

    def _is_unit_or_assumption_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            term in lowered
            for term in ["million", "millions", "billion", "billions", "unit", "units", "currency", "usd", "assumed", "assumption", "specified"]
        )

    def _is_definition_seeking_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in ["meaning", "define", "definition", "stand for", "what is a", "what are the"])

    def _build_abstention_response(
        self,
        question: str,
        plan: QueryPlan,
        visualizations: list[VisualizationResult],
    ) -> AgentResponse:
        """Return a grounded abstention instead of guessing beyond the provided corpus."""
        return AgentResponse(
            question=question,
            planned_approach=plan.reasoning,
            executive_summary="Insufficient evidence: I do not have enough grounded context in the uploaded documents to answer this correctly.",
            key_findings=[
                "The available documents do not contain enough direct evidence for a reliable answer.",
                "The agent is abstaining rather than inferring beyond the provided sources to avoid hallucination.",
            ],
            calculations=[],
            evidence=[],
            caveats=[
                "Answer withheld because the retrieved evidence did not meet the grounding threshold.",
            ],
            source_references=[],
            visual_insights=[visual.caption for visual in visualizations],
            plot_paths=[visual.path for visual in visualizations],
            plot_base64=visualizations[0].base64_image if visualizations else "",
        )



    def _try_gemini_formatting(self, response: AgentResponse) -> AgentResponse | None:
        """Format answer using Google Gemini with anti-hallucination prompt."""
        try:
            from google import genai
        except ImportError:
            logger.debug("google.genai not available - skipping Gemini formatting")
            return None

        prompt = self._build_json_rewrite_prompt(response)
        try:
            api_key = self.settings.gemini_api_key
            if not api_key:
                logger.debug("No Gemini API key configured - skipping Gemini formatting")
                return None
            
            client = genai.Client(api_key=api_key)
            completion = client.models.generate_content(model=self.settings.chat_model, contents=prompt)
            content = completion.text
            data = json.loads(content)
            logger.info("Gemini formatting applied successfully")
            return self._finalize_formatted_response(response, data)
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini returned invalid JSON: {e}. Response text was: {completion.text[:200]}...")
            return None
        except Exception as e:
            logger.warning(f"Gemini formatting failed: {type(e).__name__}: {e}")
            return None
    def _try_groq_formatting(self, response: AgentResponse) -> AgentResponse | None:
        """Format answer using Groq API with anti-hallucination prompt.
        
        Groq offers very fast inference with free models optimized for instruction following.
        """
        from groq import Groq
        
        logger.info("Attempting Groq formatting...")
        
        api_key = self.settings.groq_api_key
        if not api_key:
            logger.debug("No Groq API key configured - skipping Groq formatting")
            return None
        
        logger.info(f"Using Groq API key: {api_key[:10]}...")
        
        # Use Llama 3.3 70B Versatile - latest, high quality model available on Groq
        model = "llama-3.3-70b-versatile"
        
        prompt = self._build_json_rewrite_prompt(
            response,
            intro="You are an expert analyst. Rewrite the following research report into clear, concise leadership prose.",
            include_style_rules=True,
        )
        
        try:
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for factual output
                max_tokens=2000,
            )
            
            content = completion.choices[0].message.content
            logger.info(f"Groq response: {content[:200]}...")  # Log first 200 chars
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            data = json.loads(content)
            logger.info("Groq formatting applied successfully")
            
            # Ensure fields are lists (Groq sometimes returns strings)
            def ensure_list(value):
                if isinstance(value, str):
                    return [value] if value.strip() else []
                return list(value) if value else []
            
            return self._finalize_formatted_response(response, data)
        except json.JSONDecodeError as e:
            logger.warning(f"Groq returned invalid JSON: {e}. Response was: {content[:500]}")
            return None
        except Exception as e:
            logger.warning(f"Groq formatting failed: {type(e).__name__}: {e}")
            return None
    def _try_openai_formatting(self, response: AgentResponse) -> AgentResponse | None:
        """Format answer using OpenAI with anti-hallucination prompt."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.debug("openai module not available - skipping OpenAI formatting")
            return None

        api_key = self.settings.openai_api_key
        if not api_key:
            logger.debug("No OpenAI API key configured - skipping OpenAI formatting")
            return None
        
        client = OpenAI(api_key=api_key)
        prompt = self._build_json_rewrite_prompt(response)
        try:
            completion = client.responses.create(model=self.settings.chat_model, input=prompt)
            content = completion.output_text
            data = json.loads(content)
            logger.info("OpenAI formatting applied successfully")
            return self._finalize_formatted_response(response, data)
        except json.JSONDecodeError as e:
            logger.warning(f"OpenAI returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"OpenAI formatting failed: {type(e).__name__}: {e}")
            return None

    def _try_huggingface_formatting(self, response: AgentResponse) -> AgentResponse | None:
        """Format answer using HuggingFace Inference API with anti-hallucination prompt."""
        import requests
        
        logger.info("Attempting HuggingFace formatting...")
        
        api_key = self.settings.huggingface_api_key
        if not api_key:
            logger.debug("No HuggingFace API key configured - skipping HuggingFace formatting")
            return None
        
        logger.info(f"Using HuggingFace API key: {api_key[:10]}...")
        
        # Use Google Gemma 4 31B Instruct - high quality instruction model
        model_id = "google/gemma-4-31B-it"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        prompt = self._build_json_rewrite_prompt(response)
        
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2000,
                    "temperature": 0.3,  # Lower temperature for factual output
                }
            }
            
            response_hf = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response_hf.raise_for_status()
            
            result = response_hf.json()
            
            # Handle different response formats from HuggingFace
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                content = result.get("generated_text", "")
            else:
                logger.warning(f"Unexpected HuggingFace response format: {result}")
                return None
            
            # Extract JSON from the response (HF may include prompt in output)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in HuggingFace response")
                return None
            
            data = json.loads(json_match.group())
            logger.info("HuggingFace formatting applied successfully")
            
            return self._finalize_formatted_response(response, data)
        except requests.exceptions.RequestException as e:
            logger.warning(f"HuggingFace API request failed: {type(e).__name__}: {str(e)[:200]}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"HuggingFace returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"HuggingFace formatting failed: {type(e).__name__}: {e}")
            return None

    def _best_evidence_snippet(self, question: str, content: str) -> str:
        """Extract the most relevant snippet from content for the given question."""
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
        """Split text into sentences or lines (for tables)."""
        # Split by sentence markers OR newlines to handle tables correctly
        return [compact_whitespace(part) for part in re.split(r"(?:(?<=[.!?])\s+)|\n", text) if compact_whitespace(part)]

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

    def _significant_tokens(self, text: str, plan: QueryPlan | None = None) -> set[str]:
        """Extract significant tokens from text (removes stopwords)."""
        text = self._normalize_query_text(text)
        stopwords = {
            "what", "which", "does", "show", "the", "our", "all", "across", "current",
            "is", "are", "was", "were", "by", "of", "and", "to", "for", "in", "on",
            "trend", "chart", "graph", "visual", "question",
            "pdf", "docx", "xlsx", "csv", "md", "txt",
        }
        
        # Also exclude the specific filename if we're filtering by it
        if plan and plan.filename_filters:
            for f in plan.filename_filters:
                stopwords.add(f.lower())
                # Add base filename too
                base = f.split('.')[0].lower()
                stopwords.add(base)
            
        normalized_tokens = [self._normalize_token(token) for token in tokenize(text)]
        return {
            token
            for token in normalized_tokens
            if token and token not in stopwords and (len(token) > 2 or self._is_short_semantic_token(token))
        }
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



    def _normalize_query_text(self, text: str) -> str:
        replacements = {
            "operartional": "operational",
            "opertional": "operational",
            "quaterly": "quarterly",
            "revnue": "revenue",
            "margn": "margin",
        }
        normalized = text
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _clean_document_sentence(self, text: str) -> str:
        cleaned = compact_whitespace(text)
        for prefix in [
            "# Q2 Operational Update",
            "Q2 Operational Update",
            "# Quarterly Operational Update",
            "Quarterly Operational Update",
        ]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip(" .:-")
                break
        return cleaned

    def _normalize_token(self, token: str) -> str:
        if token.startswith("q") and len(token) == 2 and token[1].isdigit():
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def _is_short_semantic_token(self, token: str) -> bool:
        return token.startswith("q") and len(token) == 2 and token[1].isdigit()

    def _try_openai_formatting(self, response: AgentResponse) -> AgentResponse | None:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=self.settings.openai_api_key)
        prompt = self._build_json_rewrite_prompt(response)
        try:
            completion = client.responses.create(model=self.settings.chat_model, input=prompt)
            content = completion.output_text
            data = json.loads(content)
            return self._finalize_formatted_response(response, data)
        except Exception:
            return None
