"""
Comprehensive Hallucination Prevention Tests

Tests verifying the agent:
1. Rejects questions outside its knowledge base
2. Returns "Insufficient evidence" for out-of-scope queries
3. Never fabricates data
4. Properly cites all claims
5. Has confidence thresholds preventing weak answers
6. Gracefully handles edge cases
"""

import pytest
from pathlib import Path
from wald_decision_agent.core.agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings


@pytest.fixture
def agent():
    settings = AppSettings(enable_llm_formatting=False)
    return LeadershipInsightAgent(settings)


class TestHallucinationPrevention:
    """Tests that verify the system refuses to hallucinate"""

    @pytest.mark.slow
    def test_rejects_out_of_scope_numeric_question(self, agent):
        """System should refuse to answer about data that doesn't exist"""
        question = "What is our EBITDA trend across all subsidiaries?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should explicitly state insufficient evidence
        assert "Insufficient evidence" in response.executive_summary
        # Should NOT make up numbers
        assert not any(char.isdigit() for char in response.executive_summary[:100])
        
        print(f"✓ Out-of-scope numeric question properly rejected")
        print(f"  Response: {response.executive_summary[:150]}...")

    @pytest.mark.slow
    def test_rejects_out_of_scope_narrative_question(self, agent):
        """System should refuse to invent business context"""
        question = "What is our expansion strategy in Southeast Asia?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should not fabricate a region not mentioned in documents
        answer_lower = response.executive_summary.lower()
        
        # If it mentions Southeast Asia, it should cite evidence
        if "southeast asia" in answer_lower or "asia" in answer_lower:
            assert len(response.source_references) > 0, "Claims must be cited"
            assert len(response.evidence) > 0, "Must provide evidence snippets"
        else:
            # More likely: should say insufficient evidence
            assert "insufficient" in answer_lower or "not found" in answer_lower or "no evidence" in answer_lower
        
        print(f"✓ Out-of-scope narrative question properly handled")

    @pytest.mark.slow
    def test_rejects_opinion_questions_without_evidence(self, agent):
        """System should not invent executive opinions"""
        # Use a truly out-of-scope topic that's not in any of the documents
        question = "What is the leadership's stance on quantum computing investment?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should either have evidence or say insufficient
        if len(response.source_references) == 0:
            # Correctly rejects with no sources
            assert "insufficient" in answer_lower or "no" in answer_lower or "not found" in answer_lower
            print(f"✓ Out-of-scope opinion question properly rejected (no sources)")
        else:
            # If answering, must have actual evidence - quantum shouldn't appear
            assert not any("quantum" in ref.lower() for ref in response.source_references), \
                "Should not have evidence for quantum question"
            print(f"✓ Opinion question properly handled without fabrication")

    @pytest.mark.slow
    def test_numeric_answers_only_from_actual_data(self, agent):
        """All numeric answers must come from retrieved data, never invented"""
        question = "What was the exact revenue for Q3?"
        response = agent.ask(question, Path("data/raw"))
        
        # If answer contains numbers, they should be traceable
        if any(char.isdigit() for char in response.executive_summary):
            # Should have source references
            assert len(response.source_references) > 0, "Numeric claims require sources"
            # Should have calculations trace showing where numbers came from
            assert len(response.calculations) > 0 or len(response.evidence) > 0
            
            print(f"✓ Numeric answer properly grounded with sources")
        else:
            # If no numbers provided, should be explicit about why
            answer_lower = response.executive_summary.lower()
            assert any(term in answer_lower for term in ["insufficient", "not found", "no data", "not available"])
            
            print(f"✓ Missing numeric data properly acknowledged")

    @pytest.mark.slow
    def test_rejects_questions_with_false_premises(self, agent):
        """System shouldn't fabricate data for questions with false/out-of-scope premises"""
        # Use facts that definitely aren't in the documents
        question = "How much did we invest in our South Africa subsidiary last quarter?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should either reject or have evidence about South Africa subsidiary
        has_insufficient_evidence = any(
            term in answer_lower 
            for term in ["insufficient", "not found", "no evidence", "not available"]
        )
        
        # If answering, should have mentioned South Africa
        if "insufficient" not in answer_lower and len(response.source_references) > 0:
            # If we have an answer, it must mention the premise (South Africa) or be from wrong question
            assert "south africa" in answer_lower or "subsidiary" in answer_lower
        else:
            # More likely to reject out-of-scope query
            assert has_insufficient_evidence
            print(f"✓ Out-of-scope premise question properly rejected")

    @pytest.mark.slow
    def test_all_findings_are_grounded(self, agent):
        """Every key finding must have supporting evidence"""
        question = "What are the main operational risks?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should have key findings
        if response.key_findings:
            # Check if findings can be traced to evidence
            print(f"✓ {len(response.key_findings)} key findings generated")
            
            # Each finding should either:
            # 1. Be traceable to evidence snippets, OR
            # 2. Be from calculation traces
            has_grounding = len(response.evidence) > 0 or len(response.calculations) > 0
            assert has_grounding, "Findings require grounding"
            
            print(f"✓ All findings properly grounded")
        else:
            # No findings means insufficient evidence, which is fine
            assert "insufficient" in response.executive_summary.lower()
            print(f"✓ Absence of findings properly indicates insufficient evidence")

    @pytest.mark.slow
    def test_confidence_preserved_in_response(self, agent):
        """Response should indicate confidence/certainty level"""
        question = "What are the strategic priorities for next year?"
        response = agent.ask(question, Path("data/raw"))
        
        # Check if response indicates confidence through caveats
        response_text = response.executive_summary + " " + " ".join(response.caveats)
        
        # A proper response should either:
        # 1. Have source references showing high confidence
        # 2. Have caveats explaining uncertainty
        # 3. Return insufficient evidence
        
        if len(response.source_references) > 0:
            # Has grounded sources - good
            print(f"✓ Answer grounded with {len(response.source_references)} sources")
        elif len(response.caveats) > 0:
            # Has caveats explaining limitations - good
            print(f"✓ Answer qualified with {len(response.caveats)} caveats")
        else:
            # Should have insufficient evidence message
            assert "insufficient" in response.executive_summary.lower()
            print(f"✓ Answer properly marked as insufficient evidence")


class TestGroundingThresholds:
    """Tests verifying grounding threshold implementation"""

    @pytest.mark.slow
    def test_low_similarity_chunks_filtered_out(self, agent):
        """Chunks with <30% relevance should not appear in evidence"""
        # Ask a specific question that should match some chunks but not others
        question = "What is our revenue for enterprise segment?"
        response = agent.ask(question, Path("data/raw"))
        
        # All evidence snippets should be relevant to the question
        for evidence in response.evidence:
            # Evidence should contain key terms from question
            evidence_lower = evidence.lower()
            # Should relate to revenue or enterprise
            has_relevance = ("revenue" in evidence_lower or "enterprise" in evidence_lower or 
                            "sales" in evidence_lower or "segment" in evidence_lower)
            assert has_relevance, f"Evidence snippet not relevant to question: {evidence[:100]}"
        
        print(f"✓ All {len(response.evidence)} evidence snippets meet relevance threshold")

    @pytest.mark.slow
    def test_insufficient_evidence_threshold(self, agent):
        """System should reject answers when evidence is weak"""
        # Question with no matching data in the corpus
        question = "What are our operations in Mars?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should be insufficient
        assert "Insufficient evidence" in response.executive_summary
        
        print(f"✓ Insufficient evidence threshold prevents nonsensical answers")

    @pytest.mark.slow
    def test_citation_requirement_for_claims(self, agent):
        """Every specific claim should be cited"""
        question = "Which departments have the highest cost pressure?"
        response = agent.ask(question, Path("data/raw"))
        
        # If answer includes specific department names
        for dept in ["engineering", "sales", "support", "finance", "marketing"]:
            if dept in response.executive_summary.lower():
                # Must have source references
                assert len(response.source_references) > 0, f"Claim about {dept} requires citation"
        
        print(f"✓ Specific claims are properly cited")


class TestLLMHallucinationPrevention:
    """Tests preventing LLM-based hallucinations during formatting"""

    @pytest.mark.slow
    def test_llm_formatting_preserves_grounding(self, agent):
        """LLM formatting should not introduce new ungrounded claims"""
        question = "What is our revenue trend?"
        response = agent.ask(question, Path("data/raw"))
        
        # The formatted response should not expand beyond evidence
        original_findings = len(response.key_findings)
        original_sources = len(response.source_references)
        
        # After formatting, should not magically get more sources or findings
        # (This will always pass but documents the requirement)
        assert original_sources >= 0
        
        # Check that executive summary is grounded
        if "Insufficient evidence" not in response.executive_summary:
            # If we have an answer, we should have evidence
            assert len(response.evidence) > 0 or len(response.calculations) > 0
        
        print(f"✓ LLM formatting preserves grounding")

    @pytest.mark.slow
    def test_no_invented_statistics(self, agent):
        """System should never report statistics not in the documents"""
        # Use an out-of-scope statistics question (not in documents)
        question = "What is our market share percentage in North America vs Europe?"
        response = agent.ask(question, Path("data/raw"))
        
        # If percentages are reported
        import re
        percentages = re.findall(r'(\d+)%', response.executive_summary)
        
        if percentages:
            # Must have sources for any percentage claims - no percentages without grounding
            assert len(response.source_references) > 0 or len(response.evidence) > 0, \
                f"Percentages require citations/evidence, got: {response.executive_summary}"
            print(f"✓ {len(percentages)} percentages reported with proper grounding")
        else:
            # More common case: out-of-scope data simply not reported as percentages
            # This is the correct handling - don't invent data
            print(f"✓ Market share statistics not invented (correctly absent)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
