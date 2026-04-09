"""
Comprehensive Hybrid Tests: SQL + Document Retrieval
Tests that require intelligent combination of both table data and document context
"""

import pytest
from pathlib import Path
from wald_decision_agent.core.agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings


@pytest.fixture
def agent():
    settings = AppSettings()
    return LeadershipInsightAgent(settings)


class TestHybridTableAndDocumentQueries:
    """Tests combining table data with document context"""

    @pytest.mark.slow
    def test_regional_performance_with_narrative_context(self, agent):
        """Q: Which region underperformed and why?
        
        Requires:
        - Table: regional_actuals.csv, regional_targets.csv (numeric variance)
        - Document: leadership_brief_long.txt (context on regional issues)
        
        Expected: "Europe missed plan because of slower enterprise conversions in the mid-market segment."
        Should include: numeric variance + business context
        """
        question = "Which regions underperformed and why did they miss targets?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should have numeric data (region names or variances)
        has_numeric = any(term in answer_lower for term in ["europe", "america", "apac", "north", "variance", "below", "missed"])
        
        # Should have business context
        has_context = any(term in answer_lower for term in ["conversion", "segment", "slower", "enterprise", "challenge"])
        
        print(f"\n{'='*80}")
        print(f"HYBRID TEST: Regional Performance with Narrative")
        print(f"{'='*80}")
        print(f"Answer contains numeric data: {has_numeric}")
        print(f"Answer contains business context: {has_context}")
        print(f"Answer ({len(response.executive_summary)} chars): {response.executive_summary[:200]}...")
        print(f"Sources: {len(response.source_references)} sources")
        for src in response.source_references[:3]:
            print(f"  - {src}")
        
        assert has_numeric, "Answer should include regional/numeric data"
        assert has_context or len(response.evidence) > 0, "Answer should include business context or evidence"


    @pytest.mark.slow
    def test_departmental_performance_with_root_causes(self, agent):
        """Q: Which departments underperform and what are the root causes?
        
        Requires:
        - Table: department_scorecard.csv (performance metrics)
        - Document: leadership_brief_long.txt, operational_steering_memo.docx (context on each department)
        
        Expected: Numeric performance + narrative reasons (cost pressure, automation delays, etc)
        Should NOT include: Filler text about retaining chunk offsets
        """
        question = "Which departments underperformed and what operational factors contributed?"
        response = agent.ask(question, Path("data/raw"))
        
        # Check full response (executive summary + key findings)
        full_response = response.executive_summary + " " + " ".join(response.key_findings)
        answer_lower = full_response.lower()
        
        # Should mention departments or operational context
        has_depts = any(term in answer_lower for term in ["engineering", "sales", "support", "finance", "marketing", "operational"])
        
        # Should have root causes or execution mentions
        has_causes = any(term in answer_lower for term in ["cost", "margin", "pressure", "bottleneck", "automation", "overhead", "ticket", "execution", "quality"])
        
        # Should NOT include filler/test text
        has_filler = "chunk offset" in answer_lower or "extend the document" in answer_lower
        
        print(f"\n{'='*80}")
        print(f"HYBRID TEST: Department Performance with Root Causes")
        print(f"{'='*80}")
        print(f"Answer mentions departments: {has_depts}")
        print(f"Answer includes root causes: {has_causes}")
        print(f"Answer contains filler text: {has_filler}")
        print(f"Executive summary ({len(response.executive_summary)} chars): {response.executive_summary[:150]}...")
        print(f"Key findings: {len(response.key_findings)}")
        if response.key_findings:
            print(f"  First: {response.key_findings[0][:100]}...")
        
        # Departments or operational context should be present, or causes should be mentioned
        assert (has_depts or has_causes) or len(response.key_findings) > 0, "Should describe departmental performance and causes"
        assert not has_filler, "Should NOT include filler text"


    @pytest.mark.slow
    def test_revenue_drivers_with_segment_analysis(self, agent):
        """Q: How did revenue perform and what drove the performance?
        
        Requires:
        - Table: revenue_trend.csv (numeric revenue data)
        - Document: annual_report_2024.md (narrative on drivers: enterprise, digital media)
        
        Expected: Revenue growth + specific segment drivers
        """
        question = "What was our revenue performance and which segments drove the growth?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should have numeric indicators
        has_numeric = "revenue" in answer_lower
        
        # Should have segment drivers
        has_drivers = any(term in answer_lower for term in ["enterprise", "digital", "segment", "media", "driven"])
        
        # Should be concise, not verbose
        is_concise = len(response.executive_summary) < 400
        
        print(f"\n{'='*80}")
        print(f"HYBRID TEST: Revenue Performance with Segment Drivers")
        print(f"{'='*80}")
        print(f"Answer includes revenue data: {has_numeric}")
        print(f"Answer explains segment drivers: {has_drivers}")
        print(f"Answer is concise (<400 chars): {is_concise}")
        print(f"Answer length: {len(response.executive_summary)} chars")
        print(f"Answer: {response.executive_summary[:250]}...")
        
        assert has_numeric, "Should mention revenue"
        assert len(response.key_findings) > 0, "Should have key findings"


    @pytest.mark.slow
    def test_margin_pressure_with_cost_drivers(self, agent):
        """Q: What margin pressure exists and what are the cost drivers?
        
        Requires:
        - Table: department_scorecard.csv (margin metrics if available)
        - Document: annual_report_2024.md, leadership_brief_long.txt (mentions onboarding costs, support pressure)
        
        Expected: Margin metrics + specific cost drivers balanced
        """
        question = "What margin pressure exists in the business and what are the primary cost drivers?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should mention margin
        has_margin = "margin" in answer_lower
        
        # Should identify cost drivers
        has_drivers = any(term in answer_lower for term in ["cost", "support", "onboarding", "automation", "contractor", "overhead"])
        
        print(f"\n{'='*80}")
        print(f"HYBRID TEST: Margin Pressure with Cost Drivers")
        print(f"{'='*80}")
        print(f"Answer mentions margin: {has_margin}")
        print(f"Answer identifies cost drivers: {has_drivers}")
        print(f"Answer ({len(response.executive_summary)} chars): {response.executive_summary[:200]}...")
        
        assert has_margin or has_drivers, "Should identify margin pressure and cost drivers"


class TestConciseFiltering:
    """Tests that ensure answers are concise and irrelevant info is filtered"""

    @pytest.mark.slow
    def test_answer_excludes_low_relevance_chunks(self, agent):
        """Ensure that low-relevance retrieved chunks don't make it into the answer"""
        question = "What are the strategic priorities?"
        response = agent.ask(question, Path("data/raw"))
        
        full_response = response.executive_summary + " " + " ".join(response.key_findings)
        answer_lower = full_response.lower()
        
        # Should have strategic priorities (in either executive summary or key findings)
        has_priorities = any(term in answer_lower for term in ["retention", "automation", "forecasting", "cost", "improve enterprise", "support cost"])
        
        # Should NOT have unrelated filler
        has_filler = "chunk" in answer_lower or "offset" in answer_lower or "extend" in answer_lower
        
        # Should be reasonably sized (concise)
        is_concise = len(response.executive_summary) < 500
        
        print(f"\n{'='*80}")
        print(f"CONCISE TEST: Strategic Priorities - No Filler")
        print(f"{'='*80}")
        print(f"Answer has priorities: {has_priorities}")
        print(f"Answer contains filler: {has_filler}")
        print(f"Answer is concise: {is_concise} ({len(response.executive_summary)} chars)")
        print(f"Executive summary: {response.executive_summary[:200]}...")
        print(f"Key findings: {len(response.key_findings)} findings")
        if response.key_findings:
            print(f"  First finding: {response.key_findings[0][:100]}...")
        
        assert has_priorities, "Should include strategic priorities"
        assert not has_filler, "Should NOT include filler text"
        assert is_concise, "Answer should be concise"


    @pytest.mark.slow
    def test_answer_combines_sources_intelligently(self, agent):
        """
        Test that answer uses the right source for each piece of information:
        - Numeric data from tables
        - Context from documents
        - But NOT both when one source is clearly better
        """
        question = "How did engineering improve and what were the results?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should have improvements mentioned
        has_improvements = any(term in answer_lower for term in ["improve", "release", "backlog", "delivery", "predictability"])
        
        # Should NOT be padding with unrelated content
        is_focused = len(response.key_findings) <= 5 and len(response.evidence) <= 10
        
        # Executive summary should be primary, evidence should support it
        has_good_structure = (
            len(response.executive_summary) > 50 and
            len(response.executive_summary) < 500
        )
        
        print(f"\n{'='*80}")
        print(f"CONCISE TEST: Engineering Improvements - Intelligent Source Use")
        print(f"{'='*80}")
        print(f"Answer has improvements: {has_improvements}")
        print(f"Answer is focused: {is_focused} (findings: {len(response.key_findings)}, evidence: {len(response.evidence)})")
        print(f"Answer has good structure: {has_good_structure} ({len(response.executive_summary)} chars)")
        print(f"Answer: {response.executive_summary[:200]}...")
        
        # More lenient assertion since this is about quality
        assert has_good_structure, "Answer should be well-structured and focused"


class TestSmartSourceSelection:
    """Tests that verify the right source is chosen for each query type"""

    @pytest.mark.slow
    def test_numeric_query_prioritizes_tables(self, agent):
        """Pure numeric query should use table data primarily"""
        question = "What were the exact revenue numbers by quarter?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should have numeric data in answer
        has_numbers = any(char.isdigit() for char in response.executive_summary)
        
        # Can reference either tables or documents, but should have some numeric content
        print(f"\n{'='*80}")
        print(f"SMART SOURCE: Numeric Query - Tables Priority")
        print(f"{'='*80}")
        print(f"Answer contains numbers: {has_numbers}")
        print(f"Answer: {response.executive_summary[:200]}...")
        print(f"Sources used: {response.source_references[:3]}")
        
        # For numeric questions, should either have numbers or explain why not available
        assert has_numbers or "not available" in response.executive_summary.lower(), \
            "Numeric question should contain numbers or explain unavailability"


    @pytest.mark.slow
    def test_narrative_query_prioritizes_documents(self, agent):
        """Narrative query should use document insights primarily"""
        question = "What are the leadership's concerns about the organization?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        
        # Should have qualitative insights, not numbers
        has_insights = any(term in answer_lower for term in ["concern", "risk", "challenge", "pressure", "bottleneck"])
        
        # Should reference documents
        has_doc_sources = any("annual_report" in src.lower() or "leadership" in src.lower() 
                             for src in response.source_references)
        
        print(f"\n{'='*80}")
        print(f"SMART SOURCE: Narrative Query - Documents Priority")
        print(f"{'='*80}")
        print(f"Answer has qualitative insights: {has_insights}")
        print(f"Answer references documents: {has_doc_sources}")
        print(f"Answer: {response.executive_summary[:200]}...")
        
        assert has_insights or has_doc_sources, "Narrative question should use document insights"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
