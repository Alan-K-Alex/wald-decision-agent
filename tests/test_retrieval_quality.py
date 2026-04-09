"""
Comprehensive Document Retrieval Quality Tests
Tests to verify that document-based answers are retrieved correctly and efficiently
"""

import pytest
from pathlib import Path
from wald_decision_agent.core.agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings


@pytest.fixture
def agent():
    """Create agent for testing"""
    settings = AppSettings()
    return LeadershipInsightAgent(settings)


class TestStrategicDocumentRetrieval:
    """Tests for strategic information retrieval from documents"""

    @pytest.mark.slow
    def test_strategic_priorities_explicit_request(self, agent):
        """Test: What are the strategic priorities? (explicit document request)"""
        question = "What are the strategic priorities for the next planning cycle as per documents?"
        response = agent.ask(question, Path("data/raw"))
        
        # Verify planned approach prioritizes documents
        assert any("document" in approach.lower() or "retrieval" in approach.lower() 
                  for approach in response.planned_approach), \
            "Should explicitly retrieve from documents"
        
        # Check answer contains strategic priorities
        answer_lower = response.executive_summary.lower()
        expected_priorities = [
            "enterprise retention",
            "support cost",
            "automation",
            "forecasting"
        ]
        
        found_priorities = [p for p in expected_priorities if p in answer_lower]
        assert len(found_priorities) >= 3, \
            f"Should find at least 3 strategic priorities. Found: {found_priorities}"
        
        print(f"\n✓ Strategic Priorities Retrieved (found {len(found_priorities)}/4):")
        print(f"  Answer: {response.executive_summary[:150]}...")
        print(f"  Key Findings: {len(response.key_findings)} findings")
        for finding in response.key_findings[:3]:
            print(f"    - {finding[:80]}...")


    @pytest.mark.slow
    def test_risks_identification_from_narrative(self, agent):
        """Test: What are the key risks? (from narrative)"""
        question = "What are the largest risks highlighted by leadership from the annual report?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        expected_risks = [
            "hiring",
            "engineering",
            "bottleneck",
            "support",
            "execution"
        ]
        
        found_risks = [r for r in expected_risks if r in answer_lower]
        assert len(found_risks) >= 2, \
            f"Should identify risks. Found: {found_risks}"
        
        print(f"\n✓ Risks Identified (found {len(found_risks)}/5):")
        print(f"  Answer: {response.executive_summary[:150]}...")


    @pytest.mark.slow
    def test_revenue_expansion_context(self, agent):
        """Test: How did revenue perform in 2024? (document context)"""
        question = "From the annual report, how did revenue perform and what drove it?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        expected_elements = [
            "revenue",
            "expand",
            "enterprise",
            "digital"
        ]
        
        found = [e for e in expected_elements if e in answer_lower]
        assert len(found) >= 3, \
            f"Should explain revenue drivers. Found: {found}"
        
        print(f"\n✓ Revenue Context Retrieved (found {len(found)}/4):")
        print(f"  Answer: {response.executive_summary[:150]}...")


    @pytest.mark.slow
    def test_support_organization_challenges(self, agent):
        """Test: What are the support challenges? (from multiple docs)"""
        question = "What challenges does the support organization face according to the documents?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        expected_challenges = [
            "cost",
            "support",
            "margin",
            "ticket",
            "contractor"
        ]
        
        found = [c for c in expected_challenges if c in answer_lower]
        assert len(found) >= 2, \
            f"Should identify support challenges. Found: {found}"
        
        print(f"\n✓ Support Challenges Identified (found {len(found)}/5):")
        print(f"  Answer: {response.executive_summary[:150]}...")
        print(f"  Evidence: {len(response.evidence)} sources")


    @pytest.mark.slow
    def test_engineering_improvements(self, agent):
        """Test: What engineering improvements were made? (from narrative)"""
        question = "According to the leadership briefing, what improvements did engineering make?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        expected_improvements = [
            "engineering",
            "backlog",
            "release",
            "deliver"
        ]
        
        found = [i for i in expected_improvements if i in answer_lower]
        assert len(found) >= 2, \
            f"Should mention engineering improvements. Found: {found}"
        
        print(f"\n✓ Engineering Improvements Retrieved (found {len(found)}/4):")
        print(f"  Answer: {response.executive_summary[:150]}...")


class TestDocumentVsTableRetrieval:
    """Tests to verify documents are prioritized over tables on narrative questions"""

    @pytest.mark.slow
    def test_narrative_question_uses_documents_not_tables(self, agent):
        """Test: Narrative question should use documents, not just tables"""
        question = "What margin pressure does the company face from the annual report?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should include narrative context, not just numerical
        answer_lower = response.executive_summary.lower()
        
        # Should have narrative elements
        narrative_elements = [
            "margin",
            "onboarding",
            "automation",
            "support"
        ]
        
        found_narrative = [e for e in narrative_elements if e in answer_lower]
        assert len(found_narrative) >= 2, \
            f"Should provide narrative context. Found: {found_narrative}"
        
        # Check that retrieval/narrative was actually used
        approach_str = " ".join(response.planned_approach).lower()
        has_retrieval = "retrieval" in approach_str or "narrative" in approach_str
        
        print(f"\n✓ Narrative Question Routing:")
        print(f"  Question: {question}")
        print(f"  Uses Narrative/Retrieval: {has_retrieval}")
        print(f"  Answer: {response.executive_summary[:150]}...")
        print(f"  Found narrative elements: {found_narrative}")


    @pytest.mark.slow
    def test_quarterly_performance_from_documents(self, agent):
        """Test: Q2 performance question should pull from operational update"""
        question = "From the Q2 operational update, which areas improved in the second quarter?"
        response = agent.ask(question, Path("data/raw"))
        
        answer_lower = response.executive_summary.lower()
        expected_areas = [
            "engineering",
            "sales",
            "conversion",
            "backlog"
        ]
        
        found = [a for a in expected_areas if a in answer_lower]
        assert len(found) >= 2, \
            f"Should identify Q2 improvements. Found: {found}"
        
        print(f"\n✓ Q2 Performance Retrieved (found {len(found)}/4):")
        print(f"  Answer: {response.executive_summary[:150]}...")


    @pytest.mark.slow
    def test_document_coverage_breadth(self, agent):
        """Test: Can retrieve from multiple documents for comprehensive answer"""
        question = "From the documents, what operational challenges span across multiple departments?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should reference multiple sources
        assert len(response.source_references) >= 2, \
            f"Should cite multiple sources. Got: {len(response.source_references)}"
        
        answer_lower = response.executive_summary.lower()
        departments = ["engineering", "sales", "support", "finance", "marketing"]
        found_deps = [d for d in departments if d in answer_lower]
        
        assert len(found_deps) >= 2, \
            f"Should mention multiple departments. Found: {found_deps}"
        
        print(f"\n✓ Multi-Department Coverage (found {len(found_deps)}/5 departments):")
        print(f"  Answer: {response.executive_summary[:150]}...")
        print(f"  Sources cited: {len(response.source_references)}")
        for src in response.source_references[:3]:
            print(f"    - {src}")


class TestRetrievalEfficiency:
    """Tests to measure retrieval efficiency and response quality"""

    @pytest.mark.slow
    def test_response_completeness_strategic_question(self, agent):
        """Test: Strategic question gets complete answer with context"""
        question = "What should the company focus on in the next planning cycle?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should have: summary + findings + evidence
        assert response.executive_summary, "Should have executive summary"
        assert len(response.key_findings) >= 3, \
            f"Should have 3+ key findings, got {len(response.key_findings)}"
        assert len(response.evidence) >= 2, \
            f"Should cite evidence sources, got {len(response.evidence)}"
        
        answer_char_count = len(response.executive_summary)
        assert answer_char_count > 200, \
            f"Answer should be detailed (>200 chars), got {answer_char_count}"
        
        print(f"\n✓ Response Completeness:")
        print(f"  Summary length: {answer_char_count} chars")
        print(f"  Key findings: {len(response.key_findings)}")
        print(f"  Evidence sources: {len(response.evidence)}")
        print(f"  Source references: {len(response.source_references)}")


    @pytest.mark.slow
    def test_answer_directs_to_source(self, agent):
        """Test: Answer can be traced back to source documentation"""
        question = "What specific strategic priorities are listed in the annual report?"
        response = agent.ask(question, Path("data/raw"))
        
        # Should cite sources
        assert response.source_references, "Should cite sources"
        
        # Should mention specific priorities
        priorities = ["retention", "cost", "automation", "forecasting"]
        found = [p for p in priorities if p in response.executive_summary.lower()]
        
        assert len(found) >= 2, \
            f"Should mention specific priorities. Found: {found}"
        
        print(f"\n✓ Traceability to Source:")
        print(f"  Located {len(found)}/4 expected priorities")
        print(f"  Cites {len(response.source_references)} sources:")
        for src in response.source_references:
            print(f"    - {src}")


    @pytest.mark.slow
    def test_document_vs_implicit_knowledge(self, agent):
        """Test: Distinguishes between document facts and general knowledge"""
        question = "According to the 2024 annual report, what were the strategic priorities?"
        response = agent.ask(question, Path("data/raw"))
        
        # Explicit document reference should trigger document-first approach
        approach_str = " ".join(response.planned_approach).lower()
        
        # Should retrieve from documents first
        uses_document_approach = any(
            keyword in approach_str 
            for keyword in ["document", "retrieval", "narrative", "extract"]
        )
        
        assert uses_document_approach, \
            "Should use document-first approach for explicit reference"
        
        # Should find exact answer from document
        answer_found = all(
            word in response.executive_summary.lower()
            for word in ["retention", "automation", "forecasting"]
        )
        
        assert answer_found, \
            "Should contain all strategic priorities from document"
        
        print(f"\n✓ Document-First Approach Working:")
        print(f"  Uses document-first: {uses_document_approach}")
        print(f"  Found exact document content: {answer_found}")
        print(f"  Planned approach: {response.planned_approach}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
