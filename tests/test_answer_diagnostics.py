"""
Diagnostic test to see what actual responses are being returned
"""

import pytest
from pathlib import Path
from wald_agent_reference.core.agent import LeadershipInsightAgent
from wald_agent_reference.core.config import AppSettings


@pytest.fixture
def agent():
    settings = AppSettings()
    return LeadershipInsightAgent(settings)


class TestAnswerDiagnostics:
    """Diagnose what answers are actually being returned"""

    @pytest.mark.slow
    def test_strategic_priorities_actual_response(self, agent):
        """See what we actually get for strategic priorities question"""
        question = "What are the strategic priorities for the next planning cycle?"
        response = agent.ask(question, Path("data/raw"))
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nPlanned Approach: {response.planned_approach}")
        print(f"\nExecutive Summary ({len(response.executive_summary)} chars):")
        print(f"  {response.executive_summary}")
        print(f"\nKey Findings ({len(response.key_findings)} items):")
        for i, finding in enumerate(response.key_findings):
            print(f"  {i+1}. {finding}")
        print(f"\nEvidence ({len(response.evidence)} items):")
        for evidence in response.evidence:
            print(f"  - {evidence}")
        print(f"\nSource References ({len(response.source_references)} items):")
        for src in response.source_references:
            print(f"  - {src}")
        print(f"\nCalculations: {response.calculations}")
        print(f"Caveats: {response.caveats}")
        
        # Don't assert - just show what we got
        assert True


    @pytest.mark.slow
    def test_risks_actual_response(self, agent):
        """See what we actually get for risks question"""
        question = "What are the largest risks highlighted by leadership from the annual report?"
        response = agent.ask(question, Path("data/raw"))
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nPlanned Approach: {response.planned_approach}")
        print(f"\nExecutive Summary ({len(response.executive_summary)} chars):")
        print(f"  {response.executive_summary}")
        print(f"\nKey Findings ({len(response.key_findings)} items):")
        for i, finding in enumerate(response.key_findings):
            print(f"  {i+1}. {finding}")
        
        assert True


    @pytest.mark.slow
    def test_support_challenges_actual_response(self, agent):
        """See what we actually get for support organization question"""
        question = "What challenges does the support organization face according to the documents?"
        response = agent.ask(question, Path("data/raw"))
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nPlanned Approach: {response.planned_approach}")
        print(f"\nExecutive Summary ({len(response.executive_summary)} chars):")
        print(f"  {response.executive_summary}")
        print(f"\nKey Findings ({len(response.key_findings)} items):")
        for i, finding in enumerate(response.key_findings):
            print(f"  {i+1}. {finding}")
        
        assert True


    @pytest.mark.slow
    def test_engineering_improvements_actual_response(self, agent):
        """See what we actually get for engineering question"""
        question = "According to the leadership briefing, what improvements did engineering make?"
        response = agent.ask(question, Path("data/raw"))
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nPlanned Approach: {response.planned_approach}")
        print(f"\nExecutive Summary ({len(response.executive_summary)} chars):")
        print(f"  {response.executive_summary}")
        print(f"\nKey Findings ({len(response.key_findings)} items):")
        for i, finding in enumerate(response.key_findings):
            print(f"  {i+1}. {finding}")
        
        assert True
