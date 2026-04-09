"""
Test to inspect what chunks are actually being retrieved
"""

import pytest
from pathlib import Path
from wald_decision_agent.core.agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.retrieval.retrieve import HybridRetriever


@pytest.fixture
def agent():
    settings = AppSettings()
    return LeadershipInsightAgent(settings)


class TestRetrievalInspection:
    """Inspect what's being retrieved from documents"""

    def test_what_chunks_returned_for_engineering_question(self, agent):
        """See exactly what chunks are being retrieved for engineering question"""
        question = "What improvements did engineering make?"
        
        # Simulate ingestion
        from wald_decision_agent.ingestion.ingest import ingest_folder
        corpus = ingest_folder(Path("data/raw"))
        
        # Create retriever
        retriever = HybridRetriever(corpus)
        
        # Retrieve chunks
        results = retriever.retrieve(question, max_results=5)
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nRetrieved {len(results)} chunks:")
        for i, (chunk, score) in enumerate(results):
            print(f"\n{i+1}. Score: {score:.3f}")
            print(f"   Source: {chunk.source_path.name}")
            print(f"   Content: {chunk.content[:150]}...")
        
        # Check if engineering improvements are in retrieved chunks
        all_content = " ".join([chunk.content for chunk, _ in results])
        has_engineering = any(
            term in all_content.lower() 
            for term in ["engineering", "release", "backlog", "delivery", "hygiene"]
        )
        
        print(f"\nContains engineering-related terms: {has_engineering}")
        
        assert True  # Just for inspection


    def test_what_chunks_for_risks_question(self, agent):
        """See what chunks are retrieved for risks question"""
        question = "What are the largest risks?"
        
        from wald_decision_agent.ingestion.ingest import ingest_folder
        corpus = ingest_folder(Path("data/raw"))
        
        retriever = HybridRetriever(corpus)
        results = retriever.retrieve(question, max_results=5)
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        print(f"\nRetrieved {len(results)} chunks:")
        for i, (chunk, score) in enumerate(results):
            print(f"\n{i+1}. Score: {score:.3f}")
            print(f"   Source: {chunk.source_path.name}")
            print(f"   Content: {chunk.content[:150]}...")
        
        all_content = " ".join([chunk.content for chunk, _ in results])
        has_risk_terms = any(
            term in all_content.lower()
            for term in ["risk", "slower", "bottleneck", "uneven", "execution"]
        )
        
        print(f"\nContains risk-related terms: {has_risk_terms}")
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
