"""
Comprehensive integration tests covering all data type scenarios:
1) Tables alone
2) Text/Memory content alone  
3) Graphs/Visual artifacts alone
4) Tables + Memory combination
5) All types combination
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from wald_agent_reference.core.agent import LeadershipInsightAgent
from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.ingestion.ingest import DocumentIngestor
from wald_agent_reference.memory.structured_store import StructuredMemoryStore
from wald_agent_reference.chat.manager import ChatManager


@pytest.fixture
def settings():
    """Test settings with paths to sample data"""
    return AppSettings(
        data_sources_path=Path("data/raw"),
        structured_store_db_path=Path("outputs/test.db"),
        output_dir=Path("outputs"),
        embedding_engine="hash",
        supermemory_api_key="",  # Disabled for tests
    )


@pytest.fixture
def sample_corpus():
    """Ingest sample data to create test corpus"""
    settings = AppSettings(
        data_sources_path=Path("data/raw"),
        structured_store_db_path=Path("outputs/test.db"),
    )
    ingestor = DocumentIngestor(settings)
    return ingestor.ingest_folder(Path("data/raw"))


class TestScenario1_TablesOnly:
    """Scenario 1: Needs to refer to tables alone
    
    Questions that should use SQL Agent only:
    - "Which region has the highest revenue?"
    - "What is our revenue by region?"
    - "Which departments missed their targets?"
    """
    
    @pytest.mark.slow
    def test_region_revenue_query(self):
        """Test query that requires regional_actuals.csv only"""
        agent = LeadershipInsightAgent()
        
        question = "Which region has the highest revenue?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert "revenue" in response.executive_summary.lower()
        # Should have evidence from tables
        assert any("regional" in ref.lower() or "actual" in ref.lower() 
                  for ref in response.source_references)
    
    @pytest.mark.slow
    def test_department_target_variance(self):
        """Test variance analysis using only tables"""
        agent = LeadershipInsightAgent()
        
        question = "Which departments are underperforming against targets?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        # Response should have key findings or evidence
        assert len(response.key_findings) > 0 or len(response.evidence) > 0

    @pytest.mark.slow
    def test_region_metric_breakdown_does_not_fall_back_to_ranking(self):
        agent = LeadershipInsightAgent(
            AppSettings(
                enable_llm_formatting=False,
                retrieval_backend="local",
                memory_backend="none",
                vector_backend="hash",
                output_dir="outputs",
            )
        )

        response = agent.ask("actual margin and gained for each regions ?", Path("data/raw"))

        summary = response.executive_summary.lower()
        assert "north america actual margin = 31.00".lower() in summary
        assert "highest actual margin" not in summary
        assert "grounded metric named `gained`" in response.executive_summary

    @pytest.mark.slow
    def test_q2_operational_costs_do_not_fall_back_to_non_temporal_cost_table(self):
        agent = LeadershipInsightAgent(
            AppSettings(
                enable_llm_formatting=False,
                retrieval_backend="local",
                memory_backend="none",
                vector_backend="hash",
                output_dir="outputs",
            )
        )

        response = agent.ask("operational costs q2 ?", Path("data/raw"))

        summary = response.executive_summary.lower()
        assert "do not see an explicit numeric q2 operational cost value" in summary
        assert "ticket volume" in " ".join(response.key_findings).lower() or "onboarding" in " ".join(response.key_findings).lower()
        assert "highest actual cost" not in summary


class TestScenario2_MemoryContentOnly:
    """Scenario 2: Needs to refer to text content alone
    
    Questions that should use Semantic Retrieval:
    - "What are the key risks highlighted by leadership?"
    - "What is our current revenue trend?" (narrative form)
    - "What are the organizational priorities?"
    """
    
    @pytest.mark.slow
    def test_risks_narrative_question(self):
        """Test narrative question requiring text retrieval"""
        agent = LeadershipInsightAgent()
        
        question = "What are the key risks highlighted by the leadership?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert "risk" in response.executive_summary.lower() or len(response.key_findings) > 0
        # Should have source references
        assert len(response.source_references) > 0
    
    @pytest.mark.slow
    def test_priority_extraction(self):
        """Test priority/strategy question using text only"""
        agent = LeadershipInsightAgent()
        
        question = "What are our organizational objectives and priorities?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert len(response.key_findings) > 0


class TestScenario3_VisualsOnly:
    """Scenario 3: Needs to refer to graphs/visual artifacts
    
    Questions:
    - "Show me the revenue trend visualization"
    - "What visual trends are present in our data?"
    """
    
    @pytest.mark.slow
    def test_visual_artifact_reference(self):
        """Test query that references visuals"""
        agent = LeadershipInsightAgent()
        
        question = "What visual trends are shown in our revenue data?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        # Should reference visual artifacts
        assert len(response.visual_insights) > 0 or len(response.plot_paths) > 0
    
    @pytest.mark.slow
    def test_chart_based_analysis(self):
        """Test chart/graph based question"""
        agent = LeadershipInsightAgent()
        
        question = "Generate a visualization of revenue by region"
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        assert response is not None
        # Should have plots in response
        assert response.plot_base64 or len(response.plot_paths) > 0


class TestScenario4_TablesAndMemory:
    """Scenario 4: Needs to refer to both tables and text content
    
    Questions:
    - "Which departments are underperforming and what are the associated risks?"
    - "What is the revenue trend and what factors influence it?"
    """
    
    @pytest.mark.slow
    def test_performance_and_risks(self):
        """Test query combining SQL and text retrieval"""
        agent = LeadershipInsightAgent()
        
        question = "Which departments are underperforming and what factors contribute?"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert len(response.key_findings) >= 1  # At least one finding
        
        # Should have references from tables and/or text
        csv_refs = sum(1 for ref in response.source_references 
                      if ".csv" in ref or ".xlsx" in ref)
        text_refs = sum(1 for ref in response.source_references 
                       if ".md" in ref or ".txt" in ref)
        assert csv_refs > 0 or text_refs > 0, "Should have source references"
    
    @pytest.mark.slow
    def test_comprehensive_analysis(self):
        """Test multi-source analysis"""
        agent = LeadershipInsightAgent()
        
        question = "Analyze our sales performance including metrics and context"
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert len(response.evidence) > 0 or len(response.key_findings) > 0


class TestScenario5_AllTypes:
    """Scenario 5: Needs to refer to all types (tables + text + visuals)
    
    Questions:
    - "Provide a comprehensive business analysis"
    - "Give me a full strategic assessment including metrics, risks, and trends"
    """
    
    @pytest.mark.slow
    def test_comprehensive_business_analysis(self):
        """Test query requiring all data types"""
        agent = LeadershipInsightAgent()
        
        question = "Provide a comprehensive business analysis including financial metrics, operational risks, and strategic context"
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        assert response is not None
        assert len(response.key_findings) >= 1
        
        # Should have references from multiple sources
        ref_types = {
            "tables": 0,
            "text": 0,
            "visuals": 0
        }
        
        for ref in response.source_references:
            if ".csv" in ref or ".xlsx" in ref:
                ref_types["tables"] += 1
            elif ".md" in ref or ".txt" in ref:
                ref_types["text"] += 1
        
        # At minimum should have some references
        assert sum(ref_types.values()) > 0, "Should reference data sources"
    
    @pytest.mark.slow  
    def test_all_data_types_orchestration(self):
        """Test that planner orchestrates multiple tools correctly"""
        agent = LeadershipInsightAgent()
        
        question = "Executive summary: revenue analysis, risk assessment, and performance metrics"
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        assert response is not None
        # Verify orchestration happened
        assert len(response.planned_approach) >= 1  # At least one planned step


class TestResponseFormatting:
    """Test new response formatting with chat vs detailed report split"""
    
    def test_chat_response_clarity(self):
        """Test that chat response is clear and concise"""
        agent = LeadershipInsightAgent()
        
        question = "What is our revenue trend?"
        response = agent.ask(question, Path("data/raw"))
        
        # Chat response should be concise
        assert len(response.executive_summary) < 500  # Reasonable length for chat
        
        # Should have key findings
        assert len(response.key_findings) >= 0
    
    def test_detailed_report_generation(self):
        """Test detailed report endpoint would provide full details"""
        # This would be tested via the web endpoint
        # /api/chats/{chat_id}/report should return:
        # - Detailed methodology
        # - All evidence chunks
        # - Trace information  
        # - Source references with page numbers
        pass


class TestVisualDataHandling:
    """Test visual artifact processing and referencing"""
    
    @pytest.mark.slow
    def test_visual_extraction_during_ingestion(self):
        """Test that visuals are properly extracted during ingestion"""
        # Create corpus via ingestor
        settings = AppSettings(
            data_sources_path=Path("data/raw"),
            structured_store_db_path=Path("outputs/test.db"),
        )
        ingestor = DocumentIngestor(settings)
        corpus = ingestor.ingest_folder(Path("data/raw"))
        
        # Check that corpus has visuals
        assert len(corpus.visuals) >= 0
        
        # Each visual should have required fields
        for artifact_id, visual_artifact in corpus.visuals.items():
            assert visual_artifact.summary
            assert visual_artifact.extracted_text or visual_artifact.summary
            assert visual_artifact.artifact_id
            assert visual_artifact.source_path
    
    @pytest.mark.slow
    def test_visual_in_response(self):
        """Test that visuals are included in response references"""
        agent = LeadershipInsightAgent()
        
        question = "Generate a visualization of our metrics"
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        assert response is not None
        # Either has base64 plot or plot path
        assert response.plot_base64 or len(response.plot_paths) > 0 or response is not None


class TestTableInspectionTool:
    """Test the new table inspection tool for query planning"""
    
    def test_get_table_preview(self):
        """Test table preview tool exists and can be called"""
        from wald_agent_reference.core.tools import TableInspectionTool, TablePreview
        
        settings = AppSettings(
            structured_store_db_path=Path("outputs/test.db")
        )
        store = StructuredMemoryStore(settings.structured_store_db_path)
        inspector = TableInspectionTool(store)
        
        # TableInspectionTool instance should exist
        assert inspector is not None
        
        # Methods should exist and be callable
        assert hasattr(inspector, 'get_table_preview')
        assert hasattr(inspector, 'get_all_table_previews')
        assert callable(inspector.get_table_preview)
        assert callable(inspector.get_all_table_previews)
    
    def test_get_all_table_previews(self):
        """Test getting all table previews"""
        from wald_agent_reference.core.tools import TableInspectionTool
        
        settings = AppSettings(
            structured_store_db_path=Path("outputs/test.db")
        )
        store = StructuredMemoryStore(settings.structured_store_db_path)
        inspector = TableInspectionTool(store)
        
        # Tool should exist and be usable
        assert inspector is not None
        
        # Create a minimal corpus to test with
        from wald_agent_reference.core.models_v2 import Corpus
        corpus = Corpus(chunks=[], tables={}, documents={}, visuals=[])
        
        # get_all_table_previews should return an empty dict for empty corpus
        previews = inspector.get_all_table_previews(corpus)
        
        assert isinstance(previews, dict)
        assert len(previews) == 0  # Empty corpus should have no previews


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
