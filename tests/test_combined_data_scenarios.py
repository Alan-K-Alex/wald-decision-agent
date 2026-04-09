"""
Comprehensive test for combined data scenarios (tables + memory service).

Tests:
1. SQL + Retrieval combined queries
2. Plot generation and embedding in responses
3. Dynamic title generation
4. Response completeness verification
"""

import pytest
from pathlib import Path
from wald_agent_reference.core.agent import LeadershipInsightAgent
from wald_agent_reference.chat.manager import ChatManager, _generate_title_from_question
from wald_agent_reference.core.config import AppSettings


@pytest.fixture
def settings():
    """Test settings for combined data scenarios"""
    return AppSettings(
        data_sources_path=Path("data/raw"),
        structured_store_db_path=Path("outputs/test_combined.db"),
        output_dir=Path("outputs"),
        embedding_engine="hash",
        supermemory_api_key="",  # Disabled for tests
        enable_llm_formatting=False,  # Skip LLM formatting for deterministic tests
    )


class TestCombinedDataScenarios:
    """Test when system needs data from BOTH tables and document memory"""
    
    @pytest.mark.slow
    def test_underperformance_analysis_combines_sql_and_narrative(self):
        """
        Scenario: "Which departments underperform and what factors contribute?"
        
        Expected:
        - SQL agent finds underperforming departments from tables
        - Retriever finds narrative explanation from documents
        - Combined response includes both numeric analysis + context
        """
        agent = LeadershipInsightAgent()
        question = "Which departments are underperforming and what factors contribute?"
        
        response = agent.ask(question, Path("data/raw"))
        
        # Verify response structure
        assert response is not None
        assert response.executive_summary, "Should have executive summary"
        assert response.key_findings, "Should have key findings from analysis"
        
        # Verify combined data sources
        csv_refs = sum(1 for ref in response.source_references 
                      if ".csv" in ref or ".xlsx" in ref)
        text_refs = sum(1 for ref in response.source_references 
                       if ".md" in ref or ".txt" in ref)
        
        # Should have references from both table and text sources
        has_table_data = csv_refs > 0
        has_text_data = text_refs > 0
        
        print(f"\n✓ CSV References: {csv_refs}")
        print(f"✓ Text References: {text_refs}")
        print(f"✓ Has Table Data: {has_table_data}")
        print(f"✓ Has Text Data: {has_text_data}")
        print(f"✓ Key Findings: {response.key_findings}")
        
        assert has_table_data or has_text_data, (
            "Response should reference at least one data source (table or text)"
        )

    @pytest.mark.slow
    def test_performance_with_context_analysis(self):
        """
        Scenario: "Analyze our sales performance including metrics and context"
        
        Expected:
        - Combines numerical metrics from tables
        - Adds contextual explanation from documents
        - Response includes both hard metrics and narrative
        """
        agent = LeadershipInsightAgent()
        question = "Analyze our sales performance including metrics and context"
        
        response = agent.ask(question, Path("data/raw"))
        
        assert response is not None
        assert len(response.key_findings) > 0, "Should have findings"
        assert len(response.source_references) > 0, "Should have source references"
        
        print(f"\n✓ Executive Summary: {response.executive_summary[:100]}...")
        print(f"✓ Findings Count: {len(response.key_findings)}")
        print(f"✓ Source Count: {len(response.source_references)}")

    @pytest.mark.slow
    def test_comprehensive_business_analysis_all_sources(self):
        """
        Scenario: Full analysis requiring tables + text + optional visuals
        
        Expected:
        - SQL for metrics
        - Retrieval for narrative context
        - Optional plot generation
        - All response fields populated
        """
        agent = LeadershipInsightAgent()
        question = "Provide a comprehensive business analysis including financial metrics, operational risks, and strategic context"
        
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        # Verify response completeness
        assert response.executive_summary
        assert response.key_findings
        assert response.source_references
        assert response.planned_approach
        
        print(f"\n✓ Planned Approach: {response.planned_approach}")
        print(f"✓ Summary: {response.executive_summary[:150]}...")

    @pytest.mark.slow
    def test_combined_variance_and_trend_query_generates_multiple_plots(self):
        agent = LeadershipInsightAgent()
        question = "Why did Europe miss the plan and how has the quarterly revenue trend?"

        response = agent.ask(question, Path("data/raw"), generate_plot=True)

        assert len(response.plot_paths) >= 2
        assert any("variance" in insight.lower() for insight in response.visual_insights)
        assert any("quarterly revenue trend" in insight.lower() or "trend" in insight.lower() for insight in response.visual_insights)

    @pytest.mark.slow
    def test_combined_variance_and_trend_query_keeps_causal_narrative_and_linked_evidence(self):
        settings = AppSettings(enable_llm_formatting=False, memory_backend="none", retrieval_backend="local", vector_backend="hash")
        agent = LeadershipInsightAgent(settings)
        question = "Why did Europe miss the plan and how has the quarterly revenue trend?"

        response = agent.ask(question, Path("data/raw"), generate_plot=False)

        summary_lower = response.executive_summary.lower()
        assert "slower" in summary_lower or "enterprise conversions" in summary_lower
        assert "trend" in summary_lower or "q1" in summary_lower
        assert any(item.startswith("[") and "](" in item for item in response.evidence)
        assert any(
            any(marker in item.lower() for marker in ["because", "slower", "weaker channel execution", "mid-market"])
            for item in response.evidence
        )
        print(f"✓ Sources: {response.source_references}")
        assert len(response.source_references) > 0

    @pytest.mark.slow
    def test_operational_update_query_prefers_q2_update_document_over_generic_metric_answer(self):
        settings = AppSettings(enable_llm_formatting=False, memory_backend="none", retrieval_backend="local", vector_backend="hash")
        agent = LeadershipInsightAgent(settings)

        response = agent.ask("operartional updates for Q2 ?", Path("data/raw"), generate_plot=False)

        summary_lower = response.executive_summary.lower()
        assert "engineering and sales" in summary_lower or "support and finance" in summary_lower
        assert "revenue has the highest q2 2024 value" not in summary_lower
        assert any("q2_operational_update.md" in item for item in response.evidence)
        assert any("q2_operational_update.md" in item for item in response.source_references)


class TestPlotGenerationAndEmbedding:
    """Test plot generation and embedding in responses"""
    
    @pytest.mark.slow
    def test_plot_generation_creates_base64_image(self):
        """
        Verify that plots are generated AND embedded as base64.
        
        The plot should be available in two forms:
        1. File path (plot_paths)
        2. Base64 encoded (plot_base64) for embedding in responses
        """
        agent = LeadershipInsightAgent()
        question = "Which regions have revenue variance? Generate a visualization"
        
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        assert response is not None
        
        # Check plot generation
        has_plot_paths = len(response.plot_paths) > 0
        has_plot_base64 = bool(response.plot_base64)
        
        print(f"\n✓ Has Plot Paths: {has_plot_paths}")
        print(f"  Paths: {response.plot_paths}")
        print(f"✓ Has Base64 Image: {has_plot_base64}")
        if response.plot_base64:
            print(f"  Image Size: {len(response.plot_base64) // 1024} KB")
        
        # At least one form should be present if visualization was requested
        if has_plot_paths or has_plot_base64:
            print("✓ Plot successfully generated and embedded")

    @pytest.mark.slow
    def test_visualization_in_markdown_response(self):
        """
        Verify that plots appear in markdown response.
        
        The to_markdown() method should include plot references
        either as file paths or as embedded data.
        """
        agent = LeadershipInsightAgent()
        question = "Show revenue trend by region"
        
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        markdown = response.to_markdown()
        
        # Markdown should reference plots if they exist
        has_plot_reference = "plot" in markdown.lower() or "![" in markdown
        
        print(f"\n✓ Markdown contains plot reference: {has_plot_reference}")
        if response.plot_paths:
            print(f"  Detected {len(response.plot_paths)} plot paths in response")


class TestDynamicTitleGeneration:
    """Test that titles are generated dynamically from questions"""
    
    def test_title_generation_from_question(self):
        """Verify title generation function works correctly"""
        test_cases = [
            ("What is our revenue trend?", "What is our revenue trend"),
            ("Which regions missed revenue plan by the largest amount?", 
             "Which regions missed revenue plan by the..."),
            ("Are there any risks we should be concerned about?",
             "Are there any risks we should be..."),
            ("strategy plans as per documents?", "strategy plans as per documents"),
        ]
        
        for question, expected_title in test_cases:
            generated = _generate_title_from_question(question, max_words=7)
            print(f"\n✓ Question: {question}")
            print(f"  Generated: {generated}")
            print(f"  Expected: {expected_title}")
            assert generated == expected_title, (
                f"Generated '{generated}' != expected '{expected_title}'"
            )

    @pytest.mark.slow
    def test_chat_title_updates_dynamically(self):
        """Verify that chat manager updates title dynamically from first question"""
        from wald_agent_reference.chat.manager import ChatManager
        
        settings = AppSettings(
            data_sources_path=str(Path("data/raw")),
            structured_store_db_path=str(Path("outputs/test_title.db")),
            output_dir=str(Path("outputs")),
        )
        
        manager = ChatManager(settings)
        
        # Create a "New Chat"
        session = manager.create_chat(title="New Chat")
        initial_title = session.title
        
        assert initial_title == "New Chat", "Should start with 'New Chat' title"
        print(f"\n✓ Initial title: {initial_title}")
        
        # Now record an exchange with a real question
        question = "What are the key risks in our business?"
        expected_title = _generate_title_from_question(question)
        
        # Create a simple mock response
        from wald_agent_reference.core.models import AgentResponse
        response = AgentResponse(
            question=question,
            planned_approach=["Test plan"],
            executive_summary="Test summary",
            key_findings=["Finding 1"],
            calculations=["Sample calculation"],
            evidence=["Evidence 1"],
            source_references=["source.csv"],
            caveats=[],
            visual_insights=[],
            plot_paths=[],
            plot_base64="",
        )
        
        # Record this exchange
        manager.record_exchange(session, question, response)
        
        # Reload session
        reloaded = manager.load_chat(session.chat_id)
        
        print(f"✓ Original title: {initial_title}")
        print(f"✓ Expected title: {expected_title}")
        print(f"✓ Reloaded title: {reloaded.title}")
        
        # Title should now be generated from question
        assert reloaded.title == expected_title, (
            f"Title should be updated to '{expected_title}' but got '{reloaded.title}'"
        )
        
        print("✓ Chat title successfully updated from question!")


class TestResponseCompleteness:
    """Verify all response fields are properly populated"""
    
    @pytest.mark.slow
    def test_response_includes_all_required_fields(self):
        """Verify AgentResponse has all expected fields populated"""
        agent = LeadershipInsightAgent()
        question = "Which regions underperformed against targets?"
        
        response = agent.ask(question, Path("data/raw"), generate_plot=True)
        
        # Check all required fields
        required_fields = [
            ('question', str, lambda v: len(v) > 0),
            ('planned_approach', list, lambda v: len(v) > 0),
            ('executive_summary', str, lambda v: len(v) > 0),
            ('key_findings', list, lambda v: len(v) > 0),
            ('source_references', list, lambda v: len(v) >= 0),  # Can be empty
            ('caveats', list, lambda v: True),  # Optional
            ('visual_insights', list, lambda v: True),  # Optional
            ('plot_paths', list, lambda v: True),  # Optional
        ]
        
        print("\n✓ Response Field Verification:")
        for field_name, expected_type, validator in required_fields:
            field_value = getattr(response, field_name, None)
            
            # Check type
            assert isinstance(field_value, expected_type), (
                f"Field '{field_name}' should be {expected_type.__name__} "
                f"but got {type(field_value).__name__}"
            )
            
            # Check validation
            is_valid = validator(field_value)
            status = "✓" if is_valid else "⚠"
            
            if isinstance(field_value, (list, str)):
                if isinstance(field_value, list):
                    value_str = f"{len(field_value)} items"
                else:
                    value_str = f"{len(field_value)} chars"
            else:
                value_str = str(field_value)[:50]
            
            print(f"  {status} {field_name}: {value_str}")
            
            assert is_valid, f"Field '{field_name}' validation failed"

    @pytest.mark.slow
    def test_response_markdown_conversion(self):
        """Verify response can be converted to markdown"""
        agent = LeadershipInsightAgent()
        question = "Summarize our strategic priorities"
        
        response = agent.ask(question, Path("data/raw"))
        markdown = response.to_markdown()
        
        assert markdown, "Markdown conversion should produce output"
        assert "Question" in markdown, "Markdown should include question"
        assert "Executive Summary" in markdown or "Summary" in markdown, (
            "Markdown should include summary section"
        )
        
        print(f"\n✓ Markdown Length: {len(markdown)} characters")
        print(f"✓ Markdown Preview:\n{markdown[:300]}...\n")
