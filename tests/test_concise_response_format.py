"""Test that API returns concise chat responses with key findings + plots only"""

import pytest
from pathlib import Path
from wald_decision_agent.core.agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings


@pytest.fixture
def agent():
    """Create an agent for testing"""
    settings = AppSettings(
        data_sources_path=str(Path("data/raw")),
        structured_store_db_path=str(Path("outputs/test_concise.db")),
        output_dir=str(Path("outputs")),
        embedding_engine="hash",
        enable_llm_formatting=False,
    )
    return LeadershipInsightAgent(settings)


@pytest.mark.slow
def test_concise_chat_response_format(agent):
    """Verify response can be converted to concise chat format"""
    question = "Which regions underperformed against targets?"
    response = agent.ask(question, Path("data/raw"), generate_plot=True)
    
    # Convert to chat response
    chat_response = response.to_chat_response()
    
    print(f"\n✓ Concise Chat Response Structure:")
    print(f"  Question: {chat_response.question}")
    print(f"  Answer: {chat_response.answer[:100]}...")
    print(f"  Key Findings: {len(chat_response.key_findings)} items")
    print(f"  Visual Insights: {len(chat_response.visual_insights)} items")
    print(f"  Plots (base64): {len(chat_response.plots_base64)} images")
    print(f"  Source Summary: {chat_response.source_summary}")
    print(f"  Data Types: {chat_response.data_types_used}")
    
    # Verify fields are populated
    assert chat_response.question
    assert chat_response.answer
    assert len(chat_response.key_findings) > 0
    assert chat_response.source_summary
    assert chat_response.data_types_used
    
    # Verify it can be converted to dict
    dict_form = chat_response.to_dict()
    assert "question" in dict_form
    assert "answer" in dict_form
    assert "key_findings" in dict_form
    
    print(f"\n✓ Chat response successfully converted to concise format!")
    print(f"✓ Dict form keys: {list(dict_form.keys())}")
    

@pytest.mark.slow
def test_full_report_available_separately(agent):
    """Verify full report details are available even when chat is concise"""
    question = "Which departments are underperforming and what factors contribute?"
    response = agent.ask(question, Path("data/raw"))
    
    # Full details still available on AgentResponse
    print(f"\n✓ Full Report Details Available:")
    print(f"  Planned Approach: {len(response.planned_approach)} steps")
    print(f"  Calculations: {len(response.calculations)} items")
    print(f"  Evidence: {len(response.evidence)} references")
    print(f"  Caveats: {len(response.caveats)} items")
    print(f"  Source References: {len(response.source_references)} sources")
    
    # Full markdown/report available
    markdown = response.to_markdown()
    assert "Planned Approach" in markdown
    assert "Calculations" in markdown
    assert "Evidence" in markdown
    assert "Risks / Caveats" in markdown
    
    print(f"\n✓ Full markdown report ({len(markdown)} chars) available for PDF export")
    
    # Chat response is still concise
    chat_response = response.to_chat_response()
    assert "Planned Approach" not in str(chat_response.to_dict())  # Not in chat
    
    print(f"✓ Chat response remains concise (excludes methodology details)")
