"""Test that chat title is dynamically generated and returned in API responses"""

import pytest
from pathlib import Path
import json
from wald_agent_reference.chat.manager import ChatManager, _generate_title_from_question
from wald_agent_reference.core.config import AppSettings


@pytest.fixture
def chat_manager_fixture():
    """Create a chat manager for testing"""
    settings = AppSettings(
        data_sources_path=str(Path("data/raw")),
        structured_store_db_path=str(Path("outputs/test_title_api.db")),
        output_dir=str(Path("outputs")),
        chats_path=str(Path("outputs/chats_test")),
    )
    manager = ChatManager(settings)
    yield manager


def test_title_updated_in_chat_metadata(chat_manager_fixture):
    """Verify title is saved in chat.json metadata file"""
    manager = chat_manager_fixture
    
    # Create a chat
    session = manager.create_chat(title="New Chat")
    
    # Verify initial title
    metadata_path = session.metadata_path
    initial_data = json.loads(metadata_path.read_text())
    assert initial_data["title"] == "New Chat"
    
    print(f"\n✓ Initial title in metadata: {initial_data['title']}")
    
    # Simulate recording an exchange with a real question
    from wald_agent_reference.core.models import AgentResponse
    
    question = "What are the key risks highlighted by leadership?"
    expected_title = _generate_title_from_question(question)
    
    # Create a mock response
    response = AgentResponse(
        question=question,
        planned_approach=["Test"],
        executive_summary="Test",
        key_findings=["Finding 1"],
        calculations=[],
        evidence=["Evidence 1"],
        caveats=[],
        source_references=["source.csv"],
        visual_insights=[],
        plot_paths=[],
        plot_base64="",
    )
    
    # Record the exchange
    manager.record_exchange(session, question, response)
    
    # Check metadata was updated
    updated_data = json.loads(metadata_path.read_text())
    assert updated_data["title"] == expected_title
    
    print(f"✓ Title updated in metadata: {updated_data['title']}")
    
    # Verify load_chat returns the updated title
    reloaded_session = manager.load_chat(session.chat_id)
    assert reloaded_session.title == expected_title
    
    print(f"✓ Title loaded correctly: {reloaded_session.title}")


def test_api_response_includes_updated_title(chat_manager_fixture):
    """Verify that API response includes the dynamically updated title"""
    manager = chat_manager_fixture
    
    # Create a chat
    session = manager.create_chat()
    print(f"\n✓ Created chat with ID: {session.chat_id}")
    print(f"  Initial title: {session.title}")
    
    # Simulate an API response structure
    # (In real scenario, the /ask endpoint would do this)
    question = "Which departments are underperforming?"
    expected_title = _generate_title_from_question(question)
    
    # Create mock response
    from wald_agent_reference.core.models import AgentResponse
    response = AgentResponse(
        question=question,
        planned_approach=["Route: SQL"],
        executive_summary="Support and Finance are underperforming",
        key_findings=["Support at 61%", "Finance at 67%"],
        calculations=["Metric analysis performed"],
        evidence=["From department_scorecard.csv"],
        caveats=[],
        source_references=["department_scorecard.csv"],
        visual_insights=["Bar chart showing performance"],
        plot_paths=[],
        plot_base64="",
    )
    
    # Record the exchange (this updates the title in metadata)
    manager.record_exchange(session, question, response)
    
    # Reload session (simulating what the API endpoint does)
    updated_session = manager.load_chat(session.chat_id)
    
    # Simulate API response
    api_response = {
        "chat_id": session.chat_id,
        "title": updated_session.title,  # The updated title
        "question": question,
        "answer": response.executive_summary,
        "key_findings": response.key_findings,
        "report_url": f"/artifacts/{session.chat_id}/reports/"
    }
    
    print(f"\n✓ API Response structure:")
    print(f"  Title: {api_response['title']}")
    print(f"  Expected: {expected_title}")
    
    # Verify title in response
    assert api_response["title"] == expected_title
    print("✓ Title matches expected value in API response!")


def test_title_generator_falls_back_for_empty_question():
    """Verify blank or punctuation-only questions still produce a usable title."""
    assert _generate_title_from_question("") == "New Chat"
    assert _generate_title_from_question("   ?  ") == "New Chat"
