"""Test that the /api/chats/{chat_id}/ask endpoint returns concise response with title"""

import json
from pathlib import Path
import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_endpoint_returns_concise_response_with_title(client):
    """Verify /api/chats/{chat_id}/ask returns concise response + title"""
    
    # Create chat
    chat_response = client.post("/api/chats")
    assert chat_response.status_code == 200
    chat_id = chat_response.json()["chat_id"]
    
    # Upload documents
    sample_data = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not sample_data.exists():
        pytest.skip("Sample data not found")
    
    files_to_upload = list(sample_data.glob("*"))[:3]  # Upload first 3 files for speed
    if not files_to_upload:
        pytest.skip("No sample files to upload")
    
    with open(files_to_upload[0], "rb") as f:
        upload_response = client.post(
            f"/api/chats/{chat_id}/upload",
            files={"folder_contents": [
                (file.name, open(file, "rb")) for file in files_to_upload
            ]}
        )
    
    if upload_response.status_code != 200:
        pytest.skip("Failed to upload documents")
    
    # Ask question - title should be generated
    question1 = "Which regions underperformed against targets?"
    
    response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": question1, "generate_plot": True}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure contains concise fields
    assert "chat_id" in data
    assert "title" in data
    assert "answer" in data
    assert "key_findings" in data
    assert "visual_insights" in data
    assert "source_summary" in data
    assert "data_types_used" in data
    
    print(f"\n✓ Concise Response Fields Present:")
    print(f"  Title: {data['title'][:50]}...")
    print(f"  Answer: {data['answer'][:80]}...")
    print(f"  Key Findings: {len(data['key_findings'])} items")
    
    # Verify title was generated from question
    assert len(data['title']) > 0
    assert data['title'] != "New Chat"
    print(f"  Dynamic Title: {data['title']}")
    
    # Verify key_findings are limited to top 5
    assert len(data['key_findings']) <= 5
    
    # Verify no full methodology details in response
    # (those should be in PDF report only)
    response_str = json.dumps(data)
    assert "Planned Approach" not in response_str  # Should be in report, not chat
    assert "Calculations" not in response_str  # Should be in report, not chat
    
    print(f"\n✓ Full methodology NOT in response (moved to PDF report)")


@pytest.mark.integration  
@pytest.mark.slow
def test_endpoint_title_persists_across_questions(client):
    """Verify title stays consistent across multiple questions"""
    
    # Create chat
    chat_response = client.post("/api/chats")
    chat_id = chat_response.json()["chat_id"]
    
    # Upload documents
    sample_data = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not sample_data.exists():
        pytest.skip("Sample data not found")
    
    files_to_upload = list(sample_data.glob("*"))[:3]
    if not files_to_upload:
        pytest.skip("No sample files to upload")
    
    with open(files_to_upload[0], "rb") as f:
        client.post(
            f"/api/chats/{chat_id}/upload",
            files={"folder_contents": [(file.name, open(file, "rb")) for file in files_to_upload]}
        )
    
    question1 = "What is our revenue trend?"
    question2 = "How do we compare to targets?"
    
    # First question sets the title
    response1 = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": question1, "generate_plot": True}
    )
    title1 = response1.json()["title"]
    
    # Second question should keep same title
    response2 = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": question2, "generate_plot": True}
    )
    title2 = response2.json()["title"]
    
    assert title1 == title2, "Title should persist across questions"
    assert len(title1) > 0
    print(f"\n✓ Title Consistency: {title1}")

