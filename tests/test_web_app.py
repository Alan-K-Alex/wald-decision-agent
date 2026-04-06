from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.web import create_app


def test_web_app_chat_upload_ask_delete_lifecycle(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))

    create_response = client.post("/api/chats")
    assert create_response.status_code == 200
    chat_id = create_response.json()["chat_id"]

    upload_response = client.post(
        f"/api/chats/{chat_id}/upload",
        files=[
            ("files", ("notes.md", b"# Update\nRevenue expanded steadily.\n", "text/markdown")),
            ("files", ("metrics.csv", b"Department,Performance Score\nSales,88\nSupport,61\n", "text/csv")),
        ],
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["documents"] >= 1
    assert upload_response.json()["tables"] >= 1

    chat_response = client.get(f"/api/chats/{chat_id}")
    assert chat_response.status_code == 200
    db_path = Path(chat_response.json()["db_path"])
    assert db_path.name == "structured_memory.db"
    assert db_path.exists()

    ask_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "Which department has the lowest performance score?", "generate_plot": "false"},
    )
    assert ask_response.status_code == 200
    assert "Support" in ask_response.json()["markdown"]

    delete_response = client.delete(f"/api/chats/{chat_id}")
    assert delete_response.status_code == 200
    assert not db_path.exists()
    assert not (tmp_path / "outputs" / "chats" / chat_id).exists()
