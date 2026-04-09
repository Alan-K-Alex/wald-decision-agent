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


def test_web_app_follow_up_question_uses_chat_context_without_polluting_new_topic(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    create_response = client.post("/api/chats")
    chat_id = create_response.json()["chat_id"]

    files = []
    for name in ["annual_report_2024.md", "board_financial_pack.xlsx", "messy_financial_pack.xlsx", "strategy_performance_pack.pdf"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    first_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "What are the primary risks in each region?", "generate_plot": "false"},
    )
    assert first_response.status_code == 200

    follow_up_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "How were the risk scores for these estimated?", "generate_plot": "false"},
    )
    assert follow_up_response.status_code == 200
    follow_up_answer = follow_up_response.json()["answer"].lower()
    assert "do not document the formula" in follow_up_answer or "do not have enough grounded context" in follow_up_answer
    assert "risk score" in follow_up_answer
    assert "performance score" not in follow_up_answer

    new_topic_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "What are the strategic priorities for next year?", "generate_plot": "false"},
    )
    assert new_topic_response.status_code == 200
    new_topic_answer = new_topic_response.json()["answer"].lower()
    assert "strategic priorities" in new_topic_answer or "planning cycle" in new_topic_answer
    assert "risk score" not in new_topic_answer


def test_web_app_follow_up_can_verify_units_from_prior_answer_context(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    chat_id = client.post("/api/chats").json()["chat_id"]
    files = []
    for name in ["board_financial_pack.xlsx", "strategy_performance_pack.pdf", "revenue_chart.svg"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    combined_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "why did europe miss the plan and how has the quarterly revenue trend ?", "generate_plot": "false"},
    )
    assert combined_response.status_code == 200

    follow_up_response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "is it specified any where it is in millions or you assumed ?", "generate_plot": "false"},
    )
    assert follow_up_response.status_code == 200

    answer = follow_up_response.json()["answer"].lower()
    assert "do not see an explicit unit" in answer or "assumption rather than a grounded fact" in answer
    assert "million" in answer or "unit" in answer


def test_web_app_combined_explanatory_query_keeps_causal_summary_and_linked_evidence(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    chat_id = client.post("/api/chats").json()["chat_id"]
    files = []
    for name in ["board_financial_pack.xlsx", "strategy_performance_pack.pdf", "revenue_chart.svg", "leadership_brief_long.txt"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "why did europe miss the plan and how has the quarterly revenue trend ?", "generate_plot": "false"},
    )
    assert response.status_code == 200

    payload = response.json()
    answer = payload["answer"].lower()
    assert "slower" in answer or "conversion" in answer or "weakness" in answer
    assert "trend" in answer or "q1" in answer
    assert any("](/artifacts/" in item for item in payload["evidence"])
    assert any(
        any(marker in item.lower() for marker in ["because", "slower", "weaker channel execution", "mid-market"])
        for item in payload["evidence"]
    )


def test_web_app_operational_update_query_uses_q2_update_document(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    chat_id = client.post("/api/chats").json()["chat_id"]
    files = []
    for name in ["q2_operational_update.md", "revenue_trend.csv"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "operartional updates for Q2 ?", "generate_plot": "false"},
    )
    assert response.status_code == 200

    payload = response.json()
    answer = payload["answer"].lower()
    assert "engineering and sales" in answer or "support and finance" in answer
    assert "revenue has the highest q2 2024 value" not in answer
    assert any("q2_operational_update.md" in item for item in payload["evidence"])


def test_web_app_supports_add_replace_and_delete_documents(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    chat_id = client.post("/api/chats").json()["chat_id"]

    first_upload = client.post(
        f"/api/chats/{chat_id}/upload",
        files=[("files", ("notes.md", b"# Notes\nSupport is under pressure.\n", "text/markdown"))],
        data={"incremental": "true"},
    )
    assert first_upload.status_code == 200
    assert first_upload.json()["file_count"] == 1

    second_upload = client.post(
        f"/api/chats/{chat_id}/upload",
        files=[("files", ("metrics.csv", b"Department,Performance Score\nSales,88\nSupport,61\n", "text/csv"))],
        data={"incremental": "true"},
    )
    assert second_upload.status_code == 200

    after_add = client.get(f"/api/chats/{chat_id}")
    assert after_add.status_code == 200
    assert after_add.json()["doc_count"] == 2

    replace_upload = client.post(
        f"/api/chats/{chat_id}/upload",
        files=[("files", ("replacement.md", b"# Replacement\nOnly this file should remain.\n", "text/markdown"))],
        data={"incremental": "false"},
    )
    assert replace_upload.status_code == 200

    after_replace = client.get(f"/api/chats/{chat_id}")
    assert after_replace.status_code == 200
    assert after_replace.json()["doc_count"] == 1

    delete_docs = client.post(f"/api/chats/{chat_id}/delete-documents")
    assert delete_docs.status_code == 200

    after_delete = client.get(f"/api/chats/{chat_id}")
    assert after_delete.status_code == 200
    assert after_delete.json()["doc_count"] == 0
    assert after_delete.json()["messages"] == []


def test_web_app_region_breakdown_query_reports_grounded_values_per_region(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    chat_id = client.post("/api/chats").json()["chat_id"]
    files = []
    for name in ["board_financial_pack.xlsx", "regional_actuals.csv"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "actual margin and gained for each regions ?", "generate_plot": "false"},
    )
    assert response.status_code == 200

    payload = response.json()
    answer = payload["answer"].lower()
    assert "north america actual margin = 31.00".lower() in answer
    assert "highest actual margin" not in answer
    assert "grounded metric named `gained`" in answer


def test_web_app_q2_operational_cost_query_returns_grounded_narrative_gap(tmp_path: Path) -> None:
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="local",
        memory_backend="none",
        vector_backend="hash",
        output_dir=str(tmp_path / "outputs"),
        chats_dir=str(tmp_path / "outputs" / "chats"),
    )
    client = TestClient(create_app(settings))
    root = Path(__file__).resolve().parents[1] / "data" / "raw"

    chat_id = client.post("/api/chats").json()["chat_id"]
    files = []
    for name in ["q2_operational_update.md", "annual_report_2024.md", "board_financial_pack.xlsx"]:
        path = root / name
        files.append(("files", (name, path.read_bytes(), "application/octet-stream")))

    upload_response = client.post(f"/api/chats/{chat_id}/upload", files=files)
    assert upload_response.status_code == 200

    response = client.post(
        f"/api/chats/{chat_id}/ask",
        data={"question": "operational costs q2 ?", "generate_plot": "false"},
    )
    assert response.status_code == 200

    payload = response.json()
    answer = payload["answer"].lower()
    assert "do not see an explicit numeric q2 operational cost value" in answer
    assert "highest actual cost" not in answer
    assert any("q2_operational_update.md" in item for item in payload["evidence"])
