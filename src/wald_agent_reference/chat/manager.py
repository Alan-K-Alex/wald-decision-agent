from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.config import AppSettings
from ..core.logging import get_logger
from ..core.models import AgentResponse
from ..utils import slugify


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_title_from_question(question: str, max_words: int = 7) -> str:
    """Generate a concise chat title from a question.
    
    Extracts the first N words, removes question marks, and creates a title.
    Examples:
        "What is our current revenue trend?" -> "What is our current revenue trend"
        "Which regions missed revenue plan by the largest amount?" -> "Which regions missed revenue plan by..."
    """
    # Remove leading/trailing whitespace and question marks
    cleaned = question.strip().rstrip('?').strip()
    # Split into words and take first max_words
    words = cleaned.split()
    title = " ".join(words[:max_words])
    # If question was longer, add ellipsis
    if len(words) > max_words:
        title += "..."
    return title


@dataclass
class ChatSession:
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    root_dir: Path
    docs_dir: Path
    artifacts_dir: Path
    db_path: Path
    messages_path: Path
    metadata_path: Path


class ChatSettingsFactory:
    def __init__(self, base_settings: AppSettings) -> None:
        self.base_settings = base_settings

    def build(self, session: ChatSession) -> AppSettings:
        values = self.base_settings.model_dump()
        values.update(
            {
                "output_dir": str(session.root_dir),
                "reports_dir": str(session.artifacts_dir / "reports"),
                "plots_dir": str(session.artifacts_dir / "plots"),
                "vector_store_dir": str(session.root_dir / "vector_store"),
                "structured_store_path": str(session.db_path),
                "log_file": str(session.root_dir / "logs" / "chat.log"),
                "supermemory_container_tag": f"wald-agent-reference-{session.chat_id}",
            }
        )
        return AppSettings(**values)


class ChatManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.root = settings.chats_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("chat.manager")

    def create_chat(self, title: str | None = None) -> ChatSession:
        chat_id = uuid4().hex[:12]
        root_dir = self.root / chat_id
        docs_dir = root_dir / "docs"
        artifacts_dir = root_dir / "artifacts"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        session = ChatSession(
            chat_id=chat_id,
            title=title or "New Chat",
            created_at=now,
            updated_at=now,
            root_dir=root_dir,
            docs_dir=docs_dir,
            artifacts_dir=artifacts_dir,
            db_path=root_dir / "structured_memory.db",
            messages_path=root_dir / "messages.json",
            metadata_path=root_dir / "chat.json",
        )
        self._write_metadata(session)
        self._write_messages(session, [])
        self.logger.info("Created chat %s (%s)", session.chat_id, session.title)
        return session

    def list_chats(self) -> list[dict[str, Any]]:
        chats: list[dict[str, Any]] = []
        for metadata_path in sorted(self.root.glob("*/chat.json"), reverse=True):
            chat = json.loads(metadata_path.read_text(encoding="utf-8"))
            chats.append(chat)
        return sorted(chats, key=lambda item: item["updated_at"], reverse=True)

    def load_chat(self, chat_id: str) -> ChatSession:
        root_dir = self.root / chat_id
        metadata_path = root_dir / "chat.json"
        if not metadata_path.exists():
            raise FileNotFoundError(chat_id)
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return ChatSession(
            chat_id=chat_id,
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            root_dir=root_dir,
            docs_dir=root_dir / "docs",
            artifacts_dir=root_dir / "artifacts",
            db_path=root_dir / "structured_memory.db",
            messages_path=root_dir / "messages.json",
            metadata_path=metadata_path,
        )

    def upload_files(self, session: ChatSession, files: list[tuple[str, bytes]], incremental: bool = True) -> dict[str, int]:
        """
        Upload files to a chat.
        
        Args:
            session: The chat session
            files: List of (relative_path, content) tuples
            incremental: If True, append to existing files. If False, replace all files.
        """
        # Only clear runtime state if doing a full replacement (non-incremental)
        if not incremental:
            self._clear_runtime_state(session)
        else:
            # For incremental uploads, just clear the ingestion metadata cache
            # so new files are re-ingested on the next query
            metadata_file = session.root_dir / ".ingestion_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            # Also clear vector store to force re-indexing with new files
            vector_store_dir = session.root_dir / "vector_store"
            if vector_store_dir.exists():
                shutil.rmtree(vector_store_dir)
            self.logger.debug("Prepared for incremental upload for chat %s", session.chat_id)
        
        written = 0
        for relative_path, content in files:
            safe_relative = Path(relative_path)
            destination = session.docs_dir / safe_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            written += 1
        self._touch(session, title=session.title)
        self.logger.info("Uploaded %d files into chat %s (incremental=%s)", written, session.chat_id, incremental)
        return {"file_count": written}

    def clear_chat_data(self, session: ChatSession) -> None:
        """Clear all analysis data while keeping uploaded documents."""
        # Clear vector store to force re-indexing
        vector_store_dir = session.root_dir / "vector_store"
        if vector_store_dir.exists():
            shutil.rmtree(vector_store_dir)
        
        # Clear artifacts (plots/reports)
        for path in [session.artifacts_dir, session.root_dir / "logs"]:
            if path.exists():
                shutil.rmtree(path)
        session.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        # Clear database and ingestion cache
        for suffix_path in [session.db_path, session.db_path.with_suffix(".db-shm"), session.db_path.with_suffix(".db-wal")]:
            if suffix_path.exists():
                suffix_path.unlink()
        
        metadata_file = session.root_dir / ".ingestion_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Clear chat messages
        self._write_messages(session, [])
        self.logger.info("Cleared analysis data for chat %s", session.chat_id)

    def delete_uploaded_documents(self, session: ChatSession) -> None:
        """Delete all uploaded documents and dependent data (analysis, history, etc)."""
        # Delete uploaded documents
        if session.docs_dir.exists():
            shutil.rmtree(session.docs_dir)
        session.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear vector store
        vector_store_dir = session.root_dir / "vector_store"
        if vector_store_dir.exists():
            shutil.rmtree(vector_store_dir)
        
        # Clear artifacts (plots/reports depend on documents)
        for path in [session.artifacts_dir, session.root_dir / "logs"]:
            if path.exists():
                shutil.rmtree(path)
        session.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        # Clear database (ingestion data is doc-specific)
        for suffix_path in [session.db_path, session.db_path.with_suffix(".db-shm"), session.db_path.with_suffix(".db-wal")]:
            if suffix_path.exists():
                suffix_path.unlink()
        
        # Clear ingestion metadata
        metadata_file = session.root_dir / ".ingestion_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Clear chat messages (they reference deleted documents)
        self._write_messages(session, [])
        
        self.logger.info("Deleted all uploaded documents for chat %s", session.chat_id)

    def record_exchange(self, session: ChatSession, question: str, response: AgentResponse) -> None:
        messages = self.load_messages(session)
        messages.append({"role": "user", "content": question, "timestamp": _utc_now()})
        report_path = session.artifacts_dir / "reports" / f"{slugify(question)}.md"
        messages.append(
            {
                "role": "assistant",
                "content": response.executive_summary,
                "timestamp": _utc_now(),
                "plot_paths": [str(path) for path in response.plot_paths],
                "report_path": str(report_path),
                "markdown": response.to_markdown(),
            }
        )
        title = session.title if session.title != "New Chat" else _generate_title_from_question(question)
        self._write_messages(session, messages)
        self._touch(session, title=title)
        self.logger.info("Recorded exchange for chat %s | question=%s", session.chat_id, question)

    def load_messages(self, session: ChatSession) -> list[dict[str, Any]]:
        if not session.messages_path.exists():
            return []
        return json.loads(session.messages_path.read_text(encoding="utf-8"))

    def delete_chat(self, session: ChatSession) -> None:
        if session.root_dir.exists():
            shutil.rmtree(session.root_dir)
            self.logger.info("Deleted chat %s and its local artifacts", session.chat_id)

    def _clear_runtime_state(self, session: ChatSession) -> None:
        if session.docs_dir.exists():
            shutil.rmtree(session.docs_dir)
        session.docs_dir.mkdir(parents=True, exist_ok=True)
        for path in [session.artifacts_dir, session.root_dir / "vector_store", session.root_dir / "logs"]:
            if path.exists():
                shutil.rmtree(path)
        session.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (session.artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
        for suffix_path in [session.db_path, session.db_path.with_suffix(".db-shm"), session.db_path.with_suffix(".db-wal")]:
            if suffix_path.exists():
                suffix_path.unlink()
        self._write_messages(session, [])
        self.logger.info("Cleared runtime state for chat %s", session.chat_id)

    def _touch(self, session: ChatSession, title: str) -> None:
        metadata = {
            "chat_id": session.chat_id,
            "title": title,
            "created_at": session.created_at,
            "updated_at": _utc_now(),
        }
        session.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _write_metadata(self, session: ChatSession) -> None:
        self._touch(session, title=session.title)

    @staticmethod
    def _write_messages(session: ChatSession, messages: list[dict[str, Any]]) -> None:
        session.messages_path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
