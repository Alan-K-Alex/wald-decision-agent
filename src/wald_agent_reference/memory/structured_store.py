from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.models import DocumentChunk, ExtractedDocument, StructuredTable, VisualArtifact


@dataclass
class CatalogEntry:
    table_id: str
    sqlite_table: str
    source_file: str
    source_type: str
    logical_name: str
    columns: dict[str, str]
    metadata: dict[str, str]


class StructuredMemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS table_catalog (
                    table_id TEXT PRIMARY KEY,
                    sqlite_table TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    logical_name TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    extracted_text TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def persist_tables(self, tables: list[StructuredTable]) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        with self._connect() as conn:
            for table in tables:
                entry = self._persist_single(conn, table)
                entries.append(entry)
        return entries

    def _persist_single(self, conn: sqlite3.Connection, table: StructuredTable) -> CatalogEntry:
        sqlite_table = self._sqlite_table_name(table.table_id)
        frame = table.dataframe.copy()
        columns = {column: self._sqlite_column_name(column) for column in frame.columns}
        frame = frame.rename(columns=columns)
        frame.insert(0, "_row_id", range(1, len(frame) + 1))
        frame.to_sql(sqlite_table, conn, if_exists="replace", index=False)
        logical_name = table.metadata.get("table_name") or table.metadata.get("sheet_name") or table.table_id
        metadata = {
            "sheet_name": str(table.metadata.get("sheet_name", "")),
            "table_name": str(table.metadata.get("table_name", logical_name)),
            "source_file": table.source_path.name,
            "source_type": table.source_type,
        }
        conn.execute(
            """
            INSERT OR REPLACE INTO table_catalog
            (table_id, sqlite_table, source_file, source_type, logical_name, columns_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table.table_id,
                sqlite_table,
                table.source_path.name,
                table.source_type,
                logical_name,
                json.dumps(columns),
                json.dumps(metadata),
            ),
        )
        return CatalogEntry(
            table_id=table.table_id,
            sqlite_table=sqlite_table,
            source_file=table.source_path.name,
            source_type=table.source_type,
            logical_name=logical_name,
            columns=columns,
            metadata=metadata,
        )

    def load_catalog(self) -> list[CatalogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT table_id, sqlite_table, source_file, source_type, logical_name, columns_json, metadata_json FROM table_catalog"
            ).fetchall()
        entries: list[CatalogEntry] = []
        for row in rows:
            entries.append(
                CatalogEntry(
                    table_id=row[0],
                    sqlite_table=row[1],
                    source_file=row[2],
                    source_type=row[3],
                    logical_name=row[4],
                    columns=json.loads(row[5]),
                    metadata=json.loads(row[6]),
                )
            )
        return entries

    def get_all_tables(self) -> dict[str, StructuredTable] | None:
        """Get all tables from catalog."""
        catalog = self.load_catalog()
        if not catalog:
            return None
        return {entry.table_id: entry for entry in catalog}

    def load_all_chunks(self) -> list[DocumentChunk]:
        """Load all chunks from the store."""
        chunks: list[DocumentChunk] = []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, source_file, source_type, content, metadata_json FROM chunks"
                ).fetchall()
            for row in rows:
                chunks.append(
                    DocumentChunk(
                        chunk_id=row[0],
                        source_path=Path(row[1]),
                        source_type=row[2],
                        content=row[3],
                        metadata=json.loads(row[4]),
                    )
                )
        except Exception:
            pass
        return chunks

    def load_all_documents(self) -> dict[str, ExtractedDocument]:
        """Load all documents from the store."""
        documents: dict[str, ExtractedDocument] = {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT document_id, source_file, source_type, raw_text, metadata_json FROM documents"
                ).fetchall()
            for row in rows:
                doc = ExtractedDocument(
                    document_id=row[0],
                    source_path=Path(row[1]),
                    source_type=row[2],
                    raw_text=row[3],
                    metadata=json.loads(row[4]),
                )
                documents[row[0]] = doc
        except Exception:
            pass
        return documents

    def load_all_visuals(self) -> dict[str, VisualArtifact]:
        """Load all visual artifacts from the store."""
        visuals: dict[str, VisualArtifact] = {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT artifact_id, source_file, source_type, extracted_text, summary, metadata_json FROM visual_artifacts"
                ).fetchall()
            for row in rows:
                visual = VisualArtifact(
                    artifact_id=row[0],
                    source_path=Path(row[1]),
                    source_type=row[2],
                    extracted_text=row[3],
                    summary=row[4],
                    metadata=json.loads(row[5]),
                )
                visuals[row[0]] = visual
        except Exception:
            pass
        return visuals

    def execute(self, sql: str) -> tuple[list[str], list[tuple[object, ...]]]:
        with self._connect() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
        return columns, rows

    def persist_documents(self, documents: list[ExtractedDocument]) -> None:
        with self._connect() as conn:
            for document in documents:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (document_id, source_file, source_type, raw_text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        document.source_path.name,
                        document.source_type,
                        document.raw_text,
                        json.dumps(document.metadata),
                    ),
                )

    def persist_chunks(self, chunks: list[DocumentChunk]) -> None:
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, source_file, source_type, content, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source_path.name,
                        chunk.source_type,
                        chunk.content,
                        json.dumps(chunk.metadata),
                    ),
                )

    def persist_visual_artifacts(self, visuals: list[VisualArtifact]) -> None:
        with self._connect() as conn:
            for visual in visuals:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO visual_artifacts
                    (artifact_id, source_file, source_type, extracted_text, summary, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        visual.artifact_id,
                        visual.source_path.name,
                        visual.source_type,
                        visual.extracted_text,
                        visual.summary,
                        json.dumps(visual.metadata),
                    ),
                )

    @staticmethod
    def _sqlite_table_name(table_id: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9]+", "_", table_id.lower()).strip("_")
        return f"tbl_{base[:50]}"

    @staticmethod
    def _sqlite_column_name(column: object) -> str:
        text = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip().lower()).strip("_")
        return text or "column"
