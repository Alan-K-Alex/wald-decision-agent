from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from ..core.models import StructuredTable
from ..utils import coerce_text, compact_whitespace


@dataclass
class SpreadsheetParser:
    preview_rows: int = 5

    def parse_file(self, path: Path) -> list[StructuredTable]:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(path, sep=separator)
            return [self._table_from_dataframe(path, frame, sheet_name="Sheet1")]
        if suffix == ".xlsx":
            return self._parse_xlsx(path)
        if suffix == ".xls":
            excel = pd.ExcelFile(path)
            tables: list[StructuredTable] = []
            for sheet_name in excel.sheet_names:
                frame = excel.parse(sheet_name=sheet_name)
                tables.append(self._table_from_dataframe(path, frame, sheet_name=sheet_name))
            return tables
        raise ValueError(f"Unsupported spreadsheet file: {path}")

    def _parse_xlsx(self, path: Path) -> list[StructuredTable]:
        workbook = load_workbook(filename=path, data_only=True)
        tables: list[StructuredTable] = []
        for worksheet in workbook.worksheets:
            rows = list(worksheet.iter_rows(values_only=True))
            matrix = self._trim_matrix(rows)
            if not matrix:
                continue
            header_idx = self._find_header_row(matrix)
            headers = [self._normalize_header(cell, idx) for idx, cell in enumerate(matrix[header_idx])]
            body = matrix[header_idx + 1 :]
            frame = pd.DataFrame(body, columns=headers)
            tables.append(self._table_from_dataframe(path, frame, sheet_name=worksheet.title))
        return tables

    def _table_from_dataframe(self, path: Path, frame: pd.DataFrame, sheet_name: str) -> StructuredTable:
        normalized = frame.copy()
        normalized.columns = [self._normalize_header(column, idx) for idx, column in enumerate(normalized.columns)]
        normalized = normalized.dropna(how="all").reset_index(drop=True)
        retrieval_text = self._build_retrieval_text(path, sheet_name, normalized)
        table_id = f"{path.stem}:{sheet_name}"
        metadata = {
            "sheet_name": sheet_name,
            "source_file": path.name,
            "table_name": sheet_name,
            "columns": list(normalized.columns),
        }
        return StructuredTable(
            table_id=table_id,
            source_path=path,
            source_type="spreadsheet",
            dataframe=normalized,
            metadata=metadata,
            retrieval_text=retrieval_text,
        )

    def _build_retrieval_text(self, path: Path, sheet_name: str, frame: pd.DataFrame) -> str:
        headers = ", ".join(str(column) for column in frame.columns)
        preview = frame.head(self.preview_rows).fillna("").astype(str)
        preview_rows = []
        for _, row in preview.iterrows():
            preview_rows.append(" | ".join(compact_whitespace(value) for value in row.tolist()))
        body = "\n".join(preview_rows)
        return compact_whitespace(
            f"Spreadsheet {path.name} sheet {sheet_name}. Columns: {headers}. Preview rows: {body}"
        )

    @staticmethod
    def _trim_matrix(rows: list[tuple[object, ...]]) -> list[list[object]]:
        matrix = [list(row) for row in rows]
        while matrix and all(coerce_text(cell) == "" for cell in matrix[-1]):
            matrix.pop()
        if not matrix:
            return []

        max_cols = max(len(row) for row in matrix)
        for row in matrix:
            row.extend([None] * (max_cols - len(row)))

        active_cols = [
            idx
            for idx in range(max_cols)
            if any(coerce_text(row[idx]) != "" for row in matrix)
        ]
        return [[row[idx] for idx in active_cols] for row in matrix]

    @staticmethod
    def _find_header_row(matrix: list[list[object]]) -> int:
        for idx, row in enumerate(matrix):
            non_empty = sum(1 for cell in row if coerce_text(cell) != "")
            if non_empty >= max(1, len(row) // 2):
                return idx
        return 0

    @staticmethod
    def _normalize_header(value: object, idx: int) -> str:
        text = coerce_text(value)
        return text if text else f"column_{idx + 1}"
