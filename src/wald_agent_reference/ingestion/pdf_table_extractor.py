from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.logging import get_logger
from ..core.models import StructuredTable


class PDFTableExtractor:
    def __init__(self) -> None:
        self.logger = get_logger("ingestion.pdf_tables")

    def parse_file(self, path: Path) -> list[StructuredTable]:
        try:
            import pdfplumber
        except ImportError:
            return []

        tables: list[StructuredTable] = []
        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Try line-aware extraction first, then fall back to text heuristics for borderless tables.
                extracted = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 8,
                        "snap_tolerance": 3,
                    }
                ) or page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "intersection_tolerance": 8,
                    }
                ) or []
                for table_idx, raw_table in enumerate(extracted, start=1):
                    if not raw_table or len(raw_table) < 2:
                        continue
                    headers = [cell if cell else f"column_{idx + 1}" for idx, cell in enumerate(raw_table[0])]
                    frame = pd.DataFrame(raw_table[1:], columns=headers).dropna(how="all").reset_index(drop=True)
                    if frame.empty:
                        continue
                    frame.insert(0, "_source_row", list(range(2, len(frame) + 2)))
                    table_id = f"{path.stem}:page:{page_idx}:table:{table_idx}"
                    retrieval_text = (
                        f"PDF table from {path.name} page {page_idx}. "
                        f"Columns: {', '.join(map(str, frame.columns))}. "
                        f"Preview rows: {' | '.join(map(str, frame[[column for column in frame.columns if not str(column).startswith('_')]].head(1).fillna('').iloc[0].tolist()))}"
                    )
                    tables.append(
                        StructuredTable(
                            table_id=table_id,
                            source_path=path,
                            source_type="pdf_table",
                            dataframe=frame,
                            metadata={
                                "page": page_idx,
                                "sheet_name": f"Page {page_idx} Table {table_idx}",
                                "table_name": f"Page {page_idx} Table {table_idx}",
                                "source_file": path.name,
                                "columns": [column for column in frame.columns if not str(column).startswith("_")],
                                "source_range": f"Page {page_idx} Table {table_idx} rows 2-{len(frame) + 1}",
                                "header_rows": 1,
                                "row_count": len(frame),
                            },
                            retrieval_text=retrieval_text,
                        )
                    )
                if extracted:
                    self.logger.debug("Extracted %d tables from %s page %d", len(extracted), path.name, page_idx)
        return tables
