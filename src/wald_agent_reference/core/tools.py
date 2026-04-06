from __future__ import annotations

from typing import Any

from ..memory.memory_backends import build_memory_backend
from ..memory.structured_store import StructuredMemoryStore
from ..reasoning.calculator import CalculationEngine
from ..reasoning.sql_agent import SQLQueryAgent
from ..reasoning.visual_reasoner import VisualReasoner
from ..rendering.visualize import VisualizationEngine
from .config import AppSettings
from .models_v2 import Corpus, TablePreview


class TableInspectionTool:
    """Tool for intelligent query planning - allows preview of table schemas"""
    
    def __init__(self, structured_store: StructuredMemoryStore) -> None:
        self.store = structured_store
    
    def get_table_preview(self, table_id: str) -> TablePreview | None:
        """Get schema preview for a specific table.
        
        Args:
            table_id: The table identifier
            
        Returns:
            TablePreview with schema, row count, and sample data, or None if not found
        """
        try:
            table = self.store.get_table(table_id)
            if table is None:
                return None
            
            # Infer data types from dataframe
            columns = {col: str(dtype) for col, dtype in table.dtypes.items()}
            
            # Get sample rows (up to 3)
            sample_rows = table.head(3).to_dict(orient='records')
            
            return TablePreview(
                table_name=table_id,
                source_file=str(table.attrs.get('source_path', 'unknown')),
                row_count=len(table),
                columns=columns,
                sample_rows=sample_rows,
                metadata={
                    'source_type': table.attrs.get('source_type', 'unknown'),
                    'ingestion_date': table.attrs.get('ingestion_date'),
                }
            )
        except Exception as e:
            return None
    
    def get_all_table_previews(self, catalog: Corpus) -> dict[str, TablePreview]:
        """Get previews for all available tables.
        
        Args:
            catalog: The data corpus containing tables
            
        Returns:
            Dictionary mapping table_id to TablePreview
        """
        previews = {}
        for table_id, structured_table in catalog.tables.items():
            try:
                df = structured_table.dataframe
                
                # Infer data types
                columns = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                # Get sample rows
                sample_rows = df.head(3).to_dict(orient='records')
                
                preview = TablePreview(
                    table_name=table_id,
                    source_file=str(structured_table.source_path),
                    row_count=len(df),
                    columns=columns,
                    sample_rows=sample_rows,
                    metadata=structured_table.metadata,
                )
                previews[table_id] = preview
            except Exception:
                continue
        
        return previews


class ToolRouter:
    def __init__(self, settings: AppSettings) -> None:
        self.calculator = CalculationEngine()
        self.visualizer = VisualizationEngine(settings)
        self.structured_store = StructuredMemoryStore(settings.structured_store_db_path)
        self.sql_agent = SQLQueryAgent(self.structured_store)
        self.table_inspector = TableInspectionTool(self.structured_store)
        self.memory_backend = build_memory_backend(settings)
        self.visual_reasoner = VisualReasoner()

