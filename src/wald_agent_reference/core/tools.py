from __future__ import annotations

from ..memory.memory_backends import build_memory_backend
from ..memory.structured_store import StructuredMemoryStore
from ..reasoning.calculator import CalculationEngine
from ..reasoning.sql_agent import SQLQueryAgent
from ..reasoning.visual_reasoner import VisualReasoner
from ..rendering.visualize import VisualizationEngine
from .config import AppSettings


class ToolRouter:
    def __init__(self, settings: AppSettings) -> None:
        self.calculator = CalculationEngine()
        self.visualizer = VisualizationEngine(settings)
        self.structured_store = StructuredMemoryStore(settings.structured_store_db_path)
        self.sql_agent = SQLQueryAgent(self.structured_store)
        self.memory_backend = build_memory_backend(settings)
        self.visual_reasoner = VisualReasoner()
