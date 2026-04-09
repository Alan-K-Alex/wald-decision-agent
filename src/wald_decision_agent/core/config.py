from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = "wald-decision-agent"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    chat_model: str = "gemini-2.5-flash"
    vision_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    llm_provider: str = "huggingface"
    enable_llm_formatting: bool = True
    vector_backend: str = "auto"
    vector_dim: int = 256
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    retrieval_backend: str = "auto"
    vector_store_dir: str = "outputs/vector_store"
    structured_store_path: str = "outputs/structured_memory.db"
    log_level: str = "INFO"
    log_file: str = "outputs/logs/agent.log"
    plot_dpi: int = 150
    output_dir: str = "outputs"
    reports_dir: str = "outputs/reports"
    plots_dir: str = "outputs/plots"
    chats_dir: str = "outputs/chats"

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    @property
    def huggingface_api_key(self) -> str | None:
        return os.getenv("HUGGINGFACE_API_KEY")

    @property
    def groq_api_key(self) -> str | None:
        return os.getenv("GROQ_API_KEY")

    @property
    def active_api_key(self) -> str | None:
        if self.llm_provider == "groq":
            return self.groq_api_key
        if self.llm_provider == "huggingface":
            return self.huggingface_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.groq_api_key or self.huggingface_api_key or self.gemini_api_key or self.openai_api_key

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def reports_path(self) -> Path:
        return Path(self.reports_dir)

    @property
    def plots_path(self) -> Path:
        return Path(self.plots_dir)

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def structured_store_db_path(self) -> Path:
        return Path(self.structured_store_path)

    @property
    def log_path(self) -> Path:
        return Path(self.log_file)

    @property
    def chats_path(self) -> Path:
        return Path(self.chats_dir)


def load_settings(path: str | Path = "config/settings.yaml") -> AppSettings:
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        settings = AppSettings()
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        settings = AppSettings(**raw)
    
    # Ensure base directories exist
    settings.output_path.mkdir(parents=True, exist_ok=True)
    settings.chats_path.mkdir(parents=True, exist_ok=True)
    
    return settings
