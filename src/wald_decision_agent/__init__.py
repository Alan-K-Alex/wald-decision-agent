"""Wald Decision Agent."""

from dotenv import load_dotenv

# Load environment variables from .env file when package is imported
load_dotenv()

from .core import AppSettings, LeadershipInsightAgent, load_settings

WaldAgentReference = LeadershipInsightAgent

__all__ = ["AppSettings", "LeadershipInsightAgent", "WaldAgentReference", "load_settings"]
