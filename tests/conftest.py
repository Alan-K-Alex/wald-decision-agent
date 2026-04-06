from __future__ import annotations

import sys
import shutil
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Register custom pytest marks
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "memory: marks tests as memory tests")


@pytest.fixture
def client():
    """Provide FastAPI TestClient for endpoint testing"""
    from wald_agent_reference.web.app import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def chat_id():
    """Provide a unique chat ID for testing"""
    return str(uuid.uuid4())[:8]


@pytest.fixture
def temp_docs_dir(tmp_path):
    """Provide a temporary directory with sample documents"""
    # Copy sample data from data/raw
    sample_data = ROOT / "data" / "raw"
    if sample_data.exists():
        for file in sample_data.glob("*"):
            if file.is_file():
                shutil.copy(file, tmp_path / file.name)
    return tmp_path
