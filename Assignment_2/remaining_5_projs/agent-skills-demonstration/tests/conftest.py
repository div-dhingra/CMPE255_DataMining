"""Pytest fixtures and configuration for Agent Skills Demonstration."""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

# Ensure MPL cache dir is set
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_DIR.parent / ".mplcache"))

from src.modules.eda_profiler import load_raw_dataset
from src.modules.preprocessor import prepare_data_splits
from src.server.app import app


@pytest.fixture(scope="session")
def raw_df():
    """Load session-scoped raw Telco dataset."""
    return load_raw_dataset()


@pytest.fixture(scope="session")
def prepared_splits(raw_df):
    """Load session-scoped preprocessed splits."""
    return prepare_data_splits(raw_df)


@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient instance."""
    return TestClient(app)
