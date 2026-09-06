"""Shared Pytest fixtures and test environment setup."""

import pytest
from typing import List
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.core.dataset import TransactionDataset, SyntheticTransactionGenerator


@pytest.fixture(scope="session")
def client():
    """Provides a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_transactions() -> List[frozenset[str]]:
    """Small deterministic transaction database for algorithmic testing."""
    return [
        frozenset(["Milk", "Bread", "Butter"]),
        frozenset(["Bread", "Butter"]),
        frozenset(["Milk", "Bread", "Butter", "Eggs"]),
        frozenset(["Beer", "Diapers"]),
        frozenset(["Milk", "Bread", "Butter"]),
        frozenset(["Bread", "Butter", "Diapers"]),
        frozenset(["Milk", "Eggs"]),
        frozenset(["Beer", "Diapers", "Milk"]),
    ]


@pytest.fixture
def sample_dataset(sample_transactions) -> TransactionDataset:
    """Provides a TransactionDataset instance."""
    return TransactionDataset(transactions=[set(t) for t in sample_transactions])


@pytest.fixture
def synthetic_dataset() -> TransactionDataset:
    """Provides a generated synthetic dataset."""
    return SyntheticTransactionGenerator.generate(num_transactions=300, seed=123)
