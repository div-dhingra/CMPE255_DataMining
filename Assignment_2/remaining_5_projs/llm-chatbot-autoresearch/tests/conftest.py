"""Shared Pytest fixtures for LLM and Autoresearch test suite."""

import pytest
import torch
from starlette.testclient import TestClient

from src.api.app import app
from src.autoresearch.hill_climber import AutoresearchHillClimber, HyperparameterState
from src.autoresearch.ledger import ExperimentLedger
from src.autoresearch.objective import EvaluationObjective
from src.engine.chat_session import ChatSession
from src.engine.generator import GenerationConfig, TextGenerator
from src.model.config import ModelConfig
from src.model.tokenizer import Tokenizer
from src.model.transformer import SotaDecoderLLM


@pytest.fixture(scope="session")
def nano_config():
    """Lightweight config for fast unit testing."""
    return ModelConfig.preset_nano(
        dim=96,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=512,
        max_seq_len=256,
    )


@pytest.fixture(scope="session")
def tokenizer(nano_config):
    return Tokenizer(vocab_size=nano_config.vocab_size)


@pytest.fixture(scope="session")
def model(nano_config):
    torch.manual_seed(42)
    return SotaDecoderLLM(nano_config)


@pytest.fixture(scope="session")
def generator(model, tokenizer):
    return TextGenerator(model=model, tokenizer=tokenizer)


@pytest.fixture
def chat_session(generator):
    return ChatSession(generator=generator, mode="hybrid")


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def test_ledger():
    return ExperimentLedger()


@pytest.fixture
def test_climber(test_ledger, tokenizer):
    objective = EvaluationObjective(tokenizer=tokenizer)
    init_state = HyperparameterState(
        dim=96,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        context_window=128,
        temperature=0.7,
        top_p=0.9,
    )
    return AutoresearchHillClimber(
        objective=objective,
        ledger=test_ledger,
        initial_state=init_state,
        patience=2,
        max_restarts=2,
        seed=42,
    )
