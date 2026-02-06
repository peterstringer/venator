"""Shared pytest fixtures for Venator tests."""

import numpy as np
import pytest

from venator.config import VenatorConfig


@pytest.fixture
def config() -> VenatorConfig:
    """Fresh config instance for testing."""
    return VenatorConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_benign_activations(rng: np.random.Generator) -> np.ndarray:
    """Synthetic benign activations: 200 samples, 50 features, drawn from N(0, I).

    Shape: (200, 50)
    """
    return rng.standard_normal((200, 50)).astype(np.float32)


@pytest.fixture
def synthetic_anomalous_activations(rng: np.random.Generator) -> np.ndarray:
    """Synthetic anomalous activations: 50 samples, 50 features, shifted mean.

    Shifted by +5 in all dimensions to be clearly anomalous.
    Shape: (50, 50)
    """
    return (rng.standard_normal((50, 50)) + 5.0).astype(np.float32)
