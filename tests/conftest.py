"""Pytest configuration and fixtures for pybammsolvers tests."""

import pytest
import os


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture(scope="session")
def idaklu_module():
    """Fixture to provide the idaklu module."""
    try:
        import pybammsolvers

        return pybammsolvers.idaklu
    except ImportError as e:
        pytest.skip(f"Could not import pybammsolvers.idaklu: {e}")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Ensure consistent backend for any plotting
    os.environ["MPLBACKEND"] = "Agg"

    # Set encoding for consistent behavior
    os.environ["PYTHONIOENCODING"] = "utf-8"

    yield
