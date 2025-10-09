"""Pytest configuration and fixtures for pybammsolvers tests."""

import pytest
import os


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture(scope="session")
def idaklu_module():
    """Fixture to provide the idaklu module."""
    try:
        import pybammsolvers
        return pybammsolvers.idaklu
    except ImportError as e:
        pytest.skip(f"Could not import pybammsolvers.idaklu: {e}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name.lower() or "TestVectorPerformance" in str(item.parent):
            item.add_marker(pytest.mark.slow)

        # Add integration marker
        if "integration" in item.name.lower() or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Ensure consistent backend for any plotting
    os.environ["MPLBACKEND"] = "Agg"
    
    # Set encoding for consistent behavior
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    yield
