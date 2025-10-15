"""Unit tests for pybammsolvers core functionality."""

from __future__ import annotations

import pytest


class TestPackageStructure:
    """Test pybammsolvers package structure and metadata."""

    pytestmark = pytest.mark.unit

    def test_package_import(self):
        """
        Verify pybammsolvers package can be imported and exposes the idaklu module.

        The idaklu module is the core C++ extension that provides SUNDIALS IDA solver
        bindings with KLU sparse linear solver support.
        """
        import pybammsolvers

        assert hasattr(pybammsolvers, "idaklu")
        assert hasattr(pybammsolvers, "__version__")
        assert isinstance(pybammsolvers.__version__, str)
        assert len(pybammsolvers.__version__) > 0

    def test_idaklu_module_attributes(self, idaklu_module):
        """
        Verify idaklu module exposes the expected classes and functions.

        The module should provide Solution, VectorNdArray, VectorSolution for managing
        solver results, and IdakluJax for JAX integration.
        """
        # Core solution classes
        assert hasattr(idaklu_module, "solution")
        assert hasattr(idaklu_module, "VectorNdArray")
        assert hasattr(idaklu_module, "VectorSolution")

        # JAX integration
        assert hasattr(idaklu_module, "IdakluJax")
        assert hasattr(idaklu_module, "create_idaklu_jax")

        # Verify they're callable/instantiable
        assert callable(idaklu_module.solution)
        assert callable(idaklu_module.VectorNdArray)
        assert callable(idaklu_module.VectorSolution)
        assert callable(idaklu_module.create_idaklu_jax)
