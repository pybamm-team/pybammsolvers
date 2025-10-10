"""Test module-level functions.

This module tests the standalone functions provided by the idaklu module,
including observe, generate_function, and other utility functions.
"""

import pytest
import numpy as np


class TestObserveFunction:
    """Test the observe function."""

    pytestmark = pytest.mark.unit

    def test_observe_exists(self, idaklu_module):
        """Test that observe function exists."""
        assert hasattr(idaklu_module, "observe")
        assert callable(idaklu_module.observe)

    def test_observe_with_empty_arrays(self, idaklu_module):
        """Test observe with empty arrays raises TypeError."""
        with pytest.raises(TypeError):
            idaklu_module.observe(
                ts=np.array([]),
                ys=np.array([]),
                inputs=np.array([]),
                funcs=[],
                is_f_contiguous=True,
                shape=[],
            )


class TestObserveHermiteInterpFunction:
    """Test the observe_hermite_interp function."""

    pytestmark = pytest.mark.unit

    def test_observe_hermite_interp_exists(self, idaklu_module):
        """Test that observe_hermite_interp function exists."""
        assert hasattr(idaklu_module, "observe_hermite_interp")
        assert callable(idaklu_module.observe_hermite_interp)

    def test_observe_hermite_interp_with_empty_arrays(self, idaklu_module):
        """Test observe_hermite_interp with empty arrays raises TypeError."""
        with pytest.raises(TypeError):
            idaklu_module.observe_hermite_interp(
                t_interp=np.array([]),
                ts=np.array([]),
                ys=np.array([]),
                yps=np.array([]),
                inputs=np.array([]),
                funcs=[],
                shape=[],
            )


class TestGenerateFunction:
    """Test the generate_function."""

    pytestmark = pytest.mark.unit

    def test_generate_function_exists(self, idaklu_module):
        """Test that generate_function exists."""
        assert hasattr(idaklu_module, "generate_function")
        assert callable(idaklu_module.generate_function)

    def test_generate_function_with_empty_string(self, idaklu_module):
        """Test generate_function with empty string raises RuntimeError."""
        with pytest.raises(RuntimeError):
            idaklu_module.generate_function("")

    def test_generate_function_with_invalid_input(self, idaklu_module):
        """Test generate_function with invalid CasADi expression."""
        with pytest.raises(RuntimeError):
            idaklu_module.generate_function("invalid_casadi_expression")


class TestCreateCasadiSolverGroup:
    """Test the create_casadi_solver_group function."""

    pytestmark = pytest.mark.unit

    def test_create_casadi_solver_group_exists(self, idaklu_module):
        """Test that create_casadi_solver_group function exists."""
        assert hasattr(idaklu_module, "create_casadi_solver_group")
        assert callable(idaklu_module.create_casadi_solver_group)

    def test_create_casadi_solver_group_without_parameters(self, idaklu_module):
        """Test that function fails appropriately without parameters."""
        with pytest.raises(TypeError):
            idaklu_module.create_casadi_solver_group()


class TestCreateIdakluJax:
    """Test the create_idaklu_jax function."""

    pytestmark = pytest.mark.unit

    def test_create_idaklu_jax_exists(self, idaklu_module):
        """Test that create_idaklu_jax function exists."""
        assert hasattr(idaklu_module, "create_idaklu_jax")
        assert callable(idaklu_module.create_idaklu_jax)

    def test_create_idaklu_jax_callable(self, idaklu_module):
        """Test that create_idaklu_jax can be called."""
        result = idaklu_module.create_idaklu_jax()
        assert result is not None


class TestRegistrationsFunction:
    """Test the registrations function."""

    pytestmark = pytest.mark.unit

    def test_registrations_exists(self, idaklu_module):
        """Test that registrations function exists."""
        assert hasattr(idaklu_module, "registrations")
        assert callable(idaklu_module.registrations)

    def test_registrations_callable(self, idaklu_module):
        """Test that registrations function can be called."""
        result = idaklu_module.registrations()
        assert result is not None
