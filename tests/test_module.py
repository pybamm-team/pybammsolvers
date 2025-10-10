"""Test module imports and basic structure.

This module consolidates all tests related to module imports,
class/function availability, and basic documentation.
"""

import pytest
import inspect
import io
import contextlib


class TestImport:
    """Test basic module import functionality."""

    pytestmark = pytest.mark.unit

    def test_pybammsolvers_import(self):
        """Test that pybammsolvers can be imported."""
        import pybammsolvers

        assert pybammsolvers is not None

    def test_idaklu_module_import(self, idaklu_module):
        """Test that idaklu module is accessible."""
        assert idaklu_module is not None

    def test_version_import(self):
        """Test that version can be imported."""
        from pybammsolvers.version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestClasses:
    """Test that all expected classes are available."""

    pytestmark = pytest.mark.unit

    def test_solver_group_class_exists(self, idaklu_module):
        """Test that IDAKLUSolverGroup class exists."""
        assert hasattr(idaklu_module, "IDAKLUSolverGroup")
        assert callable(idaklu_module.IDAKLUSolverGroup)

    def test_idaklu_jax_class_exists(self, idaklu_module):
        """Test that IdakluJax class exists."""
        assert hasattr(idaklu_module, "IdakluJax")
        assert callable(idaklu_module.IdakluJax)

    def test_solution_class_exists(self, idaklu_module):
        """Test that solution class exists."""
        assert hasattr(idaklu_module, "solution")
        assert callable(idaklu_module.solution)

    def test_vector_classes_exist(self, idaklu_module):
        """Test that vector classes exist."""
        assert hasattr(idaklu_module, "VectorNdArray")
        assert hasattr(idaklu_module, "VectorRealtypeNdArray")
        assert hasattr(idaklu_module, "VectorSolution")

    def test_function_class_exists(self, idaklu_module):
        """Test that Function class exists (CasADi)."""
        assert hasattr(idaklu_module, "Function")


class TestFunctions:
    """Test that all expected functions are available."""

    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize(
        "func_name",
        [
            "create_casadi_solver_group",
            "observe",
            "observe_hermite_interp",
            "generate_function",
            "create_idaklu_jax",
            "registrations",
        ],
    )
    def test_function_exists_and_callable(self, idaklu_module, func_name):
        """Test that critical functions exist and are callable."""
        assert hasattr(idaklu_module, func_name), f"Missing function: {func_name}"
        func = getattr(idaklu_module, func_name)
        assert callable(func), f"Function {func_name} is not callable"


class TestDocumentation:
    """Test module and class documentation."""

    pytestmark = pytest.mark.unit

    def test_module_has_docstring(self, idaklu_module):
        """Test that the idaklu module has documentation."""
        assert hasattr(idaklu_module, "__doc__")
        assert idaklu_module.__doc__ is not None
        assert len(idaklu_module.__doc__.strip()) > 0

    @pytest.mark.parametrize(
        "class_name",
        ["IDAKLUSolverGroup", "IdakluJax", "solution"],
    )
    def test_class_has_docstring(self, idaklu_module, class_name):
        """Test that main classes have docstrings."""
        cls = getattr(idaklu_module, class_name)
        assert hasattr(cls, "__doc__")

    def test_help_functionality(self, idaklu_module):
        """Test that help() works on main components."""
        components = [
            idaklu_module,
            idaklu_module.solution,
            idaklu_module.VectorNdArray,
        ]

        for component in components:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                help(component)
            help_text = f.getvalue()
            assert len(help_text) > 0

    @pytest.mark.parametrize(
        "func_name",
        [
            "create_casadi_solver_group",
            "observe",
            "observe_hermite_interp",
            "generate_function",
        ],
    )
    def test_function_has_signature(self, idaklu_module, func_name):
        """Test that functions have reasonable signatures."""
        func = getattr(idaklu_module, func_name)

        # Should be callable
        assert callable(func)

        # Try to get signature (might fail for C++ functions)
        try:
            sig = inspect.signature(func)
            assert sig is not None
        except (ValueError, TypeError):
            # C++ functions might not have inspectable signatures
            pytest.skip(f"{func_name} signature not inspectable (C++ binding)")


class TestBasicFunctionality:
    """Test basic functionality that doesn't require complex setup."""

    pytestmark = pytest.mark.unit

    def test_registrations_function(self, idaklu_module):
        """Test that registrations function can be called."""
        result = idaklu_module.registrations()
        assert result is not None

    def test_create_idaklu_jax_function(self, idaklu_module):
        """Test that create_idaklu_jax function can be called."""
        result = idaklu_module.create_idaklu_jax()
        assert result is not None

    def test_solver_group_creation_without_parameters(self, idaklu_module):
        """Test that solver group creation fails appropriately without parameters."""
        with pytest.raises(TypeError):
            idaklu_module.create_casadi_solver_group()


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    pytestmark = pytest.mark.unit

    def test_generate_function_with_empty_string(self, idaklu_module):
        """Test generate_function with empty string."""
        with pytest.raises(RuntimeError):
            idaklu_module.generate_function("")

    def test_generate_function_with_invalid_expression(self, idaklu_module):
        """Test generate_function with invalid CasADi expression."""
        with pytest.raises(RuntimeError):
            idaklu_module.generate_function("invalid_casadi_expression")
