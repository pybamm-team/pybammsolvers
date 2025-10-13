"""Integration tests for pybammsolvers.

These tests verify interactions between different components
and may require more complex setup or be slower to run.
"""

import pytest
import numpy as np
import gc


class TestVectorIntegration:
    """Test integration between different vector types and operations."""

    pytestmark = pytest.mark.integration

    def test_mixed_vector_operations(self, idaklu_module):
        """Test mixed operations with different array types."""
        nd_vector = idaklu_module.VectorNdArray()

        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[1, 2], [3, 4]]),
            np.ones((5,)),
            np.zeros((2, 3)),
        ]

        for arr in arrays:
            nd_vector.append(arr.astype(np.float64))

        assert len(nd_vector) == len(arrays)

        # Verify retrieval maintains shape and values
        for i, original in enumerate(arrays):
            retrieved = nd_vector[i]
            np.testing.assert_array_equal(retrieved, original.astype(np.float64))


class TestErrorRecovery:
    """Test error handling and recovery in integration scenarios."""

    pytestmark = pytest.mark.integration

    def test_partial_failure_recovery(self, idaklu_module):
        """Test recovery from partial failures."""
        vector = idaklu_module.VectorNdArray()

        # Add valid arrays
        valid_arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        for arr in valid_arrays:
            vector.append(arr)

        assert len(vector) == 2

        # Try to add invalid data
        try:
            vector.append("invalid")
        except (TypeError, ValueError):
            pass  # Expected to fail

        # Verify valid data is still accessible
        assert len(vector) == 2
        np.testing.assert_array_equal(vector[0], valid_arrays[0])
        np.testing.assert_array_equal(vector[1], valid_arrays[1])

        # Should be able to continue adding valid data
        vector.append(np.array([5.0, 6.0]))
        assert len(vector) == 3

    def test_large_data_handling(self, idaklu_module):
        """Test handling of moderately large datasets."""
        vector = idaklu_module.VectorNdArray()

        large_arrays = []
        for _i in range(10):
            arr = np.random.rand(100, 50).astype(np.float64)
            large_arrays.append(arr)
            vector.append(arr)

        assert len(vector) == 10

        # Verify all arrays are accessible and correct
        for i, original in enumerate(large_arrays):
            retrieved = vector[i]
            assert retrieved.shape == original.shape
            np.testing.assert_array_equal(retrieved, original)


class TestMemoryManagement:
    """Test memory management in integration scenarios."""

    pytestmark = pytest.mark.integration

    def test_memory_cleanup_basic(self, idaklu_module):
        """Test that memory is properly cleaned up."""
        vectors = []

        # Create many vector objects
        for _ in range(100):
            vector = idaklu_module.VectorNdArray()
            for _ in range(10):
                arr = np.random.rand(100).astype(np.float64)
                vector.append(arr)
            vectors.append(vector)

        # Clear references
        vectors.clear()
        gc.collect()

        # If we get here without crashing, memory management is working

    def test_solution_vector_memory_cleanup(self, idaklu_module):
        """Test memory cleanup for solution vectors."""
        vectors = []

        # Create many solution vector objects
        for _ in range(100):
            vector = idaklu_module.VectorSolution()
            vectors.append(vector)

        # Clear references
        vectors.clear()
        gc.collect()

        # If we get here without crashing, memory management is working


class TestStressConditions:
    """Test behavior under stress conditions."""

    pytestmark = pytest.mark.integration

    def test_concurrent_access_simulation(self, idaklu_module):
        """Simulate concurrent access patterns (single-threaded)."""
        vector = idaklu_module.VectorNdArray()

        # Add initial data
        for i in range(100):
            arr = np.array([i, i + 1, i + 2], dtype=np.float64)
            vector.append(arr)

        # Simulate mixed read/write operations
        for _ in range(1000):
            # Random read
            idx = np.random.randint(0, len(vector))
            _ = vector[idx]

            # Occasional write
            if np.random.rand() < 0.1:  # 10% chance
                new_arr = np.random.rand(3).astype(np.float64)
                vector.append(new_arr)

        # Verify integrity
        assert len(vector) >= 100

    def test_boundary_stress(self, idaklu_module):
        """Test boundary conditions under stress."""
        vector = idaklu_module.VectorNdArray()

        # Mix of extreme values
        extreme_arrays = [
            np.array([1e-100], dtype=np.float64),
            np.array([1e100], dtype=np.float64),
            np.array([np.inf], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([np.finfo(np.float64).tiny], dtype=np.float64),
        ]

        for arr in extreme_arrays:
            vector.append(arr)

        # Repeatedly access these
        for _ in range(100):
            for i in range(len(vector)):
                retrieved = vector[i]
                assert retrieved is not None
