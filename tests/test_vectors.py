"""Test vector container classes.

This module consolidates all tests related to VectorNdArray,
VectorRealtypeNdArray, and VectorSolution classes.
"""

import pytest
import numpy as np


class TestVectorNdArrayBasic:
    """Test basic VectorNdArray functionality."""

    def test_creation(self, idaklu_module):
        """Test VectorNdArray can be created."""
        vector = idaklu_module.VectorNdArray()
        assert vector is not None
        assert len(vector) == 0

    def test_append_and_access(self, idaklu_module):
        """Test appending arrays and accessing them."""
        vector = idaklu_module.VectorNdArray()
        
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        
        vector.append(arr1)
        assert len(vector) == 1
        
        vector.append(arr2)
        assert len(vector) == 2
        
        # Test access
        retrieved = vector[0]
        np.testing.assert_array_equal(retrieved, arr1)
        
        retrieved = vector[1]
        np.testing.assert_array_equal(retrieved, arr2)

    def test_empty_vector_access_raises_error(self, idaklu_module):
        """Test that accessing empty vector raises IndexError."""
        vector = idaklu_module.VectorNdArray()
        with pytest.raises(IndexError):
            _ = vector[0]

    def test_multiple_arrays_different_shapes(self, idaklu_module):
        """Test vector can hold arrays of different shapes."""
        vector = idaklu_module.VectorNdArray()
        
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[1, 2], [3, 4]]),
            np.ones((5,)),
            np.zeros((2, 3)),
        ]
        
        for arr in arrays:
            vector.append(arr.astype(np.float64))
        
        assert len(vector) == len(arrays)
        
        # Verify retrieval
        for i, original in enumerate(arrays):
            retrieved = vector[i]
            np.testing.assert_array_equal(retrieved, original.astype(np.float64))


class TestVectorNdArrayEdgeCases:
    """Test edge cases for VectorNdArray."""

    def test_zero_dimensional_arrays(self, idaklu_module):
        """Test handling of 0-dimensional arrays."""
        vector = idaklu_module.VectorNdArray()
        
        scalar_array = np.array(42.0)
        assert scalar_array.ndim == 0
        
        vector.append(scalar_array)
        assert len(vector) == 1
        
        retrieved = vector[0]
        assert retrieved.shape == ()
        assert float(retrieved) == 42.0

    def test_large_dimensions(self, idaklu_module):
        """Test arrays with many dimensions but small total size."""
        vector = idaklu_module.VectorNdArray()
        
        shape = (1, 1, 1, 1, 1, 1, 1, 1, 2)
        arr = np.ones(shape, dtype=np.float64)
        
        vector.append(arr)
        assert len(vector) == 1
        
        retrieved = vector[0]
        assert retrieved.shape == shape
        np.testing.assert_array_equal(retrieved, arr)

    def test_special_float_values(self, idaklu_module):
        """Test handling of inf and nan."""
        vector = idaklu_module.VectorNdArray()
        
        # Test with infinity
        inf_array = np.array([np.inf, -np.inf, 1.0])
        vector.append(inf_array)
        retrieved = vector[0]
        assert np.isinf(retrieved[0])
        assert np.isinf(retrieved[1])
        assert retrieved[2] == 1.0
        
        # Test with NaN
        nan_array = np.array([np.nan, 1.0, 2.0])
        vector.append(nan_array)
        retrieved = vector[-1]
        assert np.isnan(retrieved[0])
        assert retrieved[1] == 1.0

    def test_extremely_small_values(self, idaklu_module):
        """Test handling of extremely small values."""
        vector = idaklu_module.VectorNdArray()
        
        tiny_values = np.array([
            np.finfo(np.float64).tiny,
            np.finfo(np.float64).eps,
            1e-300,
            0.0,
        ])
        
        vector.append(tiny_values)
        retrieved = vector[0]
        np.testing.assert_array_equal(retrieved, tiny_values)

    def test_extremely_large_values(self, idaklu_module):
        """Test handling of extremely large values."""
        vector = idaklu_module.VectorNdArray()
        
        large_values = np.array([
            1e100,
            1e200,
            np.finfo(np.float64).max / 2,
            -1e100,
        ])
        
        vector.append(large_values)
        retrieved = vector[0]
        np.testing.assert_array_equal(retrieved, large_values)

    def test_negative_indexing(self, idaklu_module):
        """Test negative indexing behavior."""
        vector = idaklu_module.VectorNdArray()
        
        arrays = [np.array([i], dtype=np.float64) for i in range(5)]
        for arr in arrays:
            vector.append(arr)
        
        try:
            last_elem = vector[-1]
            assert last_elem[0] == 4.0
            
            second_last = vector[-2]
            assert second_last[0] == 3.0
        except (IndexError, TypeError):
            pytest.skip("Negative indexing not supported")

    @pytest.mark.slow
    def test_maximum_vector_size(self, idaklu_module):
        """Test vector behavior near maximum reasonable size."""
        vector = idaklu_module.VectorNdArray()
        
        max_size = 1000
        for i in range(max_size):
            arr = np.array([i], dtype=np.float64)
            vector.append(arr)
            
            if i % 100 == 0 and i > 0:
                first_elem = vector[0]
                assert first_elem[0] == 0.0
        
        assert len(vector) == max_size
        
        # Test access at key positions
        for i in [0, max_size // 2, max_size - 1]:
            elem = vector[i]
            assert elem[0] == float(i)


class TestVectorNdArrayTypeHandling:
    """Test type coercion and conversion behavior."""

    def test_integer_array_coercion(self, idaklu_module):
        """Test coercion of integer arrays."""
        vector = idaklu_module.VectorNdArray()
        
        int_array = np.array([1, 2, 3], dtype=np.int32)
        vector.append(int_array)
        retrieved = vector[0]
        
        assert retrieved.dtype in [np.float64, np.float32, np.int32, np.int64]
        np.testing.assert_array_equal(retrieved.astype(int), [1, 2, 3])

    def test_complex_array_handling(self, idaklu_module):
        """Test handling of complex arrays."""
        vector = idaklu_module.VectorNdArray()
        
        complex_array = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        vector.append(complex_array)
        retrieved = vector[0]
        assert retrieved is not None

    def test_boolean_array_handling(self, idaklu_module):
        """Test handling of boolean arrays."""
        vector = idaklu_module.VectorNdArray()
        
        bool_array = np.array([True, False, True], dtype=bool)
        vector.append(bool_array)
        retrieved = vector[0]
        assert retrieved.dtype in [bool, np.int32, np.int64, np.float32, np.float64]


class TestVectorRealtypeNdArray:
    """Test VectorRealtypeNdArray functionality."""

    def test_creation(self, idaklu_module):
        """Test VectorRealtypeNdArray can be created."""
        vector = idaklu_module.VectorRealtypeNdArray()
        assert vector is not None
        assert len(vector) == 0


class TestVectorSolution:
    """Test VectorSolution functionality."""

    def test_creation(self, idaklu_module):
        """Test VectorSolution can be created."""
        vector = idaklu_module.VectorSolution()
        assert vector is not None
        assert len(vector) == 0
        assert hasattr(vector, "__len__")

    def test_has_append_method(self, idaklu_module):
        """Test VectorSolution has append method."""
        vector = idaklu_module.VectorSolution()
        assert hasattr(vector, "append")


@pytest.mark.slow
class TestVectorPerformance:
    """Test performance characteristics of vectors."""

    def test_large_array_append_retrieval(self, idaklu_module):
        """Test performance with large arrays."""
        import time
        
        vector = idaklu_module.VectorNdArray()
        large_array = np.random.rand(1000, 100).astype(np.float64)
        
        start_time = time.time()
        vector.append(large_array)
        append_time = time.time() - start_time
        assert append_time < 1.0
        
        start_time = time.time()
        retrieved = vector[0]
        retrieval_time = time.time() - start_time
        assert retrieval_time < 1.0
        assert retrieved.shape == large_array.shape

    def test_many_small_arrays(self, idaklu_module):
        """Test performance with many small arrays."""
        import time
        
        vector = idaklu_module.VectorNdArray()
        num_arrays = 1000
        
        start_time = time.time()
        for i in range(num_arrays):
            arr = np.random.rand(10).astype(np.float64)
            vector.append(arr)
        total_time = time.time() - start_time
        
        assert total_time < 5.0
        assert len(vector) == num_arrays
        
        # Test random access
        start_time = time.time()
        for _ in range(100):
            idx = np.random.randint(0, num_arrays)
            _ = vector[idx]
        access_time = time.time() - start_time
        assert access_time < 1.0

    def test_repeated_operations_stability(self, idaklu_module):
        """Test repeated operations for stability."""
        vector = idaklu_module.VectorNdArray()
        test_array = np.array([1.0, 2.0, 3.0])
        
        for i in range(1000):
            vector.append(test_array.copy())
            
            if i % 100 == 0:
                retrieved = vector[i]
                np.testing.assert_array_equal(retrieved, test_array)
        
        assert len(vector) == 1000

