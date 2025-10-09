"""Performance and memory tests for pybammsolvers.

These tests check performance characteristics and memory usage.
All tests in this module are marked as slow.
"""

import pytest
import numpy as np
import time
import gc


@pytest.mark.slow
class TestMemoryUsage:
    """Test memory usage and potential leaks."""

    def test_memory_cleanup_with_psutil(self, idaklu_module):
        """Test that memory is properly cleaned up using psutil."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and destroy many objects
        for _ in range(100):
            vector = idaklu_module.VectorNdArray()
            for _ in range(10):
                arr = np.random.rand(100).astype(np.float64)
                vector.append(arr)
            del vector

        # Force garbage collection
        gc.collect()

        # Check memory usage after cleanup
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Allow for some memory increase but not excessive
        max_allowed_increase = 100 * 1024   # 100 KB

        assert memory_increase < max_allowed_increase, (
            f"Memory increased by {memory_increase / 1024:.1f} KB, "
            f"which exceeds limit of {max_allowed_increase / 1024:.1f} KB"
        )


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Benchmark performance of key operations."""

    def test_append_performance_scaling(self, idaklu_module):
        """Test that append performance scales reasonably."""
        vector = idaklu_module.VectorNdArray()
        
        # Test append time doesn't degrade significantly
        times = []
        for batch in range(5):
            start_time = time.time()
            for _ in range(100):
                arr = np.random.rand(50).astype(np.float64)
                vector.append(arr)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Later batches shouldn't be much slower than early ones
        # Allow for some variation but check it's not exponential
        assert times[-1] < times[0] * 1.2, "Append performance degraded significantly"

    def test_access_performance_scaling(self, idaklu_module):
        """Test that access performance doesn't degrade with size."""
        vector = idaklu_module.VectorNdArray()
        
        # Build up a reasonably large vector
        for i in range(1000):
            vector.append(np.array([i], dtype=np.float64))
        
        # Test access at different points
        access_times = []
        for idx in [0, 250, 500, 750, 999]:
            start_time = time.time()
            for _ in range(100):
                _ = vector[idx]
            elapsed = time.time() - start_time
            access_times.append(elapsed)
        
        # Access time should be relatively constant (not index-dependent)
        avg_time = np.mean(access_times)
        for t in access_times:
            assert t < avg_time * 1.5, f"Access time {t} significantly differs from average {avg_time}"

    def test_large_array_copy_performance(self, idaklu_module):
        """Test performance of copying large arrays."""
        vector = idaklu_module.VectorNdArray()
        
        # Create a large array
        large_array = np.random.rand(10000, 100).astype(np.float64)
        
        start_time = time.time()
        vector.append(large_array)
        append_time = time.time() - start_time
        
        # Should be fast (< 1 second for 1M elements)
        assert append_time < 1.0, f"Large array append took {append_time:.3f}s"
        
        start_time = time.time()
        retrieved = vector[0]
        retrieval_time = time.time() - start_time
        
        assert retrieval_time < 1.0, f"Large array retrieval took {retrieval_time:.3f}s"
        assert retrieved.shape == large_array.shape
