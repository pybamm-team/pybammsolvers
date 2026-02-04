#!/usr/bin/env python3
"""ARM-specific benchmark script for comparing native ARM vs x86 performance.

This script runs a quick performance suite and reports timing data
with architecture information. It can be used to:
1. Compare ARM native vs emulated performance
2. Benchmark before/after ARM optimization
3. Verify ARM builds are performing correctly

Usage:
    python arm_benchmark.py [--iterations N]
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path


def get_system_info():
    """Collect system information for benchmarking context."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }


def time_function(func, iterations=5):
    """Time a function over multiple iterations and return statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "iterations": iterations,
        "times": times,
    }


def benchmark_spm_solve():
    """Benchmark SPM model solve."""
    import pybamm

    model = pybamm.lithium_ion.SPM()
    solver = pybamm.IDAKLUSolver()
    sim = pybamm.Simulation(model, solver=solver)
    sim.solve([0, 3600])


def benchmark_spme_solve():
    """Benchmark SPMe model solve."""
    import pybamm

    model = pybamm.lithium_ion.SPMe()
    solver = pybamm.IDAKLUSolver()
    sim = pybamm.Simulation(model, solver=solver)
    sim.solve([0, 3600])


def benchmark_dfn_solve():
    """Benchmark DFN model solve."""
    import pybamm

    model = pybamm.lithium_ion.DFN()
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    sim = pybamm.Simulation(model, solver=solver)
    sim.solve([0, 1800])


def benchmark_experiment():
    """Benchmark a simple experiment simulation."""
    import pybamm

    model = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C for 30 minutes",
            "Rest for 10 minutes",
            "Charge at 0.5C for 30 minutes",
        ]
    )
    solver = pybamm.IDAKLUSolver()
    sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
    sim.solve()


def run_benchmarks(iterations=5):
    """Run all benchmarks and return results."""
    # Import PyBaMM and pybammsolvers for version info
    try:
        import pybamm
        import pybammsolvers

        pybamm_version = pybamm.__version__
        pybammsolvers_version = pybammsolvers.__version__
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        sys.exit(1)

    benchmarks = {
        "spm_1hr_discharge": benchmark_spm_solve,
        "spme_1hr_discharge": benchmark_spme_solve,
        "dfn_30min_discharge": benchmark_dfn_solve,
        "simple_experiment": benchmark_experiment,
    }

    results = {
        "system_info": get_system_info(),
        "pybamm_version": pybamm_version,
        "pybammsolvers_version": pybammsolvers_version,
        "benchmarks": {},
    }

    print("=" * 60)
    print("PyBaMM ARM Benchmark Suite")
    print("=" * 60)
    print(f"\nSystem: {results['system_info']['platform']} {results['system_info']['architecture']}")
    print(f"Python: {results['system_info']['python_version']}")
    print(f"PyBaMM: {pybamm_version}")
    print(f"pybammsolvers: {pybammsolvers_version}")
    print(f"\nRunning {len(benchmarks)} benchmarks with {iterations} iterations each...\n")

    for name, func in benchmarks.items():
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            timing = time_function(func, iterations)
            results["benchmarks"][name] = timing
            print(f"mean={timing['mean']:.3f}s (min={timing['min']:.3f}s, max={timing['max']:.3f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results["benchmarks"][name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_time = sum(
        b.get("mean", 0) for b in results["benchmarks"].values() if "mean" in b
    )
    is_arm = results["system_info"]["architecture"] in ("arm64", "aarch64")
    arch_label = "ARM64 (native)" if is_arm else "x86_64"

    print(f"\nArchitecture: {arch_label}")
    print(f"Total benchmark time: {total_time:.3f}s")
    print(f"\nBenchmark results by test:")

    for name, timing in results["benchmarks"].items():
        if "mean" in timing:
            print(f"  {name}: {timing['mean']:.3f}s")
        else:
            print(f"  {name}: ERROR - {timing.get('error', 'unknown')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ARM-specific benchmarks for pybammsolvers"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5,
        help="Number of iterations per benchmark (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    results = run_benchmarks(iterations=args.iterations)

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Return non-zero if any benchmark failed
    failed = any("error" in b for b in results["benchmarks"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
