"""
Latency Benchmark Script
========================
Measures execution time of hot-path functions to identify Rust porting candidates.

Run: python benchmark_latency.py
"""

import time
import statistics
import numpy as np
from util import bs_binary_call, bs_binary_call_delta, bs_binary_call_implied_vol_closed
from kalman_filter import VolKalman1D


def benchmark(func, args=(), iterations=10000, warmup=1000):
    """
    Run function N times, return timing stats in microseconds.

    Returns dict with mean, median, p95, p99, max latencies.
    """
    # Warmup to avoid cold-start effects
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000)  # convert ns to μs

    times_sorted = sorted(times)
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'p95': times_sorted[int(0.95 * len(times))],
        'p99': times_sorted[int(0.99 * len(times))],
        'max': max(times),
        'min': min(times),
    }


def print_result(name, result, calls_per_sec=200):
    """Print benchmark result with formatting."""
    print(f"\n{name}:")
    print(f"  mean:   {result['mean']:>8.2f} μs")
    print(f"  median: {result['median']:>8.2f} μs")
    print(f"  p95:    {result['p95']:>8.2f} μs")
    print(f"  p99:    {result['p99']:>8.2f} μs")
    print(f"  max:    {result['max']:>8.2f} μs")

    # Estimated overhead at given call rate
    ms_per_sec = result['mean'] * calls_per_sec / 1000
    print(f"  @ {calls_per_sec} calls/sec: {ms_per_sec:.3f} ms/sec CPU time")


def main():
    print("=" * 65)
    print("LATENCY BENCHMARK - Hot Path Functions (Python)")
    print("=" * 65)
    print(f"Iterations: 10,000 per function (with 1,000 warmup)")

    # Realistic test parameters
    S = 97500.0      # BTC spot price
    K = 97000.0      # Strike (slightly ITM)
    T = 0.0001712    # ~15 min in years (15/(365*24*60))
    sigma = 0.45     # 45% implied vol
    r = 0.0          # Risk-free rate
    q = 0.0          # Dividend yield

    results = {}

    # =========================================================================
    # TIER 1: Per-tick functions (highest priority)
    # =========================================================================
    print("\n" + "-" * 65)
    print("TIER 1: Per-tick functions")
    print("-" * 65)

    # 1. bs_binary_call - core pricing
    result = benchmark(bs_binary_call, (S, K, T, r, sigma, q, 1.0))
    results['bs_binary_call'] = result
    print_result("bs_binary_call", result, calls_per_sec=400)

    # 2. bs_binary_call_delta
    result = benchmark(bs_binary_call_delta, (S, K, T, r, sigma, q, 1.0))
    results['bs_binary_call_delta'] = result
    print_result("bs_binary_call_delta", result, calls_per_sec=200)

    # 3. numpy z-score calculation (simulating coinbase_rtds_zscore)
    history = np.random.randn(200).astype(np.float64)  # 200 samples

    def zscore_numpy():
        mean = np.mean(history)
        std = np.std(history)
        return (history[-1] - mean) / std if std > 0 else 0.0

    result = benchmark(zscore_numpy, iterations=10000)
    results['numpy_zscore_200'] = result
    print_result("numpy z-score (200 samples)", result, calls_per_sec=50)

    # 4. Kalman filter tick
    kf = VolKalman1D(x0=0.45)

    def kalman_tick():
        return kf.process_tick(0.44, 0.46)  # bid_vol, ask_vol

    result = benchmark(kalman_tick, iterations=10000)
    results['kalman_tick'] = result
    print_result("VolKalman1D.process_tick", result, calls_per_sec=100)

    # =========================================================================
    # TIER 2: Per-trade decision functions
    # =========================================================================
    print("\n" + "-" * 65)
    print("TIER 2: Per-trade decision functions")
    print("-" * 65)

    # 5. Implied vol closed form (uses scipy.stats.norm.ppf)
    option_price = 0.55  # Observed market price

    def iv_closed():
        try:
            return bs_binary_call_implied_vol_closed(option_price, S, K, T, r, q, 1.0)
        except ValueError:
            return 0.0  # Handle edge cases

    result = benchmark(iv_closed, iterations=10000)
    results['iv_closed_form'] = result
    print_result("bs_binary_call_implied_vol_closed", result, calls_per_sec=50)

    # 6. Pure Python math.erf (for comparison)
    import math

    def pure_python_bs():
        if T <= 0 or sigma <= 0:
            return 0.0
        d2 = (math.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        phi = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        return math.exp(-r * T) * phi

    result = benchmark(pure_python_bs, iterations=10000)
    results['pure_python_bs'] = result
    print_result("pure Python BS (no function call)", result, calls_per_sec=400)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    # Sort by mean latency
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)

    print("\nRanked by mean latency (highest first):")
    print("-" * 45)
    total_estimated = 0
    for name, res in sorted_results:
        calls = 200 if 'bs_binary' in name else 50
        ms_per_sec = res['mean'] * calls / 1000
        total_estimated += ms_per_sec
        print(f"  {name:35s} {res['mean']:>7.2f} μs")

    print("-" * 45)
    print(f"\nEstimated total hot-path overhead: ~{total_estimated:.2f} ms/sec")
    print(f"(Assuming typical call frequencies during active trading)")

    print("\n" + "=" * 65)
    print("RUST PORTING PRIORITY")
    print("=" * 65)
    print("\n1. bs_binary_call + bs_binary_call_delta (most frequent)")
    print("2. bs_binary_call_implied_vol_closed (scipy.ppf overhead)")
    print("3. numpy z-score → incremental Rust calculation")
    print("4. Kalman filter (pure numeric, easy port)")

    print("\nExpected Rust speedup: 10-50x per function")
    print("Expected p99 improvement: Eliminate GC spikes (50μs → <1μs)")


if __name__ == "__main__":
    main()
