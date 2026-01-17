"""
Test that Rust math functions match Python implementations.
Run after building: python test_rust_math.py
"""

import sys
import os

# Add parent directory to path to import util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import (
    bs_binary_call as py_bs_binary_call,
    bs_binary_call_delta as py_bs_binary_call_delta,
    bs_binary_call_implied_vol_closed as py_bs_binary_call_implied_vol_closed,
)

try:
    from rust_math import (
        bs_binary_call as rust_bs_binary_call,
        bs_binary_call_delta as rust_bs_binary_call_delta,
        bs_binary_call_implied_vol_closed as rust_bs_binary_call_implied_vol_closed,
    )
except ImportError:
    print("ERROR: rust_math module not found!")
    print("Build it first with: cd rust_math && maturin develop --release")
    sys.exit(1)

import time

def test_accuracy():
    """Test that Rust results match Python within tolerance."""
    print("=" * 60)
    print("ACCURACY TEST")
    print("=" * 60)

    # Test parameters
    S = 97500.0      # BTC spot price
    K = 97000.0      # Strike
    T = 0.0001712    # ~15 min in years
    sigma = 0.45     # 45% implied vol
    r = 0.0
    q = 0.0
    payoff = 1.0

    # Test bs_binary_call
    py_price = py_bs_binary_call(S, K, T, r, sigma, q, payoff)
    rust_price = rust_bs_binary_call(S, K, T, r, sigma, q, payoff)
    diff = abs(py_price - rust_price)
    print(f"\nbs_binary_call:")
    print(f"  Python: {py_price:.10f}")
    print(f"  Rust:   {rust_price:.10f}")
    print(f"  Diff:   {diff:.2e} {'PASS' if diff < 1e-9 else 'FAIL'}")

    # Test bs_binary_call_delta
    py_delta = py_bs_binary_call_delta(S, K, T, r, sigma, q, payoff)
    rust_delta = rust_bs_binary_call_delta(S, K, T, r, sigma, q, payoff)
    diff = abs(py_delta - rust_delta)
    print(f"\nbs_binary_call_delta:")
    print(f"  Python: {py_delta:.10f}")
    print(f"  Rust:   {rust_delta:.10f}")
    print(f"  Diff:   {diff:.2e} {'PASS' if diff < 1e-9 else 'FAIL'}")

    # Test bs_binary_call_implied_vol_closed
    option_price = 0.55
    py_iv = py_bs_binary_call_implied_vol_closed(option_price, S, K, T, r, q, payoff)
    rust_iv = rust_bs_binary_call_implied_vol_closed(option_price, S, K, T, r, q, payoff)
    diff = abs(py_iv - rust_iv)
    print(f"\nbs_binary_call_implied_vol_closed:")
    print(f"  Python: {py_iv:.10f}")
    print(f"  Rust:   {rust_iv:.10f}")
    print(f"  Diff:   {diff:.2e} {'PASS' if diff < 1e-6 else 'FAIL'}")  # ppf has ~1e-9 accuracy


def test_speed():
    """Benchmark Python vs Rust speed."""
    print("\n" + "=" * 60)
    print("SPEED TEST (10,000 iterations)")
    print("=" * 60)

    S = 97500.0
    K = 97000.0
    T = 0.0001712
    sigma = 0.45
    r = 0.0
    q = 0.0
    payoff = 1.0
    option_price = 0.55

    iterations = 10000

    # bs_binary_call
    start = time.perf_counter()
    for _ in range(iterations):
        py_bs_binary_call(S, K, T, r, sigma, q, payoff)
    py_time = (time.perf_counter() - start) * 1000 / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        rust_bs_binary_call(S, K, T, r, sigma, q, payoff)
    rust_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nbs_binary_call:")
    print(f"  Python: {py_time*1000:.2f} μs")
    print(f"  Rust:   {rust_time*1000:.2f} μs")
    print(f"  Speedup: {py_time/rust_time:.1f}x")

    # bs_binary_call_delta
    start = time.perf_counter()
    for _ in range(iterations):
        py_bs_binary_call_delta(S, K, T, r, sigma, q, payoff)
    py_time = (time.perf_counter() - start) * 1000 / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        rust_bs_binary_call_delta(S, K, T, r, sigma, q, payoff)
    rust_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nbs_binary_call_delta:")
    print(f"  Python: {py_time*1000:.2f} μs")
    print(f"  Rust:   {rust_time*1000:.2f} μs")
    print(f"  Speedup: {py_time/rust_time:.1f}x")

    # bs_binary_call_implied_vol_closed
    start = time.perf_counter()
    for _ in range(iterations):
        py_bs_binary_call_implied_vol_closed(option_price, S, K, T, r, q, payoff)
    py_time = (time.perf_counter() - start) * 1000 / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        rust_bs_binary_call_implied_vol_closed(option_price, S, K, T, r, q, payoff)
    rust_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nbs_binary_call_implied_vol_closed:")
    print(f"  Python: {py_time*1000:.2f} μs")
    print(f"  Rust:   {rust_time*1000:.2f} μs")
    print(f"  Speedup: {py_time/rust_time:.1f}x")


if __name__ == "__main__":
    test_accuracy()
    test_speed()
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
