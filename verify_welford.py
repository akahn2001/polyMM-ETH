"""
Verify that Welford's streaming algorithm matches numpy exactly.
"""
import time
import random
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_state import WelfordZScore


def test_accuracy():
    """Test that Welford matches numpy for mean/std calculation."""
    print("=" * 60)
    print("ACCURACY TEST: Welford vs NumPy")
    print("=" * 60)

    # Create calculator with same params as production
    calc = WelfordZScore(lookback_seconds=600, min_samples=50, min_std=0.10, maxlen=500)

    # Generate realistic spread data
    spreads = []
    base_time = time.time() - 600  # Start 10 min ago

    for i in range(300):
        ts = base_time + i * 2  # 2 seconds apart
        spread = random.gauss(0, 5)  # Mean 0, std ~5
        spreads.append((ts, spread))
        calc.update(spread, ts)

    # Compare Welford vs numpy on the same window
    now = spreads[-1][0]
    cutoff = now - 600
    recent = [s for ts, s in spreads if ts >= cutoff]

    np_mean = np.mean(recent)
    np_std = np.std(recent)

    welford_mean = calc.mean
    welford_std = calc.std()

    print(f"\nSamples in window: {len(recent)} (Welford n={calc.n})")
    print(f"\nMean:")
    print(f"  NumPy:   {np_mean:.10f}")
    print(f"  Welford: {welford_mean:.10f}")
    print(f"  Diff:    {abs(np_mean - welford_mean):.2e}")

    print(f"\nStd:")
    print(f"  NumPy:   {np_std:.10f}")
    print(f"  Welford: {welford_std:.10f}")
    print(f"  Diff:    {abs(np_std - welford_std):.2e}")

    # Test z-score
    test_spread = 10.0  # Test value
    np_zscore = (test_spread - np_mean) / np_std if np_std > 0.1 else 0.0
    welford_zscore, _ = calc.get_zscore(test_spread)

    print(f"\nZ-score for spread={test_spread}:")
    print(f"  NumPy:   {np_zscore:.10f}")
    print(f"  Welford: {welford_zscore:.10f}")
    print(f"  Diff:    {abs(np_zscore - welford_zscore):.2e}")

    # Check if differences are acceptable (< 1e-10)
    mean_ok = abs(np_mean - welford_mean) < 1e-9
    std_ok = abs(np_std - welford_std) < 1e-9
    zscore_ok = abs(np_zscore - welford_zscore) < 1e-9

    print(f"\n{'PASS' if mean_ok else 'FAIL'}: Mean matches")
    print(f"{'PASS' if std_ok else 'FAIL'}: Std matches")
    print(f"{'PASS' if zscore_ok else 'FAIL'}: Z-score matches")

    return mean_ok and std_ok and zscore_ok


def test_sliding_window():
    """Test that the sliding window eviction works correctly."""
    print("\n" + "=" * 60)
    print("SLIDING WINDOW TEST")
    print("=" * 60)

    calc = WelfordZScore(lookback_seconds=10, min_samples=5, min_std=0.01, maxlen=100)

    base_time = time.time()

    # Add 20 values over 20 seconds
    all_values = []
    for i in range(20):
        ts = base_time + i
        val = float(i)
        all_values.append((ts, val))
        calc.update(val, ts)

    # Values in window: ts >= (last_ts - lookback) = (base_time + 19) - 10 = base_time + 9
    # So timestamps 9, 10, 11, ..., 19 = 11 values (this matches original numpy behavior)
    last_ts = base_time + 19
    cutoff = last_ts - 10
    expected_in_window = [v for ts, v in all_values if ts >= cutoff]

    print(f"\nTotal values added: 20")
    print(f"Cutoff: base_time + {cutoff - base_time:.0f}")
    print(f"Values in window: {len(expected_in_window)}")
    print(f"Welford n: {calc.n}")
    print(f"Welford history len: {len(calc.history)}")

    # Check stats match
    np_mean = np.mean(expected_in_window)
    np_std = np.std(expected_in_window)

    print(f"\nExpected values in window: {expected_in_window}")
    print(f"NumPy mean: {np_mean:.4f}, Welford mean: {calc.mean:.4f}")
    print(f"NumPy std:  {np_std:.4f}, Welford std:  {calc.std():.4f}")

    n_ok = calc.n == len(expected_in_window)
    mean_ok = abs(np_mean - calc.mean) < 1e-9
    std_ok = abs(np_std - calc.std()) < 1e-9

    print(f"\n{'PASS' if n_ok else 'FAIL'}: Count matches ({calc.n} == {len(expected_in_window)})")
    print(f"{'PASS' if mean_ok else 'FAIL'}: Mean matches")
    print(f"{'PASS' if std_ok else 'FAIL'}: Std matches")

    return n_ok and mean_ok and std_ok


def test_maxlen_eviction():
    """Test that maxlen eviction properly updates stats (critical bug test)."""
    print("\n" + "=" * 60)
    print("MAXLEN EVICTION TEST (critical)")
    print("=" * 60)

    # Small maxlen to force auto-eviction, long lookback so time eviction doesn't trigger
    calc = WelfordZScore(lookback_seconds=1000, min_samples=3, min_std=0.01, maxlen=5)

    base_time = time.time()

    # Add 10 values - maxlen=5 means oldest 5 get evicted
    for i in range(10):
        calc.update(float(i), base_time + i)

    # Should only have values 5,6,7,8,9 in stats
    expected = [5.0, 6.0, 7.0, 8.0, 9.0]

    print(f"\nAdded values 0-9, maxlen=5")
    print(f"Expected in stats: {expected}")
    print(f"Welford n: {calc.n}")
    print(f"Welford history len: {len(calc.history)}")
    print(f"Welford mean: {calc.mean:.4f}, Expected: {np.mean(expected):.4f}")
    print(f"Welford std: {calc.std():.4f}, Expected: {np.std(expected):.4f}")

    n_ok = calc.n == 5
    mean_ok = abs(calc.mean - np.mean(expected)) < 1e-9
    std_ok = abs(calc.std() - np.std(expected)) < 1e-9

    print(f"\n{'PASS' if n_ok else 'FAIL'}: Count is 5 (got {calc.n})")
    print(f"{'PASS' if mean_ok else 'FAIL'}: Mean matches")
    print(f"{'PASS' if std_ok else 'FAIL'}: Std matches")

    return n_ok and mean_ok and std_ok


def test_speed():
    """Benchmark Welford vs numpy."""
    print("\n" + "=" * 60)
    print("SPEED TEST")
    print("=" * 60)

    iterations = 1000

    # Setup - pre-populate with 200 samples
    calc = WelfordZScore(lookback_seconds=600, min_samples=200, min_std=0.10, maxlen=500)
    history = []
    base_time = time.time() - 600

    for i in range(200):
        ts = base_time + i * 3
        spread = random.gauss(0, 5)
        history.append((ts, spread))
        calc.update(spread, ts)

    # Benchmark numpy approach
    start = time.perf_counter()
    for _ in range(iterations):
        now = time.time()
        cutoff = now - 600
        recent = [s for ts, s in history if ts >= cutoff]
        mean = np.mean(recent)
        std = np.std(recent)
        if std > 0.1:
            zscore = (5.0 - mean) / std
    numpy_time = (time.perf_counter() - start) * 1000 / iterations

    # Benchmark Welford approach
    start = time.perf_counter()
    for _ in range(iterations):
        spread = random.gauss(0, 5)
        calc.update(spread, time.time())
        zscore, std = calc.get_zscore(spread)
    welford_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nNumPy:   {numpy_time*1000:.1f} μs per call")
    print(f"Welford: {welford_time*1000:.1f} μs per call")
    print(f"Speedup: {numpy_time/welford_time:.1f}x")


if __name__ == "__main__":
    acc_ok = test_accuracy()
    window_ok = test_sliding_window()
    maxlen_ok = test_maxlen_eviction()
    test_speed()

    print("\n" + "=" * 60)
    if acc_ok and window_ok and maxlen_ok:
        print("ALL TESTS PASSED - Welford matches numpy exactly")
    else:
        print("SOME TESTS FAILED - Check output above")
    print("=" * 60)
