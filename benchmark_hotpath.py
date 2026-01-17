"""
Benchmark hot-path functions to measure optimization opportunity.
"""
import time
import math
import random
import numpy as np
from collections import deque

# Simulate realistic data sizes
ZSCORE_HISTORY_SIZE = 200      # 10 min at ~0.3 samples/sec
REALIZED_VOL_HISTORY_SIZE = 900  # 15 min at 1 sample/sec

# Generate fake price history
def generate_price_history(size, base_price=97000):
    """Generate realistic BTC price history."""
    history = deque(maxlen=size)
    now = time.time()
    price = base_price
    for i in range(size):
        price += random.gauss(0, 10)  # Random walk
        history.append((now - size + i, price))
    return history

# Generate fake spread history
def generate_spread_history(size):
    """Generate realistic Coinbase-RTDS spread history."""
    history = deque(maxlen=size)
    now = time.time()
    for i in range(size):
        spread = random.gauss(0, 5)  # Mean 0, std 5
        history.append((now - size + i, spread))
    return history


# ============================================================
# CURRENT IMPLEMENTATION: compute_realized_vol (from util.py)
# ============================================================
def compute_realized_vol_current(history, lookback_seconds):
    """Current O(n) implementation."""
    if len(history) < 10:
        return None

    now = time.time()
    cutoff = now - lookback_seconds

    # O(n) filter
    recent = [(t, p) for t, p in history if t >= cutoff]

    if len(recent) < 10:
        return None

    # O(n) log returns
    log_returns = []
    for i in range(1, len(recent)):
        t1, p1 = recent[i-1]
        t2, p2 = recent[i]
        if p1 > 0 and p2 > 0:
            log_returns.append(math.log(p2 / p1))

    if len(log_returns) < 5:
        return None

    # O(n) numpy std
    std_return = np.std(log_returns)

    # Annualize (simplified)
    avg_dt = lookback_seconds / len(recent)
    if avg_dt > 0:
        annual_factor = math.sqrt(365 * 24 * 3600 / avg_dt)
        return std_return * annual_factor
    return None


# ============================================================
# CURRENT IMPLEMENTATION: _update_coinbase_rtds_zscore
# ============================================================
def compute_zscore_current(spread_history, new_spread):
    """Current O(n) implementation."""
    now = time.time()
    LOOKBACK_SECONDS = 10 * 60
    MIN_SAMPLES = 200
    MIN_STD_DEV = 0.10

    cutoff = now - LOOKBACK_SECONDS

    # O(n) filter
    recent_spreads = [s for (ts, s) in spread_history if ts >= cutoff]

    if len(recent_spreads) >= MIN_SAMPLES:
        # O(n) numpy operations
        mean = np.mean(recent_spreads)
        std = np.std(recent_spreads)

        if std > MIN_STD_DEV:
            return (new_spread - mean) / std

    return 0.0


# ============================================================
# WELFORD'S ALGORITHM: O(1) streaming statistics
# ============================================================
class WelfordStats:
    """O(1) running mean/variance using Welford's online algorithm."""
    __slots__ = ('n', 'mean', 'M2')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def remove(self, x):
        """Remove old value (for sliding window)."""
        if self.n <= 1:
            self.n = 0
            self.mean = 0.0
            self.M2 = 0.0
            return

        self.n -= 1
        delta = x - self.mean
        self.mean -= delta / self.n
        delta2 = x - self.mean
        self.M2 -= delta * delta2
        self.M2 = max(0, self.M2)  # Numerical stability

    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(self.variance())


class StreamingRealizedVol:
    """O(1) realized vol using Welford's algorithm with sliding window."""

    def __init__(self, lookback_seconds=900, max_samples=1000):
        self.lookback = lookback_seconds
        self.log_returns = deque(maxlen=max_samples)  # (timestamp, log_return)
        self.stats = WelfordStats()
        self.last_price = None
        self.last_time = None

    def update(self, price, ts=None):
        ts = ts or time.time()

        # Compute log return
        if self.last_price is not None and self.last_price > 0 and price > 0:
            log_ret = math.log(price / self.last_price)

            # Add new return
            self.log_returns.append((ts, log_ret))
            self.stats.update(log_ret)

        self.last_price = price
        self.last_time = ts

        # Evict old returns (outside lookback window)
        cutoff = ts - self.lookback
        while self.log_returns and self.log_returns[0][0] < cutoff:
            old_ts, old_ret = self.log_returns.popleft()
            self.stats.remove(old_ret)

    def get_annualized_vol(self):
        if self.stats.n < 10:
            return None

        std_return = self.stats.std()

        # Estimate avg dt from recent data
        if len(self.log_returns) >= 2:
            total_time = self.log_returns[-1][0] - self.log_returns[0][0]
            avg_dt = total_time / len(self.log_returns)
            if avg_dt > 0:
                annual_factor = math.sqrt(365 * 24 * 3600 / avg_dt)
                return std_return * annual_factor

        return None


class StreamingZScore:
    """O(1) z-score using Welford's algorithm."""

    def __init__(self, lookback_seconds=600, min_samples=200, min_std=0.10):
        self.lookback = lookback_seconds
        self.min_samples = min_samples
        self.min_std = min_std
        self.spreads = deque()  # (timestamp, spread)
        self.stats = WelfordStats()

    def update(self, spread, ts=None):
        ts = ts or time.time()

        # Add new spread
        self.spreads.append((ts, spread))
        self.stats.update(spread)

        # Evict old spreads
        cutoff = ts - self.lookback
        while self.spreads and self.spreads[0][0] < cutoff:
            old_ts, old_spread = self.spreads.popleft()
            self.stats.remove(old_spread)

    def get_zscore(self, current_spread):
        if self.stats.n < self.min_samples:
            return 0.0

        std = self.stats.std()
        if std < self.min_std:
            return 0.0

        return (current_spread - self.stats.mean) / std


# ============================================================
# BENCHMARKS
# ============================================================
def benchmark_realized_vol():
    print("=" * 60)
    print("BENCHMARK: compute_realized_vol()")
    print("=" * 60)

    # Setup
    history = generate_price_history(REALIZED_VOL_HISTORY_SIZE)
    iterations = 1000

    # Benchmark current (O(n))
    start = time.perf_counter()
    for _ in range(iterations):
        compute_realized_vol_current(history, 900)
    current_time = (time.perf_counter() - start) * 1000 / iterations

    # Benchmark streaming (O(1))
    streaming = StreamingRealizedVol(lookback_seconds=900)
    # Warm up with history
    for ts, price in history:
        streaming.update(price, ts)

    start = time.perf_counter()
    for _ in range(iterations):
        # Simulate new price tick
        streaming.update(97000 + random.gauss(0, 10))
        streaming.get_annualized_vol()
    streaming_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nHistory size: {len(history)} samples")
    print(f"Current (O(n)):   {current_time*1000:.1f} μs per call")
    print(f"Streaming (O(1)): {streaming_time*1000:.1f} μs per call")
    print(f"Speedup: {current_time/streaming_time:.1f}x")
    print(f"Time saved per call: {(current_time - streaming_time)*1000:.1f} μs")

    # At 10 ticks/sec
    ticks_per_sec = 10
    saved_per_sec = (current_time - streaming_time) * ticks_per_sec
    print(f"\nAt {ticks_per_sec} ticks/sec:")
    print(f"  Time saved: {saved_per_sec:.2f} ms/sec")
    print(f"  CPU freed: {saved_per_sec/10:.1f}%")


def benchmark_zscore():
    print("\n" + "=" * 60)
    print("BENCHMARK: _update_coinbase_rtds_zscore()")
    print("=" * 60)

    # Setup
    history = generate_spread_history(ZSCORE_HISTORY_SIZE)
    iterations = 1000

    # Benchmark current (O(n))
    start = time.perf_counter()
    for _ in range(iterations):
        compute_zscore_current(history, random.gauss(0, 5))
    current_time = (time.perf_counter() - start) * 1000 / iterations

    # Benchmark streaming (O(1))
    streaming = StreamingZScore(lookback_seconds=600, min_samples=200)
    # Warm up with history
    for ts, spread in history:
        streaming.update(spread, ts)

    start = time.perf_counter()
    for _ in range(iterations):
        new_spread = random.gauss(0, 5)
        streaming.update(new_spread)
        streaming.get_zscore(new_spread)
    streaming_time = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nHistory size: {len(history)} samples")
    print(f"Current (O(n)):   {current_time*1000:.1f} μs per call")
    print(f"Streaming (O(1)): {streaming_time*1000:.1f} μs per call")
    print(f"Speedup: {current_time/streaming_time:.1f}x")
    print(f"Time saved per call: {(current_time - streaming_time)*1000:.1f} μs")

    # At 10 ticks/sec
    ticks_per_sec = 10
    saved_per_sec = (current_time - streaming_time) * ticks_per_sec
    print(f"\nAt {ticks_per_sec} ticks/sec:")
    print(f"  Time saved: {saved_per_sec:.2f} ms/sec")
    print(f"  CPU freed: {saved_per_sec/10:.1f}%")


def benchmark_total_impact():
    print("\n" + "=" * 60)
    print("TOTAL HOT PATH IMPACT")
    print("=" * 60)

    # Setup both
    price_history = generate_price_history(REALIZED_VOL_HISTORY_SIZE)
    spread_history = generate_spread_history(ZSCORE_HISTORY_SIZE)

    iterations = 1000

    # Current combined
    start = time.perf_counter()
    for _ in range(iterations):
        compute_realized_vol_current(price_history, 900)
        compute_realized_vol_current(price_history, 300)  # 5min too
        compute_zscore_current(spread_history, random.gauss(0, 5))
    current_total = (time.perf_counter() - start) * 1000 / iterations

    # Streaming combined
    rv_15m = StreamingRealizedVol(lookback_seconds=900)
    rv_5m = StreamingRealizedVol(lookback_seconds=300)
    zscore = StreamingZScore()

    for ts, price in price_history:
        rv_15m.update(price, ts)
        rv_5m.update(price, ts)
    for ts, spread in spread_history:
        zscore.update(spread, ts)

    start = time.perf_counter()
    for _ in range(iterations):
        price = 97000 + random.gauss(0, 10)
        spread = random.gauss(0, 5)
        rv_15m.update(price)
        rv_5m.update(price)
        rv_15m.get_annualized_vol()
        rv_5m.get_annualized_vol()
        zscore.update(spread)
        zscore.get_zscore(spread)
    streaming_total = (time.perf_counter() - start) * 1000 / iterations

    print(f"\nPer tick (all 3 calculations):")
    print(f"  Current:   {current_total*1000:.0f} μs")
    print(f"  Streaming: {streaming_total*1000:.0f} μs")
    print(f"  Speedup:   {current_total/streaming_total:.0f}x")

    saved_per_tick = current_total - streaming_total
    print(f"\nTime saved per tick: {saved_per_tick*1000:.0f} μs ({saved_per_tick:.3f} ms)")

    # Impact at different tick rates
    for ticks in [5, 10, 20]:
        saved_ms = saved_per_tick * ticks
        print(f"At {ticks} ticks/sec: save {saved_ms:.1f} ms/sec ({saved_ms/10:.1f}% CPU)")


if __name__ == "__main__":
    benchmark_realized_vol()
    benchmark_zscore()
    benchmark_total_impact()
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
