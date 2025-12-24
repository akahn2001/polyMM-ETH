"""
Binance WebSocket Latency Tester

Tests latency to Binance BTCUSDT price stream.
Compare results from different regions to find optimal server location.

Usage:
    python binance_latency.py
"""

import asyncio
import time
import json
import statistics
import websockets
from datetime import datetime

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
NUM_MESSAGES = 50


async def test_ws_connection_latency():
    """Test WebSocket connection establishment latency."""
    latencies = []
    print(f"\nTesting WebSocket connection: {BINANCE_WS}")

    for i in range(5):
        try:
            start = time.perf_counter()
            async with websockets.connect(
                BINANCE_WS,
                ping_interval=None,
                close_timeout=5,
            ) as ws:
                # Wait for first message
                await asyncio.wait_for(ws.recv(), timeout=5)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
        except Exception as e:
            print(f"  Error on connection {i}: {e}")

    return latencies


async def test_message_rate():
    """Measure message arrival rate and gaps."""
    gaps = []
    timestamps = []

    print(f"\nMeasuring message rate ({NUM_MESSAGES} messages)...")

    try:
        async with websockets.connect(
            BINANCE_WS,
            ping_interval=20,
            close_timeout=5,
        ) as ws:
            last_time = None
            count = 0

            async for msg in ws:
                now = time.perf_counter()
                data = json.loads(msg)

                # Binance provides event time in milliseconds
                event_time_ms = data.get("E", 0)

                if last_time is not None:
                    gap_ms = (now - last_time) * 1000
                    gaps.append(gap_ms)

                timestamps.append({
                    "local_time": now,
                    "event_time_ms": event_time_ms,
                    "bid": float(data.get("b", 0)),
                    "ask": float(data.get("a", 0)),
                })

                last_time = now
                count += 1

                if count >= NUM_MESSAGES:
                    break

    except Exception as e:
        print(f"  Error: {e}")

    return gaps, timestamps


async def test_exchange_latency(timestamps):
    """
    Estimate latency by comparing local receive time to Binance event time.
    Note: Requires synchronized clocks for accurate measurement.
    """
    if not timestamps:
        return []

    latencies = []
    local_epoch = time.time()
    local_perf = time.perf_counter()

    for ts in timestamps:
        if ts["event_time_ms"] > 0:
            # Convert local perf_counter to epoch time
            local_epoch_at_receive = local_epoch + (ts["local_time"] - local_perf)
            local_ms = local_epoch_at_receive * 1000

            # Latency = local receive time - Binance event time
            latency = local_ms - ts["event_time_ms"]
            if 0 < latency < 5000:  # Sanity check: 0-5 seconds
                latencies.append(latency)

    return latencies


def print_stats(name: str, values: list[float]):
    """Print statistics."""
    if not values:
        print(f"\n{name}: No data")
        return

    values_sorted = sorted(values)
    n = len(values_sorted)

    print(f"\n{name} ({n} samples):")
    print(f"  Min:    {min(values):.1f}ms")
    print(f"  Max:    {max(values):.1f}ms")
    print(f"  Avg:    {statistics.mean(values):.1f}ms")
    print(f"  Median: {statistics.median(values):.1f}ms")
    if n >= 20:
        p95_idx = int(n * 0.95)
        print(f"  P95:    {values_sorted[p95_idx]:.1f}ms")
    if n > 1:
        print(f"  StdDev: {statistics.stdev(values):.1f}ms")


async def main():
    print("=" * 60)
    print("Binance WebSocket Latency Test")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Test 1: Connection latency
    connect_latencies = await test_ws_connection_latency()
    print_stats("WebSocket connection + first message", connect_latencies)

    # Test 2: Message gaps
    gaps, timestamps = await test_message_rate()
    print_stats("Message gaps (time between updates)", gaps)

    # Test 3: Exchange latency (clock sync dependent)
    exchange_latencies = await test_exchange_latency(timestamps)
    print_stats("Exchange latency (local - event time)", exchange_latencies)

    # Message rate
    if gaps:
        msgs_per_sec = 1000 / statistics.mean(gaps)
        print(f"\nMessage rate: {msgs_per_sec:.1f} messages/second")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if connect_latencies:
        print(f"Connection latency: {statistics.median(connect_latencies):.0f}ms median")
    if exchange_latencies:
        print(f"Exchange latency:   {statistics.median(exchange_latencies):.0f}ms median")
        print("\n  Note: Exchange latency requires synchronized clocks (NTP).")
        print("  If values seem off, your system clock may be drifting.")
    if gaps:
        print(f"Update frequency:   {msgs_per_sec:.1f} msgs/sec")

    print("\nLower latency = faster price signals = better edge")


if __name__ == "__main__":
    asyncio.run(main())
