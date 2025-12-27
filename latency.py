"""
Polymarket API Latency Tester

Run this script from different regions/VPS providers to compare latency.
Tests both REST API and WebSocket connection times.

Usage:
    python latency_check.py

Deploy to different cloud regions (AWS, GCP, DigitalOcean) to find optimal location.
"""

import time
import asyncio
import statistics
import requests
from requests.adapters import HTTPAdapter
import websockets
import json
from datetime import datetime

# ============================================================
# HTTP CONNECTION POOLING - Reuse connections for accurate test
# ============================================================
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
_session.mount('https://', _adapter)
_session.mount('http://', _adapter)
requests.get = _session.get
requests.post = _session.post

# Polymarket endpoints
CLOB_REST_API = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API = "https://gamma-api.polymarket.com"

# Test parameters
NUM_REST_TESTS = 20
NUM_WS_TESTS = 5
TEST_TOKEN_ID = "54291121785752774138560938726921956522966432081615467614354174799263009645501"  # BTC market


def test_rest_latency(endpoint: str, path: str, params: dict = None) -> list[float]:
    """Test REST API latency with multiple requests."""
    url = f"{endpoint}{path}"
    latencies = []

    print(f"\nTesting REST: {url}")
    print(f"  Params: {params}")

    for i in range(NUM_REST_TESTS):
        try:
            start = time.perf_counter()
            response = requests.get(url, params=params, timeout=10)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            if i == 0:
                print(f"  Status: {response.status_code}")
        except Exception as e:
            print(f"  Error on request {i}: {e}")

    return latencies


async def test_ws_connection_latency() -> list[float]:
    """Test WebSocket connection establishment latency."""
    latencies = []

    print(f"\nTesting WebSocket connection: {CLOB_WS}")

    for i in range(NUM_WS_TESTS):
        try:
            start = time.perf_counter()
            async with websockets.connect(
                CLOB_WS,
                ping_interval=None,
                close_timeout=5,
            ) as ws:
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

                # Subscribe and measure first message latency
                subscribe_msg = {
                    "type": "subscribe",
                    "assets_ids": [TEST_TOKEN_ID],
                    "markets": []
                }

                msg_start = time.perf_counter()
                await ws.send(json.dumps(subscribe_msg))
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                msg_end = time.perf_counter()

                if i == 0:
                    print(f"  First message latency: {(msg_end - msg_start) * 1000:.1f}ms")

        except Exception as e:
            print(f"  Error on connection {i}: {e}")

    return latencies


async def test_ws_message_latency() -> list[float]:
    """Test WebSocket message round-trip latency by measuring time between messages."""
    latencies = []

    print(f"\nTesting WebSocket message stream latency...")

    try:
        async with websockets.connect(
            CLOB_WS,
            ping_interval=20,
            close_timeout=5,
        ) as ws:
            # Subscribe
            subscribe_msg = {
                "type": "subscribe",
                "assets_ids": [TEST_TOKEN_ID],
                "markets": []
            }
            await ws.send(json.dumps(subscribe_msg))

            # Measure time between messages (approximates server-to-client latency)
            last_time = None
            msg_count = 0

            async for msg in ws:
                now = time.perf_counter()
                if last_time is not None:
                    gap_ms = (now - last_time) * 1000
                    if gap_ms < 1000:  # Only count gaps < 1s (actual updates, not idle)
                        latencies.append(gap_ms)
                last_time = now
                msg_count += 1

                if msg_count >= 50 or len(latencies) >= 20:
                    break

    except Exception as e:
        print(f"  Error: {e}")

    return latencies


def print_stats(name: str, latencies: list[float]):
    """Print latency statistics."""
    if not latencies:
        print(f"\n{name}: No data")
        return

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    print(f"\n{name} ({n} samples):")
    print(f"  Min:    {min(latencies):.1f}ms")
    print(f"  Max:    {max(latencies):.1f}ms")
    print(f"  Avg:    {statistics.mean(latencies):.1f}ms")
    print(f"  Median: {statistics.median(latencies):.1f}ms")
    if n >= 20:
        p95_idx = int(n * 0.95)
        print(f"  P95:    {latencies_sorted[p95_idx]:.1f}ms")
    print(f"  StdDev: {statistics.stdev(latencies) if n > 1 else 0:.1f}ms")


async def main():
    print("=" * 60)
    print("Polymarket API Latency Test")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Test 1: CLOB REST API - Get order book
    book_latencies = test_rest_latency(
        CLOB_REST_API,
        "/book",
        {"token_id": TEST_TOKEN_ID}
    )
    print_stats("CLOB /book (order book)", book_latencies)

    # Test 2: CLOB REST API - Get markets
    markets_latencies = test_rest_latency(
        CLOB_REST_API,
        "/markets",
        {"limit": 1}
    )
    print_stats("CLOB /markets", markets_latencies)

    # Test 3: Gamma API
    gamma_latencies = test_rest_latency(
        GAMMA_API,
        "/markets",
        {"limit": 1}
    )
    print_stats("Gamma /markets", gamma_latencies)

    # Test 4: WebSocket connection
    ws_connect_latencies = await test_ws_connection_latency()
    print_stats("WebSocket connection", ws_connect_latencies)

    # Test 5: WebSocket message stream
    ws_msg_latencies = await test_ws_message_latency()
    print_stats("WebSocket message gaps", ws_msg_latencies)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if book_latencies:
        print(f"Order book fetch:     {statistics.median(book_latencies):.0f}ms median")
    if ws_connect_latencies:
        print(f"WebSocket connect:    {statistics.median(ws_connect_latencies):.0f}ms median")

    print("\nTo find optimal region, run this script from:")
    print("  - AWS: us-east-1, eu-west-1, ap-northeast-1")
    print("  - GCP: us-east1, europe-west1")
    print("  - DigitalOcean: NYC, AMS, SGP")
    print("\nLower latency = faster order placement/cancellation")


if __name__ == "__main__":
    asyncio.run(main())
