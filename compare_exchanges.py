"""
Compare latency between Binance and Kraken from your VPS.
Tests both REST API and WebSocket connections.

Usage:
    python compare_exchanges.py
"""

import asyncio
import time
import statistics
import json
import requests
from requests.adapters import HTTPAdapter
import websockets

# Connection pooling for fair comparison
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
_session.mount('https://', _adapter)
_session.mount('http://', _adapter)

# Endpoints
BINANCE_REST = "https://api.binance.com/api/v3/ticker/bookTicker"
KRAKEN_REST = "https://api.kraken.com/0/public/Ticker"
BITSTAMP_REST = "https://www.bitstamp.net/api/v2/ticker/btcusd/"

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
KRAKEN_WS = "wss://ws.kraken.com"

NUM_REST_TESTS = 20
NUM_WS_MESSAGES = 10


def test_rest_latency(name: str, url: str, params: dict = None) -> list:
    """Test REST API latency with connection reuse."""
    latencies = []

    # Warmup request (establishes connection)
    try:
        _session.get(url, params=params, timeout=10)
    except:
        pass

    for i in range(NUM_REST_TESTS):
        try:
            start = time.perf_counter()
            resp = _session.get(url, params=params, timeout=10)
            end = time.perf_counter()

            if resp.status_code == 200:
                latencies.append((end - start) * 1000)
        except Exception as e:
            if i == 0:
                print(f"  {name} error: {e}")

    return latencies


async def test_binance_ws() -> tuple:
    """Test Binance WebSocket - connection time and message latency."""
    connect_times = []
    message_times = []

    for _ in range(3):
        try:
            start = time.perf_counter()
            async with websockets.connect(BINANCE_WS, ping_interval=None) as ws:
                connect_time = (time.perf_counter() - start) * 1000
                connect_times.append(connect_time)

                # Measure time to receive messages
                for _ in range(NUM_WS_MESSAGES):
                    msg_start = time.perf_counter()
                    await asyncio.wait_for(ws.recv(), timeout=5)
                    message_times.append((time.perf_counter() - msg_start) * 1000)

        except Exception as e:
            print(f"  Binance WS error: {e}")

    return connect_times, message_times


async def test_kraken_ws() -> tuple:
    """Test Kraken WebSocket - connection time and message latency."""
    connect_times = []
    message_times = []

    for _ in range(3):
        try:
            start = time.perf_counter()
            async with websockets.connect(KRAKEN_WS, ping_interval=None) as ws:
                connect_time = (time.perf_counter() - start) * 1000
                connect_times.append(connect_time)

                # Subscribe to BTC/USD ticker
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "ticker"}
                }
                await ws.send(json.dumps(subscribe_msg))

                # Wait for subscription confirmation then measure ticker updates
                confirmed = False
                for _ in range(NUM_WS_MESSAGES + 5):
                    msg_start = time.perf_counter()
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)

                    # Skip system/subscription messages
                    if isinstance(data, dict):
                        continue

                    if isinstance(data, list) and len(data) > 1:
                        message_times.append((time.perf_counter() - msg_start) * 1000)
                        if len(message_times) >= NUM_WS_MESSAGES:
                            break

        except Exception as e:
            print(f"  Kraken WS error: {e}")

    return connect_times, message_times


def print_stats(name: str, latencies: list):
    """Print latency statistics."""
    if not latencies:
        print(f"  {name}: No data")
        return

    print(f"  {name}:")
    print(f"    Min: {min(latencies):.1f}ms")
    print(f"    Avg: {statistics.mean(latencies):.1f}ms")
    print(f"    Max: {max(latencies):.1f}ms")
    if len(latencies) > 1:
        print(f"    StdDev: {statistics.stdev(latencies):.1f}ms")


async def main():
    print("=" * 60)
    print("Exchange Latency Comparison")
    print("=" * 60)

    # REST API tests
    print("\n[REST API - with connection reuse]")

    binance_rest = test_rest_latency("Binance", BINANCE_REST, {"symbol": "BTCUSDT"})
    print_stats("Binance REST", binance_rest)

    kraken_rest = test_rest_latency("Kraken", KRAKEN_REST, {"pair": "XBTUSD"})
    print_stats("Kraken REST", kraken_rest)

    bitstamp_rest = test_rest_latency("Bitstamp", BITSTAMP_REST)
    print_stats("Bitstamp REST", bitstamp_rest)

    # WebSocket tests
    print("\n[WebSocket Connections]")

    binance_conn, binance_msg = await test_binance_ws()
    print_stats("Binance WS Connect", binance_conn)
    print_stats("Binance WS Message", binance_msg)

    kraken_conn, kraken_msg = await test_kraken_ws()
    print_stats("Kraken WS Connect", kraken_conn)
    print_stats("Kraken WS Message", kraken_msg)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (median values)")
    print("=" * 60)

    results = []
    if binance_rest:
        med = statistics.median(binance_rest)
        results.append(("Binance REST", med))
        print(f"  Binance REST:   {med:.0f}ms")
    if kraken_rest:
        med = statistics.median(kraken_rest)
        results.append(("Kraken REST", med))
        print(f"  Kraken REST:    {med:.0f}ms")
    if bitstamp_rest:
        med = statistics.median(bitstamp_rest)
        results.append(("Bitstamp REST", med))
        print(f"  Bitstamp REST:  {med:.0f}ms")

    if binance_msg:
        print(f"  Binance WS msg: {statistics.median(binance_msg):.0f}ms avg between ticks")
    if kraken_msg:
        print(f"  Kraken WS msg:  {statistics.median(kraken_msg):.0f}ms avg between ticks")

    # Winner
    if results:
        winner = min(results, key=lambda x: x[1])
        print(f"\n  Fastest REST: {winner[0]} ({winner[1]:.0f}ms)")


if __name__ == "__main__":
    asyncio.run(main())
