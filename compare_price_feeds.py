"""
Compare Binance vs Kraken BTC/USD price feeds.

Streams both exchanges and records prices every 0.20 seconds.
Dumps to CSV every minute for later analysis.

Usage:
    python compare_price_feeds.py

The CSV will show if Kraken leads Binance or vice versa.
"""

import asyncio
import json
import time
import csv
import os
import socket
from datetime import datetime
from zoneinfo import ZoneInfo
import websockets
import requests

# Websocket URLs
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
KRAKEN_SPOT_WS = "wss://ws.kraken.com"
KRAKEN_FUTURES_WS = "wss://futures.kraken.com/ws/v1"  # Has index price!
KRAKEN_USDT_API = "https://api.kraken.com/0/public/Ticker"
POLYMARKET_RTDS_WS = "wss://ws-live-data.polymarket.com"

# Polymarket RTDS subscription for BTC/USD Chainlink price
RTDS_SUBSCRIBE_MSG = {
    "action": "subscribe",
    "subscriptions": [
        {
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }
    ],
}

# Sampling config
SAMPLE_INTERVAL = 0.20  # seconds
DUMP_INTERVAL = 60  # seconds
OUTPUT_DIR = "price_comparison"

# Global state for prices
binance_mid_usdt = None
binance_mid_usd = None
binance_bid = None
binance_ask = None
binance_ts = None

kraken_mid_usd = None
kraken_bid = None
kraken_ask = None
kraken_ts = None

# Kraken Futures index price (aggregate of multiple exchanges)
kraken_index_usd = None
kraken_index_ts = None

# Polymarket RTDS Chainlink price
rtds_price_usd = None
rtds_ts = None

usdt_usd_rate = 1.0  # Updated periodically

# Price samples buffer
samples = []


def _set_tcp_nodelay(ws):
    """Disable Nagle's algorithm for lower latency."""
    try:
        transport = ws.transport
        if transport is not None:
            sock = transport.get_extra_info('socket')
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass


async def update_usdt_usd_rate():
    """Periodically update USDT/USD rate from Kraken."""
    global usdt_usd_rate

    while True:
        try:
            response = await asyncio.to_thread(
                requests.get,
                KRAKEN_USDT_API,
                params={'pair': 'USDTUSD'},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            if not data.get('error'):
                ticker_data = data['result']['USDTZUSD']
                bid = float(ticker_data['b'][0])
                ask = float(ticker_data['a'][0])
                usdt_usd_rate = (bid + ask) / 2.0
                print(f"[USDT/USD] Rate updated: {usdt_usd_rate:.6f}")
        except Exception as e:
            print(f"[USDT/USD] Error updating rate: {e}")

        await asyncio.sleep(60)  # Update every minute


async def stream_binance():
    """Stream Binance BTCUSDT and convert to USD."""
    global binance_mid_usdt, binance_mid_usd, binance_bid, binance_ask, binance_ts

    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                BINANCE_WS,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                _set_tcp_nodelay(ws)
                backoff = 1.0
                print("[BINANCE] Connected")

                async for msg in ws:
                    data = json.loads(msg)

                    binance_bid = float(data["b"])
                    binance_ask = float(data["a"])
                    binance_mid_usdt = 0.5 * (binance_bid + binance_ask)
                    binance_mid_usd = binance_mid_usdt * usdt_usd_rate
                    binance_ts = time.time()

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"[BINANCE] Error: {e}, reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)


async def stream_kraken_spot():
    """Stream Kraken XBT/USD spot (native USD pair)."""
    global kraken_mid_usd, kraken_bid, kraken_ask, kraken_ts

    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                KRAKEN_SPOT_WS,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                _set_tcp_nodelay(ws)
                backoff = 1.0
                print("[KRAKEN SPOT] Connected")

                # Subscribe to XBT/USD ticker
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "ticker"}
                }
                await ws.send(json.dumps(subscribe_msg))

                async for msg in ws:
                    data = json.loads(msg)

                    # Skip system messages (dicts with "event" key)
                    if isinstance(data, dict):
                        if data.get("event") == "subscriptionStatus":
                            print(f"[KRAKEN SPOT] Subscribed: {data.get('status')}")
                        continue

                    # Ticker updates are arrays: [channelID, tickerData, "ticker", "XBT/USD"]
                    if isinstance(data, list) and len(data) >= 2:
                        ticker = data[1]
                        if isinstance(ticker, dict):
                            # a = ask [price, wholeLotVolume, lotVolume]
                            # b = bid [price, wholeLotVolume, lotVolume]
                            try:
                                kraken_ask = float(ticker["a"][0])
                                kraken_bid = float(ticker["b"][0])
                                kraken_mid_usd = 0.5 * (kraken_bid + kraken_ask)
                                kraken_ts = time.time()
                            except (KeyError, IndexError, TypeError):
                                pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"[KRAKEN SPOT] Error: {e}, reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)


async def stream_kraken_index():
    """Stream Kraken Futures index price (aggregate of multiple exchanges)."""
    global kraken_index_usd, kraken_index_ts

    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                KRAKEN_FUTURES_WS,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                _set_tcp_nodelay(ws)
                backoff = 1.0
                print("[KRAKEN INDEX] Connected to Futures WS")

                # Subscribe to PI_XBTUSD ticker (perpetual inverse BTC/USD)
                subscribe_msg = {
                    "event": "subscribe",
                    "feed": "ticker",
                    "product_ids": ["PI_XBTUSD"]
                }
                await ws.send(json.dumps(subscribe_msg))

                async for msg in ws:
                    data = json.loads(msg)

                    # Look for ticker messages with index field
                    if isinstance(data, dict):
                        feed = data.get("feed")
                        if feed == "ticker" and "index" in data:
                            try:
                                kraken_index_usd = float(data["index"])
                                kraken_index_ts = time.time()
                            except (ValueError, TypeError):
                                pass
                        elif feed == "ticker_snapshot" and "index" in data:
                            # Initial snapshot
                            try:
                                kraken_index_usd = float(data["index"])
                                kraken_index_ts = time.time()
                                print(f"[KRAKEN INDEX] First index: ${kraken_index_usd:.2f}")
                            except (ValueError, TypeError):
                                pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"[KRAKEN INDEX] Error: {e}, reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)


async def rtds_ping_loop(ws):
    """Send periodic PING to keep RTDS connection alive."""
    while True:
        try:
            await ws.send("PING")
        except Exception:
            return
        await asyncio.sleep(5)


async def stream_rtds():
    """Stream Polymarket RTDS Chainlink BTC/USD price."""
    global rtds_price_usd, rtds_ts

    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                POLYMARKET_RTDS_WS,
                ping_interval=None,
                open_timeout=10,
            ) as ws:
                _set_tcp_nodelay(ws)
                backoff = 1.0
                print("[RTDS] Connected to Polymarket")

                # Subscribe to BTC/USD Chainlink price
                await ws.send(json.dumps(RTDS_SUBSCRIBE_MSG))

                # Start ping loop
                asyncio.create_task(rtds_ping_loop(ws))

                async for msg in ws:
                    try:
                        data = json.loads(msg)

                        # Try nested format first: payload.data[0].value
                        try:
                            price = data["payload"]["data"][0]["value"]
                            rtds_price_usd = float(price)
                            rtds_ts = time.time()
                            continue
                        except (KeyError, IndexError, TypeError):
                            pass

                        # Try flat format: payload.value
                        try:
                            price = data["payload"]["value"]
                            rtds_price_usd = float(price)
                            rtds_ts = time.time()
                        except (KeyError, TypeError):
                            pass

                    except json.JSONDecodeError:
                        # Might be "PONG" or other non-JSON
                        pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"[RTDS] Error: {e}, reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)


async def sample_prices():
    """Sample prices every SAMPLE_INTERVAL seconds."""
    global samples

    print(f"[SAMPLER] Starting, interval={SAMPLE_INTERVAL}s")

    while True:
        now = time.time()

        if binance_mid_usd is not None or kraken_mid_usd is not None or kraken_index_usd is not None or rtds_price_usd is not None:
            sample = {
                "timestamp": now,
                "datetime": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "binance_mid_usd": binance_mid_usd,
                "binance_bid": binance_bid,
                "binance_ask": binance_ask,
                "binance_age_ms": (now - binance_ts) * 1000 if binance_ts else None,
                "kraken_mid_usd": kraken_mid_usd,
                "kraken_bid": kraken_bid,
                "kraken_ask": kraken_ask,
                "kraken_spot_age_ms": (now - kraken_ts) * 1000 if kraken_ts else None,
                "kraken_index_usd": kraken_index_usd,
                "kraken_index_age_ms": (now - kraken_index_ts) * 1000 if kraken_index_ts else None,
                "rtds_price_usd": rtds_price_usd,
                "rtds_age_ms": (now - rtds_ts) * 1000 if rtds_ts else None,
                "usdt_usd_rate": usdt_usd_rate,
            }

            # Calculate spreads between exchanges (all vs Binance as baseline)
            if binance_mid_usd is not None and kraken_mid_usd is not None:
                sample["kraken_spot_minus_binance"] = kraken_mid_usd - binance_mid_usd
            else:
                sample["kraken_spot_minus_binance"] = None

            if binance_mid_usd is not None and kraken_index_usd is not None:
                sample["kraken_index_minus_binance"] = kraken_index_usd - binance_mid_usd
            else:
                sample["kraken_index_minus_binance"] = None

            if binance_mid_usd is not None and rtds_price_usd is not None:
                sample["rtds_minus_binance"] = rtds_price_usd - binance_mid_usd
            else:
                sample["rtds_minus_binance"] = None

            samples.append(sample)

        await asyncio.sleep(SAMPLE_INTERVAL)


async def dump_to_csv():
    """Dump samples to CSV every DUMP_INTERVAL seconds."""
    global samples

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate filename with start time
    start_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"price_comparison_{start_time}.csv")

    fieldnames = [
        "datetime", "timestamp",
        "binance_mid_usd", "binance_bid", "binance_ask", "binance_age_ms",
        "kraken_mid_usd", "kraken_bid", "kraken_ask", "kraken_spot_age_ms",
        "kraken_index_usd", "kraken_index_age_ms",
        "rtds_price_usd", "rtds_age_ms",
        "kraken_spot_minus_binance", "kraken_index_minus_binance", "rtds_minus_binance",
        "usdt_usd_rate"
    ]

    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"[CSV] Output file: {csv_path}")

    while True:
        await asyncio.sleep(DUMP_INTERVAL)

        if not samples:
            continue

        # Copy and clear samples
        to_write = samples.copy()
        samples = []

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(to_write)

        # Print summary
        if to_write:
            index_diffs = [s["kraken_index_minus_binance"] for s in to_write if s["kraken_index_minus_binance"] is not None]
            spot_diffs = [s["kraken_spot_minus_binance"] for s in to_write if s["kraken_spot_minus_binance"] is not None]
            rtds_diffs = [s["rtds_minus_binance"] for s in to_write if s["rtds_minus_binance"] is not None]

            summary_parts = [f"[CSV] Wrote {len(to_write)} samples."]

            if index_diffs:
                avg_idx = sum(index_diffs) / len(index_diffs)
                summary_parts.append(f"KrakenIdx: ${avg_idx:+.2f}")

            if rtds_diffs:
                avg_rtds = sum(rtds_diffs) / len(rtds_diffs)
                summary_parts.append(f"RTDS: ${avg_rtds:+.2f}")

            if spot_diffs:
                avg_spot = sum(spot_diffs) / len(spot_diffs)
                summary_parts.append(f"KrakenSpot: ${avg_spot:+.2f}")

            print(" | ".join(summary_parts))


async def status_printer():
    """Print current prices every 5 seconds."""
    await asyncio.sleep(3)  # Wait for connections

    while True:
        parts = []

        if binance_mid_usd is not None:
            parts.append(f"Binance: ${binance_mid_usd:.2f}")
        else:
            parts.append("Binance: --")

        if kraken_index_usd is not None:
            diff = (kraken_index_usd - binance_mid_usd) if binance_mid_usd else 0
            parts.append(f"KrkIdx: ${kraken_index_usd:.2f} ({diff:+.1f})")
        else:
            parts.append("KrkIdx: --")

        if rtds_price_usd is not None:
            diff = (rtds_price_usd - binance_mid_usd) if binance_mid_usd else 0
            parts.append(f"RTDS: ${rtds_price_usd:.2f} ({diff:+.1f})")
        else:
            parts.append("RTDS: --")

        parts.append(f"n={len(samples)}")

        print(f"[STATUS] " + " | ".join(parts))

        await asyncio.sleep(5)


async def main():
    print("=" * 60)
    print("BTC/USD Price Feed Comparison")
    print("  - Binance BTCUSDT (converted to USD)")
    print("  - Kraken Spot XBT/USD (bid/ask mid)")
    print("  - Kraken Futures Index (multi-exchange aggregate)")
    print("  - Polymarket RTDS (Chainlink oracle)")
    print(f"Sample interval: {SAMPLE_INTERVAL}s")
    print(f"CSV dump interval: {DUMP_INTERVAL}s")
    print("=" * 60)

    # Run all tasks concurrently
    await asyncio.gather(
        update_usdt_usd_rate(),
        stream_binance(),
        stream_kraken_spot(),
        stream_kraken_index(),
        stream_rtds(),
        sample_prices(),
        dump_to_csv(),
        status_printer(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user")
        # Final dump
        if samples:
            print(f"[EXIT] {len(samples)} samples not written (use Ctrl+C less aggressively)")
