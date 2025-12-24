import asyncio
import json
import time
import websockets
import requests

import global_state
from util import update_binance_fair_value_for_market, update_fair_value_for_market
from trading import perform_trade, MIN_ORDER_INTERVAL, cancel_order_async

# Threshold for early cancellation (before full recalculation)
EARLY_CANCEL_THRESHOLD = 3.0  # $3 move triggers immediate cancel

BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
BINANCE_TICKER_API = "https://api.binance.com/api/v3/ticker/bookTicker"
KRAKEN_API = "https://api.kraken.com/0/public/Ticker"


def get_binance_btcusdt_mid(verbose=False):
    """
    Query Binance REST API once to get current BTCUSDT mid price.

    Returns:
        float: BTCUSDT mid price, or None if query fails
    """
    try:
        response = requests.get(
            BINANCE_TICKER_API,
            params={'symbol': 'BTCUSDT'},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        bid = float(data['bidPrice'])
        ask = float(data['askPrice'])
        mid = (bid + ask) / 2.0

        if verbose:
            print(f"[Binance API] BTCUSDT: bid={bid:.2f}, ask={ask:.2f}, mid={mid:.2f}")

        return mid

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[Binance API] Error querying Binance ticker API: {e}")
        return None
    except (KeyError, ValueError) as e:
        if verbose:
            print(f"[Binance API] Error parsing Binance response: {e}")
        return None


def get_usdt_usd_rate(verbose=False):
    """
    Query Kraken API once to get the real USDT/USD exchange rate (fiat vs stablecoin).

    Returns the mid price: (bid + ask) / 2

    If rate = 0.9998: 1 USDT = $0.9998 USD (USDT at slight discount)
    If rate = 1.0002: 1 USDT = $1.0002 USD (USDT at slight premium)

    Returns:
        float: USDT/USD exchange rate, or 1.0 if query fails
    """
    try:
        # Query Kraken's USDT/USD pair (stablecoin vs fiat)
        response = requests.get(
            KRAKEN_API,
            params={'pair': 'USDTUSD'},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        # Check for Kraken API errors
        if data.get('error') and len(data['error']) > 0:
            raise ValueError(f"Kraken API error: {data['error']}")

        # Extract bid/ask from Kraken's response format
        # Response: {"result": {"USDTZUSD": {"b": ["0.9998", ...], "a": ["1.0002", ...]}}}
        ticker_data = data['result']['USDTZUSD']
        bid = float(ticker_data['b'][0])  # Best bid price
        ask = float(ticker_data['a'][0])  # Best ask price
        mid = (bid + ask) / 2.0

        if verbose:
            print(f"[USDT/USD Rate] Kraken USDT/USD: bid={bid:.6f}, ask={ask:.6f}, mid={mid:.6f}")
            if mid > 1.0:
                premium_pct = (mid - 1.0) * 100
                print(f"  USDT at {premium_pct:.4f}% premium (1 USDT ≈ ${mid:.6f})")
            elif mid < 1.0:
                discount_pct = (1.0 - mid) * 100
                print(f"  USDT at {discount_pct:.4f}% discount (1 USDT ≈ ${mid:.6f})")
            else:
                print(f"  USDT at parity (1 USDT ≈ $1.00)")

        return mid

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[USDT/USD Rate] Error querying Kraken API: {e}")
            print(f"  Defaulting to 1.0 (assuming USDT = USD)")
        return 1.0
    except (KeyError, ValueError, IndexError) as e:
        if verbose:
            print(f"[USDT/USD Rate] Error parsing Kraken response: {e}")
            print(f"  Defaulting to 1.0 (assuming USDT = USD)")
        return 1.0


# Track last processed Binance price to avoid unnecessary recalcs
_last_binance_mid_usd = None


def _update_binance_theos(binance_mid_usdt: float):
    """
    Internal callback to update Binance-based fair values for all BTC markets.

    Called on each Binance price tick. Adjusts BTCUSDT to BTCUSD using the
    global USDT/USD exchange rate, then calculates theo for each market.
    """
    global _last_binance_mid_usd

    if global_state is None or update_binance_fair_value_for_market is None:
        return

    # Get USDT/USD conversion rate
    usdtusd = getattr(global_state, 'usdtusd', 1.0)

    # Convert BTCUSDT -> BTCUSD
    binance_mid_usd = binance_mid_usdt * usdtusd

    # Store price history for momentum calculation (deque auto-evicts old entries)
    now = time.time()
    global_state.binance_price_history.append((now, binance_mid_usd))

    # EARLY CANCEL: If price moved significantly, cancel the VULNERABLE side immediately
    # Binance UP → cancel ask (we'd sell too cheap)
    # Binance DOWN → cancel bid (we'd buy too high)
    if _last_binance_mid_usd is not None:
        price_move = binance_mid_usd - _last_binance_mid_usd
        if abs(price_move) >= EARLY_CANCEL_THRESHOLD:
            vulnerable_side = "ask" if price_move > 0 else "bid"
            wom = getattr(global_state, "working_orders_by_market", {})
            for market_id, per_mkt in wom.items():
                if not isinstance(per_mkt, dict):
                    continue
                entry = per_mkt.get(vulnerable_side)
                if isinstance(entry, dict) and entry.get("id"):
                    try:
                        asyncio.create_task(cancel_order_async(entry["id"]))
                        per_mkt[vulnerable_side] = None  # Clear locally
                    except RuntimeError:
                        pass  # No event loop

    # Check if price actually changed (avoid redundant calculations)
    if _last_binance_mid_usd is not None and abs(binance_mid_usd - _last_binance_mid_usd) < 0.01:
        return  # Price unchanged, skip recalculation

    _last_binance_mid_usd = binance_mid_usd

    # Update price blend Kalman filter
    if hasattr(global_state, 'price_blend_filter') and global_state.price_blend_filter is not None:
        global_state.price_blend_filter.update_binance(binance_mid_usd)
        global_state.blended_price = global_state.price_blend_filter.get_blended_price()

    # Update fair value for all BTC markets using Binance price
    if hasattr(global_state, 'btc_markets'):
        for market_id in global_state.btc_markets:
            try:
                # Update Binance-specific theo
                update_binance_fair_value_for_market(market_id, binance_mid_usd)

                # Update main theo (uses blended price) and trigger trading
                if update_fair_value_for_market is not None:
                    update_fair_value_for_market(market_id)

                # Fire trading logic
                if perform_trade is not None:
                    try:
                        asyncio.create_task(perform_trade(market_id))
                    except RuntimeError:
                        # No event loop running (shouldn't happen but be safe)
                        pass

            except Exception as e:
                # Silently skip markets that don't have vol yet, etc.
                pass


async def stream_binance_btcusdt_mid(on_mid=None, *, verbose=False):
    """
    Streams Binance BTCUSDT best bid/ask (bookTicker) and computes mid = (bid+ask)/2.

    on_mid: optional callback(mid: float, bid: float, ask: float, ts: float)
    If global_state exists, will also update global_state.mid_price and global_state.mid_ts.

    Also automatically updates Binance-based fair values for all BTC markets.
    """
    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                BINANCE_WS,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1024,
            ) as ws:
                backoff = 1.0
                if verbose:
                    print("[BINANCE] connected")

                async for msg in ws:
                    data = json.loads(msg)

                    # bookTicker fields:
                    # b = best bid price, B = best bid qty, a = best ask price, A = best ask qty
                    bid = float(data["b"])
                    ask = float(data["a"])
                    mid = 0.5 * (bid + ask)
                    ts = time.time()

                    if global_state is not None:
                        # Store Binance prices separately (don't overwrite RTDS mid_price)
                        global_state.binance_mid_price = mid
                        global_state.binance_mid_ts = ts
                        global_state.binance_mid_bid = bid
                        global_state.binance_mid_ask = ask

                    # Update Binance-based fair values for all markets
                    _update_binance_theos(mid)

                    if on_mid is not None:
                        on_mid(mid, bid, ask, ts)

                    if verbose:
                        print(f"[BINANCE] bid={bid:.2f} ask={ask:.2f} mid={mid:.2f}")
                        pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            if verbose:
                print(f"[BINANCE] error: {e} (reconnecting in {backoff:.1f}s)")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.7)

# Example run
if __name__ == "__main__":
    # Test USDT/USD rate query
    print("Querying USDT/USD exchange rate...")
    rate = get_usdt_usd_rate(verbose=True)
    print(f"\nReturned rate: {rate}\n")

    # Stream BTC prices
    asyncio.run(stream_binance_btcusdt_mid(verbose=True))
