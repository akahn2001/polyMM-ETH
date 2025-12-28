import asyncio
import json
import socket
import time
import websockets
from datetime import datetime
from zoneinfo import ZoneInfo

import global_state


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


from util import update_binance_fair_value_for_market, update_fair_value_for_market, bs_binary_call, update_realized_vol
from trading import perform_trade, MIN_ORDER_INTERVAL, cancel_order_async, EARLY_CANCEL_OPTION_MOVE

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"

# Track last processed Coinbase price to avoid unnecessary recalcs
_last_coinbase_mid_usd = None


def _update_coinbase_theos(coinbase_mid_usd: float):
    """
    Internal callback to update Coinbase-based fair values for all BTC markets.

    Called on each Coinbase price tick. Updates theo for each market and triggers trading.
    Coinbase is native USD, so no conversion needed.
    """
    global _last_coinbase_mid_usd

    if global_state is None or update_fair_value_for_market is None:
        return

    # Store price history for momentum calculation (deque auto-evicts old entries)
    now = time.time()
    global_state.coinbase_price_history.append((now, coinbase_mid_usd))

    # EARLY CANCEL: If option price would move significantly, cancel the VULNERABLE side immediately
    # Coinbase UP → cancel ask (we'd sell too cheap)
    # Coinbase DOWN → cancel bid (we'd buy too high)
    if _last_coinbase_mid_usd is not None:
        price_move = coinbase_mid_usd - _last_coinbase_mid_usd

        # Calculate option price move using BS model
        option_move = 0.0

        # Use appropriate spot price based on configuration
        if global_state.USE_COINBASE_PRICE:
            S_current = coinbase_mid_usd  # Using pure Coinbase
        else:
            S_current = getattr(global_state, 'blended_price', None)  # Using blend

        exp = getattr(global_state, 'exp', None)
        strike = getattr(global_state, 'strike', None)

        if S_current is not None and exp is not None and strike is not None and price_move != 0:
            # Get vol from first market (they should all be similar)
            sigma = None
            fair_vol = getattr(global_state, 'fair_vol', {})
            if fair_vol:
                sigma = next(iter(fair_vol.values()), None)

            if sigma is not None:
                try:
                    now_et = datetime.now(ZoneInfo("America/New_York"))
                    T = (exp - now_et).total_seconds() / (60 * 60 * 24 * 365)
                    if T > 0:
                        current_option = bs_binary_call(S_current, strike, T, 0.0, sigma, 0.0, 1.0)
                        shifted_option = bs_binary_call(S_current + price_move, strike, T, 0.0, sigma, 0.0, 1.0)
                        option_move = abs(shifted_option - current_option)
                except Exception:
                    pass  # Fall back to not canceling

        if option_move >= EARLY_CANCEL_OPTION_MOVE:
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
    if _last_coinbase_mid_usd is not None and abs(coinbase_mid_usd - _last_coinbase_mid_usd) < 0.01:
        return  # Price unchanged, skip recalculation

    _last_coinbase_mid_usd = coinbase_mid_usd

    # Update realized vol estimates (5m and 15m) - use Coinbase history if enabled
    update_realized_vol()

    # Update fair value for all BTC markets and trigger trading
    if hasattr(global_state, 'btc_markets'):
        for market_id in global_state.btc_markets:
            try:
                # Update main theo (uses Coinbase if USE_COINBASE_PRICE=True, else blend)
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


async def stream_coinbase_btcusd_mid(on_mid=None, *, verbose=False):
    """
    Streams Coinbase BTC-USD best bid/ask (ticker channel) and computes mid = (bid+ask)/2.

    on_mid: optional callback(mid: float, bid: float, ask: float, ts: float)
    If global_state exists, will also update global_state.coinbase_mid_price and global_state.coinbase_mid_ts.

    Also automatically updates Coinbase-based fair values for all BTC markets.
    """
    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                COINBASE_WS,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1024,
            ) as ws:
                # Disable Nagle's algorithm for lower latency
                _set_tcp_nodelay(ws)

                backoff = 1.0
                if verbose:
                    print("[COINBASE] connected")

                # Subscribe to BTC-USD ticker channel
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["ticker"]
                }
                await ws.send(json.dumps(subscribe_msg))

                async for msg in ws:
                    data = json.loads(msg)

                    msg_type = data.get("type")

                    # Skip subscription confirmation messages
                    if msg_type in ("subscriptions", "heartbeat"):
                        continue

                    # Process ticker updates
                    if msg_type == "ticker":
                        # Ticker includes best_bid and best_ask
                        try:
                            bid = float(data.get("best_bid", 0))
                            ask = float(data.get("best_ask", 0))

                            if bid > 0 and ask > 0:
                                mid = 0.5 * (bid + ask)
                                ts = time.time()

                                if global_state is not None:
                                    # Store Coinbase prices
                                    global_state.coinbase_mid_price = mid
                                    global_state.coinbase_mid_ts = ts
                                    global_state.coinbase_mid_bid = bid
                                    global_state.coinbase_mid_ask = ask

                                # Update Coinbase-based fair values for all markets
                                _update_coinbase_theos(mid)

                                if on_mid is not None:
                                    on_mid(mid, bid, ask, ts)

                                if verbose:
                                    print(f"[COINBASE] bid={bid:.2f} ask={ask:.2f} mid={mid:.2f}")
                        except (ValueError, TypeError, KeyError):
                            pass  # Skip malformed messages

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            if verbose:
                print(f"[COINBASE] error: {e} (reconnecting in {backoff:.1f}s)")
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 1.7)


# Example run
if __name__ == "__main__":
    asyncio.run(stream_coinbase_btcusd_mid(verbose=True))
