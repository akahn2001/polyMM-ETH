import asyncio
import json
import socket
import websockets
from datetime import datetime, timezone
from util import update_fair_value_for_market
from trading import perform_trade

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

RTDS_URL = "wss://ws-live-data.polymarket.com"

# Build the subscription payload for BTC/USD from Chainlink
SUBSCRIBE_MSG = {
    "action": "subscribe",
    "subscriptions": [
        {
            "topic": "crypto_prices_chainlink",
            "type": "*",  # all message types for this topic
            # NOTE: filters must be a JSON-encoded string per docs
            "filters": "{\"symbol\":\"btc/usd\"}"
        }
    ],
}


async def ping_loop(ws, interval_sec: int = 5):
    """
    Send periodic PING messages to keep the RTDS connection alive.
    Docs say you should send PING every ~5 seconds. :contentReference[oaicite:3]{index=3}
    """
    while True:
        try:
            await ws.send("PING")
        except Exception as e:
            print("Ping failed, stopping ping loop:", e)
            return
        await asyncio.sleep(interval_sec)


async def stream_btc_usd():
    print(f"[RTDS] Connecting to {RTDS_URL}...")
    while True:  # Reconnection loop
        try:
            print("[RTDS] Attempting websocket connection...")
            async with websockets.connect(RTDS_URL, ping_interval=None, open_timeout=10) as ws:
                # Disable Nagle's algorithm for lower latency
                _set_tcp_nodelay(ws)

                print("[RTDS] Websocket connected, sending subscription...")
                # Subscribe to BTC/USD Chainlink crypto price stream
                await ws.send(json.dumps(SUBSCRIBE_MSG))
                print("[RTDS] Connected and subscribed to crypto_prices_chainlink for btc/usd")

                # Start background ping task
                asyncio.create_task(ping_loop(ws))

                # Listen for messages forever
                while True:
                    msg = await ws.recv()

                    # RTDS messages are JSON
                    try:
                        data = json.loads(msg)
                        #print(data)
                        try:
                            timestamp = data["payload"]["data"][0]["timestamp"]
                            mid_price = data["payload"]["data"][0]["value"]
                            global_state.mid_price = mid_price
                            global_state.timestamp = timestamp

                            # Mark RTDS as connected on first successful price
                            if not global_state.rtds_connected:
                                print(f"[RTDS] First price received: ${mid_price:.2f}")
                                global_state.rtds_connected = True

                            # Update price blend Kalman filter with RTDS observation
                            if hasattr(global_state, 'price_blend_filter') and global_state.price_blend_filter is not None:
                                global_state.price_blend_filter.update_rtds(mid_price)
                                global_state.blended_price = global_state.price_blend_filter.get_blended_price()

                            for market_id in global_state.btc_markets:
                                update_fair_value_for_market(market_id)
                                asyncio.create_task(perform_trade(market_id))

                        except:
                            timestamp = data["payload"]["timestamp"]
                            mid_price = data["payload"]["value"]
                            global_state.mid_price = mid_price
                            global_state.timestamp = timestamp

                            # Mark RTDS as connected on first successful price
                            if not global_state.rtds_connected:
                                print(f"[RTDS] First price received: ${mid_price:.2f}")
                                global_state.rtds_connected = True

                            # Update price blend Kalman filter with RTDS observation
                            if hasattr(global_state, 'price_blend_filter') and global_state.price_blend_filter is not None:
                                global_state.price_blend_filter.update_rtds(mid_price)
                                global_state.blended_price = global_state.price_blend_filter.get_blended_price()

                            for market_id in global_state.btc_markets:
                                update_fair_value_for_market(market_id)
                                asyncio.create_task(perform_trade(market_id))
                    except json.JSONDecodeError:
                        # Might be "PONG" or some non-JSON message
                        continue

        except Exception as e:
            print(f"[RTDS] Connection error: {e}")
            global_state.rtds_connected = False
            print("[RTDS] Reconnecting in 5 seconds...")
            await asyncio.sleep(5)





if __name__ == "__main__":
    asyncio.run(stream_btc_usd())