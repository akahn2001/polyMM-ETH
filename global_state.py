import threading
import pandas as pd
import json
import ast
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field

USER_OWNER_ID = "36e9b72d-fb6b-151f-0ad7-869f32584268"

# Shadow trading flag
trading_enabled = False  # Scheduler enables this after configuring first market, DO NOT TOGGLE THIS MANUALLY
dry_run = False  # Set to False to enable live trading, DO TOGGLE THIS MANUALLY

# Bot start timestamp (for uptime tracking)
bot_start_ts = None

@dataclass
class MarketPosition:
    net_yes: float = 0.0         # +10 means long 10 YES, -5 means short 5 YES
    vwap_yes: float = 0.0        # volume-weighted avg price of YES position
    realized_pnl: float = 0.0    # optional, can fill in later

positions_by_market: dict[str, MarketPosition] = {}

working_orders_by_market: dict[str, dict[str, dict]] = {}
order_to_quote_side: dict[str, tuple[str, str]] = {}
# All trade IDs we've already processed so we don't double-count fills
processed_trade_ids = set()
# Track how much of each order we've already counted as filled
filled_size_by_order = {}

last_order_time: dict[tuple[str, str], float] = {}


# ============ Market Data ============

# List of all tokens being tracked (current active market)
all_tokens = []

# All tokens to subscribe to (all upcoming markets)
all_subscription_tokens = []

# Currently active market ID for trading
active_market_id = None

# Flag to signal websocket should reconnect (e.g., after CSV reload with new tokens)
websocket_reconnect_needed = False

# RTDS connection status
rtds_connected = False

# Mapping between tokens in the same market (YES->NO, NO->YES)
REVERSE_TOKENS = {} # maps YES to NO

# Order book data for all markets
all_data = {}
condition_to_token_id = {}
token_to_condition_id = {}
open_orders = {}

net_position = 0
strike = 88088.64
exp = datetime(2025, 12, 25, 15, 0, tzinfo=ZoneInfo("America/New_York"))

client = None

timestamp = None
mid_price = None

# Binance price stream data (BTCUSDT)
binance_mid_price = None
binance_mid_ts = None
binance_mid_bid = None
binance_mid_ask = None

# Price blending (Kalman filter combining Binance + RTDS)
price_blend_filter = None
blended_price = None

vol_filters = {}
fair_vol = {}

# RTDS-based fair values
fair_value = {}
binary_delta = {}

# Binance-based fair values (parallel to RTDS)
binance_fair_value = {}
binance_binary_delta = {}

btc_markets = set()

markouts = []

# USDT/USD exchange rate (from Kraken)
usdtusd = .999425

# Binance price history for momentum calculation (deque for O(1) append, auto-evicts old)
binance_price_history = deque(maxlen=500)  # (timestamp, price_usd) tuples