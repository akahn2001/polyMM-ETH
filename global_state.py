import threading
import asyncio
import math
import pandas as pd
import json
import ast
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field


class WelfordZScore:
    """
    O(1) streaming z-score calculator using Welford's online algorithm.
    Maintains a sliding time window and computes mean/std incrementally.
    """
    __slots__ = ('lookback_seconds', 'min_samples', 'min_std', 'history', 'n', 'mean', 'M2', '_maxlen')

    def __init__(self, lookback_seconds=600, min_samples=200, min_std=0.10, maxlen=500):
        self.lookback_seconds = lookback_seconds
        self.min_samples = min_samples
        self.min_std = min_std
        self._maxlen = maxlen
        self.history = deque(maxlen=maxlen)  # (timestamp, value)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared deviations from mean

    def _add(self, x):
        """Add a value to running statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def _remove(self, x):
        """Remove a value from running statistics."""
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
        # Numerical stability: M2 should never be negative
        if self.M2 < 0:
            self.M2 = 0.0

    def update(self, value, ts):
        """Add new value and evict old ones outside the lookback window."""
        # Handle maxlen auto-eviction: if at capacity, remove oldest from stats BEFORE append
        if self._maxlen is not None and len(self.history) >= self._maxlen:
            old_ts, old_val = self.history[0]  # This will be auto-evicted by append
            self._remove(old_val)

        # Add new value
        self.history.append((ts, value))
        self._add(value)

        # Evict old values outside lookback window
        cutoff = ts - self.lookback_seconds
        while self.history and self.history[0][0] < cutoff:
            old_ts, old_val = self.history.popleft()
            self._remove(old_val)

    def std(self):
        """Return current standard deviation."""
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / self.n)

    def get_zscore(self, current_value):
        """Calculate z-score for a value against current statistics."""
        if self.n < self.min_samples:
            return 0.0, 0.0  # (zscore, std)

        std = self.std()
        if std < self.min_std:
            return 0.0, std

        zscore = (current_value - self.mean) / std
        return zscore, std

USER_OWNER_ID = "36e9b72d-fb6b-151f-0ad7-869f32584268"

# Position check lock - prevents race condition where multiple concurrent tasks place orders
# Must be initialized in async context, not at module level
position_check_lock = None

# Shadow trading flag
trading_enabled = False  # Scheduler enables this after configuring first market, DO NOT TOGGLE THIS MANUALLY
dry_run = False  # Set to False to enable live trading, DO TOGGLE THIS MANUALLY

# Price source configuration
# Options: "COINBASE" = Coinbase + bias correction, "BLEND" = Kalman blend (Binance/RTDS), "RTDS" = pure RTDS
PRICE_SOURCE = "RTDS"

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
rtds_last_update_time = None  # Track when RTDS last updated (for staleness detection)
user_ws_last_event_time = None  # Track when user websocket last sent ORDER/TRADE event (for position staleness detection)

# Binance price stream data (BTCUSDT)
binance_mid_price = None
binance_mid_ts = None
binance_mid_bid = None
binance_mid_ask = None

# Coinbase price stream data (BTC/USD, native USD)
coinbase_mid_price = None
coinbase_mid_ts = None
coinbase_mid_bid = None
coinbase_mid_ask = None

# Coinbase bias correction (tracks systematic difference from RTDS)
coinbase_rtds_spread_history = deque(maxlen=1000)  # Track (timestamp, rtds - coinbase) tuples
coinbase_bias_correction = 2.0  # Dollars to add to Coinbase for theo (initial assumption: CB is $2 below RTDS)

# Coinbase-RTDS spread z-score tracking (for predictive edge when using RTDS)
coinbase_rtds_zscore_history = deque(maxlen=500)  # Track (timestamp, coinbase - rtds) for z-score - LEGACY, kept for compatibility
coinbase_rtds_zscore = 0.0  # Current z-score: positive = Coinbase high (RTDS will rise), negative = Coinbase low (RTDS will fall)
coinbase_rtds_zscore_calculator = WelfordZScore(lookback_seconds=600, min_samples=200, min_std=0.10, maxlen=500)  # O(1) streaming z-score

# Price blending (Kalman filter combining Binance + RTDS)
price_blend_filter = None
blended_price = None

vol_filters = {}
fair_vol = {}

# Realized vol estimates from Binance price history
realized_vol_5m = None   # 5-minute realized vol (annualized)
realized_vol_15m = None  # 15-minute realized vol (annualized)

# RTDS-based fair values
fair_value = {}
binary_delta = {}

# Binance-based fair values (parallel to RTDS)
binance_fair_value = {}
binance_binary_delta = {}

btc_markets = set()

markouts = deque(maxlen=2000)  # Bound memory - ~16 hours of fills

# USDT/USD exchange rate (from Kraken)
usdtusd = .999425

# Binance price history for momentum calculation (deque for O(1) append, auto-evicts old)
binance_price_history = deque(maxlen=500)  # (timestamp, price_usd) tuples

# Coinbase price history for momentum calculation (deque for O(1) append, auto-evicts old)
coinbase_price_history = deque(maxlen=500)  # (timestamp, price_usd) tuples