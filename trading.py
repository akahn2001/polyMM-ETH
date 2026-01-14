import global_state
import asyncio
import time
import math
import threading
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from markouts import record_fill
from global_state import MarketPosition
from py_clob_client.exceptions import PolyApiException
from py_clob_client.clob_types import OpenOrderParams
from util import bs_binary_call
from trading_config import (
    MAX_IMBALANCE_ADJUSTMENT,
    MAX_MOMENTUM_ADJUSTMENT,
    MAX_Z_SCORE_SKEW,
    MAX_TOTAL_SIGNAL_ADJUSTMENT,
    USE_BOOK_IMBALANCE,
    BOOK_IMBALANCE_LEVELS,
    Z_SCORE_CONFIDENCE_MIDPOINT,
    Z_SCORE_CONFIDENCE_STEEPNESS,
    AGGRESSIVE_MODE_ENABLED,
    AGGRESSIVE_Z_THRESHOLD,
    AGGRESSIVE_ZSKEW_THRESHOLD,
    AGGRESSIVE_MAX_TOTAL_ADJUSTMENT,
    AGGRESSIVE_MAX_Z_SCORE_SKEW,
    AGGRESSIVE_SIZE
)

# TODO: 12/17 WE STILL AREN'T CANCELLING ORDERS CORRECTLY- currently using print statements to debug the open_orders from api. vs. protected orders set logic
# TODO: My suspicion is that we are somehow protecting too many orders and this is leaving stale orders in the book

RECONCILE_INTERVAL = 0.50  # seconds (start here; tighten later if stable)
RECONCILE_MAX_CANCEL_BATCH = 50  # keep batches mo  dest to avoid Cloudflare pain

# Per-market locks to prevent concurrent perform_trade() calls from double-counting capacity
market_locks = {}  # {market_id: asyncio.Lock}
market_locks_mutex = threading.Lock()  # Protects market_locks dict creation

def tick_down(p: float, tick: float) -> float:
    return math.floor(p / tick + 1e-12) * tick

def tick_up(p: float, tick: float) -> float:
    return math.ceil(p / tick - 1e-12) * tick

def _oid(order) -> str | None:
    if isinstance(order, dict):
        return order.get("id") or order.get("order_id")
    return getattr(order, "id", None) or getattr(order, "order_id", None)

def _status(order) -> str:
    if isinstance(order, dict):
        return (order.get("status") or "").upper()
    return (getattr(order, "status", "") or "").upper()

async def fetch_open_orders_for_market(market_id: str):
    """
    Returns OpenOrder[] for this market.
    Uses Polymarket's 'get active orders' endpoint via py-clob-client.
    """
    params = OpenOrderParams(market=market_id)
    return await asyncio.to_thread(global_state.client.get_open_orders, market_id)

def get_protected_ids_for_market(market_id: str) -> set[str]:
    wo = getattr(global_state, "working_orders_by_market", {}).get(market_id, {})
    protected = set()
    for k in ("bid", "ask"):
        entry = wo.get(k)
        if isinstance(entry, dict) and entry.get("id"):
            protected.add(entry["id"])
    return protected

async def reconcile_market_orders(market_id: str):
    """
    1) Pull live open orders from API (filtered to market)
    2) Cancel anything not protected
    3) Repair wo[bid]/wo[ask] if the tracked IDs are no longer open
    """
    client = global_state.client
    if client is None:
        return

    wom = getattr(global_state, "working_orders_by_market", {})
    wo = wom.get(market_id)
    if wo is None:
        return  # not tracking this market yet

    try:
        open_orders = await fetch_open_orders_for_market(market_id)
    except Exception as e:
        print(f"[RECONCILE] get_orders failed for market={market_id}: {e}")
        return

    open_ids = set()
    cancel_ids = []

    # Read protected IDs inside lock to avoid race with manage_side()
    async with global_state.position_check_lock:
        protected = get_protected_ids_for_market(market_id)

    for o in (open_orders or []):
        oid = _oid(o)
        if not oid:
            continue

        st = _status(o)
        # Treat missing status as open; otherwise only live-ish
        if st and st not in ("LIVE", "OPEN", "PARTIALLY_FILLED", "MATCHED"):
            continue

        open_ids.add(oid)

        if oid not in protected:
            cancel_ids.append(oid)

    # Repair local tracking if our "bid"/"ask" order is not actually open anymore
    # Acquire lock to prevent race with manage_side() which also modifies wo[]
    # CRITICAL: Delay clearing filled orders to prevent position limit breaches
    # When order is missing from open_ids, it could be filled (TRADE event pending) or cancelled
    # Keep tracking for 1 second to ensure TRADE events are processed into filled_yes first
    async with global_state.position_check_lock:
        for side_key in ("bid", "ask"):
            entry = wo.get(side_key)
            if isinstance(entry, dict):
                oid = entry.get("id")
                if oid and oid not in open_ids:
                    # Order is no longer open - delay clearing to avoid race with TRADE event processing
                    last_reconcile_ts = entry.get("reconcile_cleared_ts", 0)
                    now_ts = time.time()

                    if last_reconcile_ts == 0:
                        # First reconcile where order is missing - mark timestamp but keep tracking
                        entry["reconcile_cleared_ts"] = now_ts
                        if VERBOSE:
                            print(f"[RECONCILE] Order {oid} not in open_ids, marking for cleanup (will clear in 1s)")
                    elif now_ts - last_reconcile_ts > 1.0:
                        # Order has been missing for >1s - safe to clear now (fill should be processed)
                        wo[side_key] = None
                        if VERBOSE:
                            print(f"[RECONCILE] Clearing stale order {oid} from working_orders (>1s since filled)")

    # Cancel extras in batches
    if cancel_ids:
        if VERBOSE:
            print(f"[RECONCILE] market={market_id} cancelling {len(cancel_ids)} non-protected orders")
        for i in range(0, len(cancel_ids), RECONCILE_MAX_CANCEL_BATCH):
            batch = cancel_ids[i:i + RECONCILE_MAX_CANCEL_BATCH]
            await cancel_many_orders(batch)

def get_all_protected_ids() -> set[str]:
    protected = set()
    wom = getattr(global_state, "working_orders_by_market", {})
    for _, per_mkt in wom.items():
        if not isinstance(per_mkt, dict):
            continue
        for side_key in ("bid", "ask"):
            entry = per_mkt.get(side_key)
            if isinstance(entry, dict) and entry.get("id"):
                protected.add(entry["id"])
    return protected

def _extract_id(o):
    if isinstance(o, dict):
        return o.get("id") or o.get("orderID") or o.get("order_id")
    return getattr(o, "id", None) or getattr(o, "order_id", None)

async def reconcile_loop_all():
    while True:
        t0 = time.time()
        try:
            protected = get_all_protected_ids()
            #print("PROTECTED IDS FOR DEBUG!!!: ", protected)
            # TODO: see if we are protecting too many orders
            open_orders = await asyncio.to_thread(global_state.client.get_open_orders_all)
            global_state.open_orders = open_orders  # whatever list you got back
            global_state.open_orders_ts = time.time()
            #print("DEBUG: open_orders: ", open_orders)

            open_ids = []
            for o in open_orders or []:
                oid = _extract_id(o)
                if oid:
                    open_ids.append(oid)

            cancel_ids = [oid for oid in open_ids if oid not in protected]

            if cancel_ids:
                if VERBOSE:
                    print(f"[RECONCILE_ALL] open={len(open_ids)} protected={len(protected)} cancelling={len(cancel_ids)}")
                await cancel_many_orders(cancel_ids)

        except PolyApiException as e:
            if e.status_code == 401:
                print(f"[RECONCILE_ALL] 401 Unauthorized - refreshing credentials...")
                try:
                    global_state.client.refresh_creds()
                except Exception as refresh_err:
                    print(f"[RECONCILE_ALL] Failed to refresh creds: {refresh_err}")
            else:
                print(f"[RECONCILE_ALL] API error: {e}")
        except Exception as e:
            print(f"[RECONCILE_ALL] error: {e}")

        await asyncio.sleep(.50)
        if VERBOSE:
            dt = time.time() - t0
            print(f"[RECONCILE_ALL] loop took {dt:.3f}s")

def pending_yes_from_open_orders(
    market_id: str,
    open_orders: list[dict],
) -> float:
    """
    Convert open orders into pending YES exposure for this market.
    Assumes open_orders entries contain: token_id/asset_id, side, size.

    Also includes locally tracked working orders to prevent race conditions
    where orders are placed but not yet reflected in API-sourced open_orders.
    """
    yes_token = global_state.condition_to_token_id[market_id]
    no_token  = global_state.REVERSE_TOKENS[yes_token]

    pending = 0.0
    counted_order_ids = set()

    # Count orders from API-sourced open_orders
    for o in open_orders:
        order_id = o.get("id")
        if order_id:
            counted_order_ids.add(order_id)

        token = o.get("token_id") or o.get("asset_id") or o.get("tokenId") or o.get("assetId")
        side  = (o.get("side") or "").upper()
        size  = float(o.get("size") or o.get("original_size") or o.get("remaining_size") or 0.0)

        if size <= 0:
            continue

        if token == yes_token:
            if side == "BUY":
                pending += size
            elif side == "SELL":
                pending -= size

        elif token == no_token:
            if side == "BUY":
                pending -= size
            elif side == "SELL":
                pending += size

    # Add locally tracked working orders if not already counted
    # This handles race condition where order was just placed but API hasn't updated yet
    if hasattr(global_state, "working_orders_by_market"):
        wo = global_state.working_orders_by_market.get(market_id, {})
        for side_key in ["bid", "ask"]:
            order = wo.get(side_key)
            if order and isinstance(order, dict):
                order_id = order.get("id")
                # Skip if already counted from open_orders
                if order_id and order_id in counted_order_ids:
                    continue

                token = order.get("token")
                size = order.get("size", 0)
                side = (order.get("side") or "").upper()

                if size <= 0:
                    continue

                if token == yes_token:
                    if side == "BUY":
                        pending += size
                    elif side == "SELL":
                        pending -= size
                elif token == no_token:
                    if side == "BUY":
                        pending -= size
                    elif side == "SELL":
                        pending += size

    return pending

async def reconcile_loop():
    """
    Periodically reconciles for every market in working_orders_by_market.
    """
    while True:
        try:
            wom = getattr(global_state, "working_orders_by_market", {})
            market_ids = list(wom.keys()) if isinstance(wom, dict) else []
            for mkt in market_ids:
                await reconcile_market_orders(mkt)
        except PolyApiException as e:
            if e.status_code == 401:
                print(f"[RECONCILE_LOOP] 401 Unauthorized - refreshing credentials...")
                try:
                    global_state.client.refresh_creds()
                except Exception as refresh_err:
                    print(f"[RECONCILE_LOOP] Failed to refresh creds: {refresh_err}")
            else:
                print(f"[RECONCILE_LOOP] API error: {e}")
        except Exception as e:
            print(f"[RECONCILE_LOOP] unexpected error: {e}")

        await asyncio.sleep(RECONCILE_INTERVAL)

# TODO: QUOTE TIGHTER- I BELIEVE NEW ROUNDING LOGIC MEANS SPREAD=.03 IS NOT AS TIGHT AS WE THINK, TRY .02 or EVEN .015
# TODO: REDUCE LATENCY, CLEAR PRINT STATEMENTS, REDUCE BACKGROUND TASKS
# had .04 base width before, skew_k=1.0, min_order_interval=1.0, price_move_tol = .0035

BASE_QUOTE_SPREAD = 0.050 # Up to .055
MAX_POSITION = 60
BASE_SIZE = 20.0 # Base size/max pos was 5 / 30
ALIGNED_SIGNAL_SIZE = 20  # Order size when z_skew and book_imbalance agree
#INV_SKEW_PER_SHARE = 0.00050

SKEW_K = 1.0          # 0.3–1.0, start ~0.6
SKEW_CAP = 0.04       # max skew in price points (5c)

MIN_PRICE = 0.01
MAX_PRICE = 0.99
PRICE_MOVE_TOL = 0.0015          # don't cancel/replace if existing quote is within 0.5c of target
TICK_SIZE = .01
MIN_TICKS_BUILD = 0   # ticks from touch when building position (more conservative)
MIN_TICKS_REDUCE = 0   # ticks from touch when reducing position (want to get filled)
MIN_EDGE_TO_QUOTE = 0.02  # minimum edge (in price points) required to quote a side

MIN_ORDER_INTERVAL = .20  # seconds → max 5 orders/sec per market+side, # changed this back to 1
POST_FILL_COOLDOWN = .25  # seconds to pause quoting on a side after getting filled (GTC only)

# Binance momentum adjustment
USE_BINANCE_MOMENTUM = False  # Toggle to use Binance momentum for predictive quoting
BINANCE_MOMENTUM_LOOKBACK = 0.5  # Seconds to look back for momentum calculation
# MAX_MOMENTUM_ADJUSTMENT imported from trading_config

# Dynamic spread based on option price sensitivity
OPTION_MOVE_LOOKBACK = 1.0        # Seconds to look back for BTC move
OPTION_MOVE_THRESHOLD = 0.02      # 2 cents - start widening when option moved this much (was 0.02)
OPTION_MOVE_SPREAD_SCALE = 0.25    # spread multiplier per cent above threshold (was 0.5)
MAX_OPTION_SPREAD_MULT = 1.0      # was 2.0, decreasing to 1.0 to turn this feature off!

# Book imbalance adjustment - constants imported from trading_config
# USE_BOOK_IMBALANCE, BOOK_IMBALANCE_LEVELS, MAX_IMBALANCE_ADJUSTMENT

# Early cancel threshold (option price sensitivity)
EARLY_CANCEL_OPTION_MOVE = .30    # 1/2/26: was .35, changing to .30

# Coinbase-RTDS z-score threshold (predictive edge detection when using RTDS)
COINBASE_RTDS_ZSCORE_THRESHOLD = 0.60  # Skip vulnerable side when |z| > 0.70
Z_SCORE_COMBINED_THRESHOLD = 0.30      # Combined threshold (half of main)
Z_SKEW_COMBINED_THRESHOLD = 0.025      # 2.5¢ predicted option move threshold for combined rule

# Z-score skew - MAX_Z_SCORE_SKEW imported from trading_config

# Total signal adjustment cap - MAX_TOTAL_SIGNAL_ADJUSTMENT imported from trading_config

VERBOSE = False


async def _send_order_locked(token_id: str, side: str, price: float, size: float, tif: str, *, market_id: str | None = None):
    """
    Internal order sender - REQUIRES caller holds global_state.position_check_lock

    Contains all order sending logic including position checks, throttling, and API calls.
    This function assumes the caller has already acquired position_check_lock.

    Returns order_id on success, None on failure/throttle.

    side: "BUY" or "SELL"
    tif:  "IOC" or "GTC"
    token_id: YES or NO token id to trade
    market_id: optional, used only for nicer logs & throttling key
    """

    # 0) Risk gate
    if not getattr(global_state, "trading_enabled", True):
        #print(f"[SEND_ORDER] trading_enabled=False, skipping {side} {size} @ {price:.4f} ({tif}) on token {token_id}")
        return None

    client = global_state.client
    if client is None:
        #print("[SEND_ORDER] No client set in global_state.client")
        return None

    side = side.upper()
    tif = tif.upper()

    if market_id is None:
        # If you ever call send_order without market_id, safest is to block
        #print("[RISK] send_order called without market_id; blocking to enforce MAX_POSITION.")
        return None

    # Position check logic (lock must be held by caller)
    try:
        yes_token = global_state.condition_to_token_id[market_id]
        no_token = global_state.REVERSE_TOKENS[yes_token]

        # filled net YES - this is the ACTUAL position from confirmed fills
        pos_obj = global_state.positions_by_market.get(market_id)
        filled_yes = pos_obj.net_yes if pos_obj is not None else 0.0

        # pending net YES from working_orders_by_market (GTC orders)
        # ONLY count orders that INCREASE absolute position (consume limit headroom)
        # Do NOT count orders that reduce position (they shouldn't create phantom headroom)
        pending_yes = 0.0
        wo = getattr(global_state, "working_orders_by_market", {}).get(market_id, {})
        for side_key in ("bid", "ask"):
            entry = wo.get(side_key)
            if not isinstance(entry, dict):
                continue
            tok = entry.get("token")
            sde = (entry.get("side") or "").upper()
            qty = float(entry.get("size") or 0.0)
            if qty <= 0:
                continue

            # Calculate delta for this pending order
            if tok == yes_token:
                order_delta = qty if sde == "BUY" else -qty
            elif tok == no_token:
                order_delta = -qty if sde == "BUY" else +qty
            else:
                continue

            # Only count if this order would INCREASE absolute position
            # (i.e., pushes position further from zero in the same direction)
            if filled_yes >= 0 and order_delta > 0:
                # Long position, order adds more long → count it
                pending_yes += order_delta
            elif filled_yes <= 0 and order_delta < 0:
                # Short position, order adds more short → count it
                pending_yes += order_delta
            # Otherwise: order reduces position, don't count (no phantom headroom)

        # Include pending_order_delta (tracks in-flight IOC orders)
        # Delta is added when IOC order is placed, removed when TRADE event confirms fill
        pending_delta = getattr(global_state, "pending_order_delta", {}).get(market_id, 0.0)
        effective_yes = filled_yes + pending_yes + pending_delta

        # delta from THIS order
        if token_id == yes_token:
            delta_yes = +size if side == "BUY" else -size
        elif token_id == no_token:
            delta_yes = -size if side == "BUY" else +size
        else:
            #print(f"[RISK] token_id not recognized for market {market_id}; blocking.")
            return None

        projected = effective_yes + delta_yes

        # If already beyond limit, only allow orders that REDUCE absolute exposure
        if abs(effective_yes) > MAX_POSITION:
            if abs(projected) >= abs(effective_yes):
                #print(
                 #   f"[RISK] Blocked (not reducing): eff_yes={effective_yes:.1f} delta={delta_yes:.1f} proj={projected:.1f} MAX={MAX_POSITION}")
                return None
        else:
            # Normal case: inside band -> don't allow crossing out
            if abs(projected) > MAX_POSITION:
                #print(
                 #   f"[RISK] Blocked: eff_yes={effective_yes:.1f} delta={delta_yes:.1f} proj={projected:.1f} MAX={MAX_POSITION}")
                return None

    except Exception as e:
        # Fail-safe: if we can't compute risk, don't trade.
        #print(f"[RISK] Could not compute projected position; blocking order. err={e}")
        return None
    # End new blocker

    # 1) Throttle by (key, side, tif)
    # Use token_id as part of key so YES and NO are throttled separately
    if market_id is not None:
        key = (market_id, token_id, side, tif)
    else:
        key = (token_id, side, tif)

    now = time.time()
    last = global_state.last_order_time.get(key, 0.0)

    if now - last < MIN_ORDER_INTERVAL:
        # Too soon; skip sending
        # print(f"[THROTTLE] {side} {key_id} {tif}: {now - last:.3f}s since last order (min {MIN_ORDER_INTERVAL}s)")
        return None

    # Update timestamp BEFORE sending to clip bursts
    global_state.last_order_time[key] = now

    # Track pending order delta BEFORE placing order (for IOC orders only - GTC tracked in working_orders)
    order_delta = 0.0
    if tif == "IOC":
        if not hasattr(global_state, "pending_order_delta"):
            global_state.pending_order_delta = {}
        if market_id not in global_state.pending_order_delta:
            global_state.pending_order_delta[market_id] = 0.0
        if not hasattr(global_state, "ioc_order_deltas"):
            global_state.ioc_order_deltas = {}  # {order_id: (market_id, delta, timestamp)}

        # Calculate delta for this order
        yes_token = global_state.condition_to_token_id.get(market_id)
        no_token = global_state.REVERSE_TOKENS.get(yes_token) if yes_token else None
        if token_id == yes_token:
            order_delta = +size if side == "BUY" else -size
        elif token_id == no_token:
            order_delta = -size if side == "BUY" else +size

        # Add to pending delta BEFORE placing order
        global_state.pending_order_delta[market_id] += order_delta

    # 2) DRY RUN mode – only print what we *would* do
    if getattr(global_state, "dry_run", False):
        print(f"[DRY_RUN] Would place {tif} {side} {size} @ {price:.4f} on token {token_id} (market={market_id})")
        return "DRY_RUN"

    # 3) Real order – use create_order(token_id, ...) directly
    try:
        order_id = await asyncio.to_thread(
            client.create_order,   # NOTE: token-level
            token_id,
            side,
            price,
            size,
            tif,
        )
        # Track IOC order_id -> delta mapping for position tracking
        if tif == "IOC" and order_delta != 0.0 and order_id:
            global_state.ioc_order_deltas[order_id] = (market_id, order_delta, now)
        #print(f"[SEND_ORDER] Placed {tif} {side} {size} @ {price:.4f} on token {token_id} (market={market_id}) -> {order_id}")
        return order_id

    except PolyApiException as e:
        # Revert pending delta on failure (IOC only)
        if order_delta != 0.0:
            global_state.pending_order_delta[market_id] -= order_delta
        if e.status_code == 401:
            print(f"[SEND_ORDER] 401 Unauthorized - refreshing credentials...")
            try:
                global_state.client.refresh_creds()
            except Exception as refresh_err:
                print(f"[SEND_ORDER] Failed to refresh creds: {refresh_err}")
        return None
    except Exception as e:
        # Revert pending delta on failure (IOC only)
        if order_delta != 0.0:
            global_state.pending_order_delta[market_id] -= order_delta
        #print(f"[SEND_ORDER][ERROR] Unexpected exception: {e}")
        return None


async def send_order(token_id: str, side: str, price: float, size: float, tif: str, *, market_id: str | None = None):
    """
    Public order sender - acquires position_check_lock and calls _send_order_locked().

    Use this for IOC orders and other callers outside of manage_side().
    manage_side() should call _send_order_locked() directly with extended lock scope.

    side: "BUY" or "SELL"
    tif:  "IOC" or "GTC"
    token_id: YES or NO token id to trade
    market_id: optional, used for position checks and throttling
    """
    async with global_state.position_check_lock:
        return await _send_order_locked(token_id, side, price, size, tif, market_id=market_id)


async def cancel_order_async(order_id: str, max_retries: int = 2):
    for attempt in range(1, max_retries + 1):
        try:
            await asyncio.to_thread(global_state.client.cancel_orders, [order_id])
            #print(f"[CANCEL] Successfully cancelled {order_id} on attempt {attempt}")
            return
        except PolyApiException as e:
            #print(f"[CANCEL] Attempt {attempt} failed for {order_id}: {e}")
            if attempt == max_retries:
                #print(f"[CANCEL] Giving up cancelling {order_id} after {max_retries} attempts")
                return
            await asyncio.sleep(0.25 * attempt)  # simple backoff
        except Exception as e:
            #print(f"[CANCEL] Unexpected exception for {order_id} on attempt {attempt}: {e}")
            return

async def cancel_many_orders(order_ids: list[str], max_retries: int = 3):
    """
    Cancel many orders in a single API call.
    Retries on transient Polymarket API failures.
    """
    if not order_ids:
        return

    for attempt in range(1, max_retries + 1):
        try:
            resp = await asyncio.to_thread(global_state.client.cancel_orders, order_ids)
            #print(f"[CANCEL_MANY] Successfully cancelled {len(order_ids)} orders on attempt {attempt}")
            return resp

        except Exception as e:
            name = e.__class__.__name__
            msg = str(e)

            # Detect Polymarket-specific exception by name or message pattern
            if "PolyApiException" in name or "Request exception" in msg:
                #print(f"[CANCEL_MANY] Attempt {attempt} failed: {e}")

                if attempt == max_retries:
                    #print(f"[CANCEL_MANY] Giving up after {max_retries} attempts.")
                    return None

                await asyncio.sleep(0.5 * attempt)  # backoff
                continue

            # Unknown fatal exception → don't retry
            #print(f"[CANCEL_MANY] Unexpected exception: {name}: {e}")
            return None

def compute_book_imbalance(market_id: str) -> tuple[float, float, float]:
    """
    Compute order book imbalance for a market.

    Returns: (imbalance, bid_depth, ask_depth)
      imbalance: -1 to +1 where +1 = all bids, -1 = all asks
      bid_depth: total size on bid side
      ask_depth: total size on ask side
    """
    book = global_state.all_data.get(market_id)
    if not book:
        return 0.0, 0.0, 0.0

    bids = book.get("bids")
    asks = book.get("asks")
    if not bids or not asks:
        return 0.0, 0.0, 0.0

    # Sum depth on each side (top N levels or all)
    if BOOK_IMBALANCE_LEVELS > 0:
        # Top N levels only
        bid_items = list(bids.items())[-BOOK_IMBALANCE_LEVELS:]  # highest bids
        ask_items = list(asks.items())[:BOOK_IMBALANCE_LEVELS]   # lowest asks
        bid_depth = sum(size for price, size in bid_items)
        ask_depth = sum(size for price, size in ask_items)
    else:
        # All levels
        bid_depth = sum(size for price, size in bids.items())
        ask_depth = sum(size for price, size in asks.items())

    total = bid_depth + ask_depth
    if total == 0:
        return 0.0, 0.0, 0.0

    imbalance = (bid_depth - ask_depth) / total
    return imbalance, bid_depth, ask_depth

def compute_edge_and_quotes(market_id):
    book = global_state.all_data.get(market_id)
    if not book:
        return None

    bids = book["bids"]
    asks = book["asks"]
    if not bids or not asks:
        return None

    best_bid = bids.peekitem(-1)[0]
    best_ask = asks.peekitem(0)[0]
    mid = 0.5 * (best_bid + best_ask)

    fair_px = global_state.fair_value.get(market_id)
    if fair_px is None:
        return None

    edge_bid = best_bid - fair_px   # how rich current bid is vs your fair
    edge_ask = fair_px - best_ask   # how cheap current ask is vs your fair

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "fair": fair_px,
        "edge_bid": edge_bid,
        "edge_ask": edge_ask,
    }

def update_position_yes_space(market_id: str, outcome_token: str, side: str, size: float, price: float, order_type: str = "GTC"):
    """
    Update net YES exposure for a market, given a trade in either YES or NO token.

    - outcome_token: token that was traded (YES or NO)
    - side: "buy" or "sell" on that token (case-insensitive)
    - size: number of tokens
    - price: fill price
    - order_type: "GTC" for maker/passive fills, "IOC" for taker/aggressive fills

    Convention:
      net_yes > 0  => long YES
      net_yes < 0  => short YES
    """
    if size <= 0:
        #print("[UPDATE POS] size <= 0, skipping")
        return

    record_fill(market_id, outcome_token, side, price, size, ts=time.time(), order_type=order_type)

    # Map market -> YES token
    try:
        yes_token = global_state.condition_to_token_id[market_id]
    except KeyError:
        #print("[UPDATE POS] Unknown market_id in condition_to_token_id:", market_id)
        return

    no_token = global_state.REVERSE_TOKENS.get(yes_token)

    # Get or create MarketPosition
    pos = global_state.positions_by_market.get(market_id)
    if pos is None:
        pos = global_state.MarketPosition()
        global_state.positions_by_market[market_id] = pos

    side = side.lower()

    # Determine delta_yes
    if outcome_token == yes_token:
        # Direct YES trade
        if side == "buy":
            delta_yes = size
        elif side == "sell":
            delta_yes = -size
        else:
            #print("[UPDATE POS] Unknown side for YES token:", side)
            return
        which = "YES"

    elif no_token is not None and outcome_token == no_token:
        # NO trade mapped to YES-space
        if side == "buy":
            delta_yes = -size   # buy NO = short YES
        elif side == "sell":
            delta_yes = size    # sell NO = long YES
        else:
            #print("[UPDATE POS] Unknown side for NO token:", side)
            return
        which = "NO"

    else:
        print(
            "[UPDATE POS] outcome_token not recognized for market:",
            market_id,
            "outcome_token:", outcome_token,
            "yes_token:", yes_token,
            "no_token:", no_token,
        )
        return

    old_net = pos.net_yes
    new_net = old_net + delta_yes

    # Simple VWAP update
    if old_net == 0 or (old_net > 0 and new_net > 0) or (old_net < 0 and new_net < 0):
        total_notional = pos.vwap_yes * abs(old_net) + price * size
        pos.net_yes = new_net
        pos.vwap_yes = total_notional / abs(pos.net_yes) if pos.net_yes != 0 else 0.0
    else:
        pos.net_yes = new_net
        if pos.net_yes == 0:
            pos.vwap_yes = 0.0

    # Track fill time for post-fill cooldown (GTC fills only)
    # BUY YES fill → pause "bid" side, BUY NO fill → pause "ask" side
    if order_type == "GTC" and side == "buy":
        if not hasattr(global_state, "last_fill_time"):
            global_state.last_fill_time = {}
        if market_id not in global_state.last_fill_time:
            global_state.last_fill_time[market_id] = {"bid": 0.0, "ask": 0.0}

        if outcome_token == yes_token:
            global_state.last_fill_time[market_id]["bid"] = time.time()
        elif outcome_token == no_token:
            global_state.last_fill_time[market_id]["ask"] = time.time()

    if VERBOSE:
        print(
            f"[UPDATE POS] market={market_id} token_side={which}/{side} "
            f"delta_yes={delta_yes} old_net={old_net} new_net={pos.net_yes} "
            f"vwap_yes={pos.vwap_yes:.4f} price={price}"
        )

async def perform_trade(market_id: str):
    """
    MM logic for BTC 15m up/down market where:
      - Decisions & edges are in YES-space
      - Cheap YES -> BUY YES
      - Rich YES  -> BUY NO (instead of SELL YES)
    """
    # Only trade the active market configured by the scheduler
    if market_id != global_state.active_market_id:
        return

    # Acquire per-market lock to prevent concurrent perform_trade() calls from double-counting capacity
    # This makes the entire "check position → calculate capacity → place orders" sequence atomic
    with market_locks_mutex:
        if market_id not in market_locks:
            market_locks[market_id] = asyncio.Lock()

    async with market_locks[market_id]:
        await _perform_trade_locked(market_id)


async def _perform_trade_locked(market_id: str):
    """
    Internal implementation of perform_trade() - assumes caller holds market lock.
    """
    now = time.time()  # Cache time once for this call

    # CRITICAL: Check if price data is stale (prevent trading on disconnected/frozen websocket)
    PRICE_STALENESS_THRESHOLD = 30  # seconds
    price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')

    # Check appropriate price feed based on PRICE_SOURCE
    if price_source == "RTDS":
        last_update = getattr(global_state, 'rtds_last_update_time', None)
        source_name = "RTDS"
    elif price_source == "COINBASE":
        last_update = getattr(global_state, 'coinbase_mid_ts', None)
        source_name = "Coinbase"
    elif price_source == "BLEND":
        # For blended price, check both RTDS and Binance
        rtds_update = getattr(global_state, 'rtds_last_update_time', None)
        binance_update = getattr(global_state, 'binance_mid_ts', None)
        if rtds_update is None or binance_update is None:
            return
        last_update = min(rtds_update, binance_update)  # Check oldest of the two
        source_name = "Blended (RTDS+Binance)"
    else:
        last_update = None
        source_name = "Unknown"

    if last_update is None:
        # Price feed has never received data - don't trade
        return

    time_since_update = now - last_update
    if time_since_update > PRICE_STALENESS_THRESHOLD:
        # Price is stale - websocket may be silently disconnected
        # Cancel all orders and wait for fresh data
        if not hasattr(global_state, '_price_stale_warning_printed') or now - global_state._price_stale_warning_printed > 60:
            print(f"⚠️  [CRITICAL] {source_name} PRICE IS STALE ({time_since_update:.1f}s old)! Canceling all orders and halting trading.")
            print(f"    Last price update: {time_since_update:.1f}s ago (threshold: {PRICE_STALENESS_THRESHOLD}s)")
            print(f"    Websocket may be silently disconnected. Waiting for reconnection...")
            global_state._price_stale_warning_printed = now

        # Cancel all working orders to prevent trading on stale prices
        working = global_state.working_orders_by_market.get(market_id, {})
        if working:
            order_ids = list(working.keys())
            if order_ids:
                try:
                    await asyncio.wait_for(global_state.client.cancel_orders(order_ids), timeout=2.0)
                    print(f"    Canceled {len(order_ids)} orders due to stale {source_name} price")
                except:
                    pass
        return

    # CRITICAL: Check if user websocket events are stale (prevent position tracking freeze)
    # If user websocket is dead, TRADE events won't be processed and filled_yes will be frozen
    # This can cause catastrophic position limit breaches
    USER_WS_STALENESS_THRESHOLD = 90  # seconds - conservative since we might not get events if no fills
    user_ws_last_event = getattr(global_state, 'user_ws_last_event_time', None)

    if user_ws_last_event is not None:
        time_since_user_ws_event = now - user_ws_last_event
        if time_since_user_ws_event > USER_WS_STALENESS_THRESHOLD:
            # User websocket is stale - position tracking may be frozen
            # Cancel all orders and halt trading until websocket reconnects
            if not hasattr(global_state, '_user_ws_stale_warning_printed') or now - global_state._user_ws_stale_warning_printed > 60:
                print(f"⚠️  [CRITICAL] USER WEBSOCKET IS STALE ({time_since_user_ws_event:.1f}s since last event)!")
                print(f"    Position tracking may be frozen. Canceling all orders and halting trading.")
                print(f"    Threshold: {USER_WS_STALENESS_THRESHOLD}s. Waiting for user websocket reconnection...")
                global_state._user_ws_stale_warning_printed = now

            # Cancel all working orders to prevent position tracking desync
            working = global_state.working_orders_by_market.get(market_id, {})
            if working:
                order_ids = []
                for side_key in ["bid", "ask"]:
                    entry = working.get(side_key)
                    if entry and isinstance(entry, dict):
                        oid = entry.get("id")
                        if oid:
                            order_ids.append(oid)
                if order_ids:
                    try:
                        await asyncio.wait_for(global_state.client.cancel_orders(order_ids), timeout=2.0)
                        print(f"    Canceled {len(order_ids)} orders due to stale user websocket")
                    except:
                        pass
            return

    #return # TODO: return to prevent even printing
    if VERBOSE:
        print(f"[MM] perform_trade called for market_id={market_id}, dry_run={getattr(global_state, 'dry_run', None)}")
    info = compute_edge_and_quotes(market_id)
    if info is None:
        if VERBOSE:
            print(f"[MM] perform_trade: no edge info for {market_id}, returning")
        return

    best_bid_yes = info["best_bid"]
    best_ask_yes = info["best_ask"]
    book_mid = .5*(best_bid_yes+best_ask_yes)

    # Use book mid as base for quoting (market's current forward-looking price)
    # We'll add z_skew_residual to avoid double-counting when market has already moved
    theo = info["fair"]  # global_state.fair_value[market_id]
    fair_yes = book_mid

    # Add momentum adjustment if enabled (uses Coinbase if USE_COINBASE_PRICE=True, else Binance)
    if not USE_BINANCE_MOMENTUM:
        pass  # Skip momentum calculation entirely
    elif USE_BINANCE_MOMENTUM:
        momentum = 0.0

        # Use appropriate price history based on configuration
        price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
        if price_source in ("COINBASE", "RTDS"):
            # Use Coinbase history for both (RTDS doesn't have its own)
            price_history = global_state.coinbase_price_history
        else:  # BLEND
            price_history = global_state.binance_price_history

        if len(price_history) >= 2:
            current_price = price_history[-1][1]

            # Find price from MOMENTUM_LOOKBACK seconds ago
            for ts, price in reversed(list(price_history)[:-1]):
                if now - ts >= BINANCE_MOMENTUM_LOOKBACK:  # Note: variable name is legacy, works for both Coinbase and Binance
                    old_price = price
                    momentum = current_price - old_price
                    break

        # Reprice option with momentum-adjusted spot (includes gamma!)
        # Get current spot price and option parameters
        price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
        if price_source == "COINBASE":
            S_current = global_state.coinbase_mid_price
        elif price_source == "RTDS":
            S_current = global_state.mid_price
        else:  # BLEND
            S_current = global_state.blended_price

        sigma = global_state.fair_vol.get(market_id)

        if S_current is not None and sigma is not None and momentum != 0:
            # Get option parameters
            K = global_state.strike
            now_et = datetime.now(ZoneInfo("America/New_York"))
            T = (global_state.exp - now_et).total_seconds() / (60 * 60 * 24 * 365)
            r = 0.0
            q = 0.0
            payoff = 1.0

            if T > 0:
                # Price option at current spot
                current_option_price = bs_binary_call(S_current, K, T, r, sigma, q, payoff)

                # Price option at spot + momentum
                S_after_momentum = S_current + momentum
                new_option_price = bs_binary_call(S_after_momentum, K, T, r, sigma, q, payoff)

                # Predicted move (automatically includes gamma!)
                predicted_option_move = new_option_price - current_option_price

                # Cap adjustment to avoid going crazy
                predicted_option_move = max(-MAX_MOMENTUM_ADJUSTMENT, min(MAX_MOMENTUM_ADJUSTMENT, predicted_option_move))

                # Quote ahead of where market will move
                fair_yes = book_mid + predicted_option_move
            else:
                # At expiry, no adjustment
                fair_yes = book_mid
        else:
            # Missing data, no adjustment
            fair_yes = book_mid

    # Book imbalance adjustment: lean quotes in direction of order flow
    book_imbalance = 0.0
    imbalance_adj = 0.0  # Track for total signal adjustment cap
    if USE_BOOK_IMBALANCE:
        imbalance, bid_depth, ask_depth = compute_book_imbalance(market_id)
        book_imbalance = imbalance
        # More bids than asks → price likely rising → nudge fair up
        imbalance_adj = imbalance * MAX_IMBALANCE_ADJUSTMENT
        # Note: We apply this AFTER z_skew is calculated to cap total adjustment
        if VERBOSE and abs(imbalance) > 0.3:
            print(f"[MM] Book imbalance: {imbalance:.2f} (bid={bid_depth:.0f}, ask={ask_depth:.0f}), adj={imbalance_adj:.4f}")

    # Store imbalance in global_state for markout tracking
    if not hasattr(global_state, "book_imbalance"):
        global_state.book_imbalance = {}
    global_state.book_imbalance[market_id] = book_imbalance

    pos_obj = global_state.positions_by_market.get(market_id)
    filled_yes = pos_obj.net_yes if pos_obj is not None else 0.0

    pending_yes = pending_yes_from_open_orders(market_id, global_state.open_orders)
    net_yes = filled_yes + pending_yes  # effective net, used for risk + skew
    #net_yes = filled_yes

    if VERBOSE:
        print(f"[MM] pos_yes filled={filled_yes} pending={pending_yes} effective={net_yes}")

    def clamp_price_to_tick(p: float) -> float:
        p_rounded = round(p / TICK_SIZE) * TICK_SIZE
        return max(MIN_PRICE, min(p_rounded, MAX_PRICE))

    def yes_to_no_price(p_yes: float) -> float:
        p_no = 1.0 - p_yes
        return clamp_price_to_tick(p_no)

    client = global_state.client
    if client is None:
        if VERBOSE:
            print("[MM] perform_trade: No client set, returning")
        return

    yes_token = global_state.condition_to_token_id[market_id]
    no_token  = global_state.REVERSE_TOKENS[yes_token]

    if not hasattr(global_state, "working_orders_by_market"):
        global_state.working_orders_by_market = {}
    wo = global_state.working_orders_by_market.get(market_id)
    if wo is None:
        wo = {"bid": None, "ask": None}
        global_state.working_orders_by_market[market_id] = wo

    #warmup logic, widen out initially
    W = 10 # duration in seconds
    START_WIDE = 1.25  # 80% wider at t=0
    age = now - getattr(global_state, "bot_start_ts", now)
    warm = max(0.0, 1.0 - age / W)  # 1 -> 0
    mult_warm = 1.0 + (START_WIDE - 1.0) * warm

    # Multipliers (currently disabled - kept for potential future use)
    mult = 1
    option_move_mult = 1.0  # MAX_OPTION_SPREAD_MULT = 1.0 disables dynamic spread widening

    quote_spread = BASE_QUOTE_SPREAD * mult * mult_warm * option_move_mult

    # Store spread multiplier for display
    if not hasattr(global_state, 'spread_mult_by_market'):
        global_state.spread_mult_by_market = {}
    global_state.spread_mult_by_market[market_id] = option_move_mult

    # --- time-to-expiry position scaling ---
    # Reduce max position near expiry to limit gamma risk
    effective_max_position = MAX_POSITION
    now_et = datetime.now(ZoneInfo("America/New_York"))
    minutes_to_expiry = (global_state.exp - now_et).total_seconds() / 60.0

    if minutes_to_expiry < 5.0:
        # Halve max position in final 5 minutes, round to nearest multiple of 5
        effective_max_position = round(MAX_POSITION / 2 / 5) * 5
        effective_max_position = max(5, effective_max_position)  # minimum of 5
        if VERBOSE:
            print(f"[MM] Near expiry ({minutes_to_expiry:.1f}min left), reducing max position to {effective_max_position}")

    # --- dynamic inventory skew (scaled to spread, capped, stronger when "dangerous") ---
    # half spread should be the (possibly delta-adjusted) one you actually plan to quote
    half_spread = quote_spread / 2.0  # if you compute quote_spread dynamically
    # half_spread = QUOTE_SPREAD / 2.0 # if not

    inv_ratio = 0.0
    if effective_max_position > 0:
        inv_ratio = max(-1.0, min(1.0, net_yes / effective_max_position))  # net_yes = effective yes (filled+pending)

    skew = -inv_ratio * (SKEW_K * half_spread)  # long YES -> skew down, short YES -> skew up
    skew *= mult  # make skew stronger when market is dangerous (high delta / near expiry)
    skew = max(-SKEW_CAP, min(SKEW_CAP, skew))  # hard cap

    fair_adj_yes = fair_yes + skew

    # Z-score skew: Adjust fair value based on predicted RTDS movement
    # We calculate the RESIDUAL skew (what market hasn't already priced)
    z_skew = 0.0
    z_skew_residual = 0.0
    z_skew_raw = 0.0  # Track uncapped z_skew for aggressive mode detection
    z_score = getattr(global_state, 'coinbase_rtds_zscore', 0.0)
    if z_score != 0.0:
        # Use cached spread std (calculated in coinbase_price_stream when z-score updates)
        # This avoids expensive np.std() recalculation on every perform_trade() call
        spread_std = getattr(global_state, 'coinbase_rtds_spread_std', 0.0)

        if spread_std > 0:
            # Predicted RTDS move = z_score × spread_std
            predicted_rtds_move = z_score * spread_std

            # Reprice option to capture gamma (not just delta approximation)
            price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
            if price_source == "COINBASE":
                S_current = global_state.coinbase_mid_price
            elif price_source == "RTDS":
                S_current = global_state.mid_price
            else:  # BLEND
                S_current = global_state.blended_price

            sigma = global_state.fair_vol.get(market_id)

            if S_current is not None and sigma is not None:
                K = global_state.strike
                now_et = datetime.now(ZoneInfo("America/New_York"))
                T = (global_state.exp - now_et).total_seconds() / (60 * 60 * 24 * 365)

                if T > 0:
                    # Price option at current spot
                    current_option = bs_binary_call(S_current, K, T, 0.0, sigma, 0.0, 1.0)

                    # Price option at spot + predicted move (includes gamma!)
                    future_option = bs_binary_call(S_current + predicted_rtds_move, K, T, 0.0, sigma, 0.0, 1.0)

                    # Z-skew is the predicted option value change
                    z_skew_raw = future_option - current_option

                    # Apply sigmoid confidence scaling based on z_score magnitude
                    # This filters noise when z_score is small (low signal) and amplifies when z_score is large (high signal)
                    z_score_abs = abs(z_score)
                    z_confidence = 1.0 / (1.0 + math.exp(-Z_SCORE_CONFIDENCE_STEEPNESS * (z_score_abs - Z_SCORE_CONFIDENCE_MIDPOINT)))
                    z_skew_adjusted = z_skew_raw * z_confidence

                    # Calculate residual from adjusted z_skew (not raw)
                    # This correctly compares sigmoid-scaled prediction vs full market move
                    market_implied_move = book_mid - theo
                    z_skew_residual = z_skew_adjusted - market_implied_move

                    # Store adjusted z_skew (capping happens later, after aggressive_mode is determined)
                    z_skew = z_skew_adjusted

                    if VERBOSE and abs(market_implied_move) > 0.01:
                        print(f"[MM] Z-skew: raw={z_skew_raw:.4f}, conf={z_confidence:.2f}, adjusted={z_skew_adjusted:.4f}, market_priced={market_implied_move:.4f}, residual={z_skew_residual:.4f}")

    # Aggressive mode: increase cap when high conviction z-score signal
    # Conditions: |z_score| > threshold, |z_skew_residual| > threshold, signals aligned
    aggressive_mode = (
        AGGRESSIVE_MODE_ENABLED and
        abs(z_score) > AGGRESSIVE_Z_THRESHOLD and
        abs(z_skew_residual) > AGGRESSIVE_ZSKEW_THRESHOLD and
        z_score * book_imbalance > 0  # Require z-score and book imbalance alignment
    )

    # Cap z_skew and z_skew_residual - use higher cap in aggressive mode
    z_skew_cap = AGGRESSIVE_MAX_Z_SCORE_SKEW if aggressive_mode else MAX_Z_SCORE_SKEW
    z_skew = max(-z_skew_cap, min(z_skew_cap, z_skew))
    z_skew_residual = max(-z_skew_cap, min(z_skew_cap, z_skew_residual))

    # Apply signal adjustments (book imbalance + z-score residual skew) with total cap
    # This prevents crossing the spread when both signals fire strongly in same direction
    # Use z_skew_residual to avoid double-counting what market has already priced
    total_signal_adj = imbalance_adj + z_skew_residual

    # Track original values for diagnostics (after z_skew cap but before total cap)
    original_imbalance_adj = imbalance_adj
    original_z_skew_residual = z_skew_residual

    effective_cap = AGGRESSIVE_MAX_TOTAL_ADJUSTMENT if aggressive_mode else MAX_TOTAL_SIGNAL_ADJUSTMENT

    # Store aggressive_mode in global_state for markout tracking
    if not hasattr(global_state, 'aggressive_mode_by_market'):
        global_state.aggressive_mode_by_market = {}
    global_state.aggressive_mode_by_market[market_id] = aggressive_mode

    if aggressive_mode:
        print(f"[MM] AGGRESSIVE MODE: z={z_score:.2f}, z_skew_raw={z_skew_raw:.4f}, z_skew_res={z_skew_residual:.4f}, imb={book_imbalance:.2f}, z_cap={z_skew_cap:.4f}, total_cap={effective_cap:.4f}")

    # Cap total adjustment if exceeds limit
    if abs(total_signal_adj) > effective_cap:
        # Scale both signals proportionally to stay within cap
        scale_factor = effective_cap / abs(total_signal_adj)
        imbalance_adj *= scale_factor
        z_skew_residual *= scale_factor
        total_signal_adj = imbalance_adj + z_skew_residual

        if VERBOSE:
            print(f"[MM] Signal cap applied: total={original_imbalance_adj + original_z_skew_residual:.4f} -> {total_signal_adj:.4f} "
                  f"(imb={original_imbalance_adj:.4f}->{imbalance_adj:.4f}, z_res={original_z_skew_residual:.4f}->{z_skew_residual:.4f})")

    # Apply combined signal adjustment
    fair_adj_yes += total_signal_adj

    # Store z-score skew for display and markout tracking
    # Store both capped z_skew (for display) and residual (for quoting)
    if not hasattr(global_state, 'z_skew_by_market'):
        global_state.z_skew_by_market = {}
    global_state.z_skew_by_market[market_id] = z_skew  # capped, for display

    if not hasattr(global_state, 'z_skew_residual_by_market'):
        global_state.z_skew_residual_by_market = {}
    global_state.z_skew_residual_by_market[market_id] = z_skew_residual  # capped residual, actually used

    # Store imbalance adjustment for markout tracking
    if not hasattr(global_state, 'imbalance_adj_by_market'):
        global_state.imbalance_adj_by_market = {}
    global_state.imbalance_adj_by_market[market_id] = imbalance_adj

    # Dynamic sizing: use larger size when z_skew and book_imbalance agree
    if z_skew * book_imbalance > 0:
        raw_size = ALIGNED_SIGNAL_SIZE
    else:
        raw_size = BASE_SIZE
    quote_size = int(max(5.0, raw_size))

    half_spread = quote_spread / 2.0
    raw_bid_yes = fair_adj_yes - half_spread
    raw_ask_yes = fair_adj_yes + half_spread

    # 1) theo-based targets (what you already do)
    target_bid_yes = max(MIN_PRICE, min(MAX_PRICE, tick_down(fair_adj_yes - half_spread, TICK_SIZE)))
    target_ask_yes = max(MIN_PRICE, min(MAX_PRICE, tick_up(fair_adj_yes + half_spread, TICK_SIZE)))

    # 2) back off from touch - asymmetric based on inventory
    # When building position: stay further from touch (adverse selection protection)
    # When reducing position: stay closer to touch (want to get filled)
    # BUY YES builds when net_yes >= 0, reduces when net_yes < 0
    # BUY NO builds when net_yes <= 0, reduces when net_yes > 0
    ticks_bid = MIN_TICKS_REDUCE if net_yes < 0 else MIN_TICKS_BUILD
    ticks_ask = MIN_TICKS_REDUCE if net_yes > 0 else MIN_TICKS_BUILD

    # In aggressive mode, only skip back-off on the side with predicted edge
    # z_score > 0 (bullish) -> allow bid to cross, keep ask backed off
    # z_score < 0 (bearish) -> allow ask to cross, keep bid backed off
    if aggressive_mode:
        if z_score > 0:
            # Bullish - skip bid back-off to allow crossing, keep ask backed off
            target_ask_yes = max(target_ask_yes, best_ask_yes + ticks_ask * TICK_SIZE)
        else:
            # Bearish - skip ask back-off to allow crossing, keep bid backed off
            target_bid_yes = min(target_bid_yes, best_bid_yes - ticks_bid * TICK_SIZE)
    else:
        # Normal mode - back off on both sides
        target_bid_yes = min(target_bid_yes, best_bid_yes - ticks_bid * TICK_SIZE)
        target_ask_yes = max(target_ask_yes, best_ask_yes + ticks_ask * TICK_SIZE)

    # 3) re-clip + re-tick (keep it clean)
    target_bid_yes = max(MIN_PRICE, min(MAX_PRICE, tick_down(target_bid_yes, TICK_SIZE)))
    target_ask_yes = max(MIN_PRICE, min(MAX_PRICE, tick_up(target_ask_yes, TICK_SIZE)))

    # 4) your "ask side" is BUY NO at 1 - (YES ask). Round NO bid down (never overpay)
    target_bid_no = max(MIN_PRICE, min(MAX_PRICE, tick_down(1.0 - target_ask_yes, TICK_SIZE)))

    edge_bid_yes = best_bid_yes - fair_adj_yes
    edge_ask_yes = fair_adj_yes - best_ask_yes
    if VERBOSE:
        print(f"[MM] edges: edge_bid_yes={edge_bid_yes:.4f}, edge_ask_yes={edge_ask_yes:.4f}")

    # Account for pending orders in the SAME direction to prevent position limit violations
    # - pending_yes > 0: pending BUY orders (will increase YES position) → reduce buy capacity
    # - pending_yes < 0: pending SELL orders (will decrease YES position) → reduce sell capacity
    # We DON'T subtract opposite-direction pending to avoid "phantom headroom" (orders that haven't filled yet)
    max_buy_yes  = max(0.0, effective_max_position - filled_yes - max(0, pending_yes))
    max_buy_no   = max(0.0, filled_yes + effective_max_position - max(0, -pending_yes))
    if VERBOSE:
        print(f"[MM] capacity: max_buy_yes={max_buy_yes}, max_buy_no={max_buy_no}")

    # ----- resting GTC path -----
    async def manage_side(side_key: str):
        nonlocal wo

        existing = wo[side_key]

        if side_key == "bid":
            desired_price = target_bid_yes
            max_size = max_buy_yes
            token_id = yes_token
            side_str = "BUY"
            edge = fair_adj_yes - target_bid_yes  # how much below fair we're bidding
        else:
            desired_price = target_bid_no
            max_size = max_buy_no
            token_id = no_token
            side_str = "BUY"
            edge = target_ask_yes - fair_adj_yes  # how much above fair we're asking
        if VERBOSE:
            print(f"[MM] manage_side {side_key}: desired_price={desired_price}, max_size={max_size}, edge={edge:.4f}, existing={existing}")

        # Don't quote if we don't have minimum edge
        if edge < MIN_EDGE_TO_QUOTE:
            if existing is not None:
                if VERBOSE:
                    print(f"[MM] manage_side {side_key}: cancel (no edge: {edge:.4f} < {MIN_EDGE_TO_QUOTE}) id={existing['id']}")
                if not existing.get("cancel_requested"):
                    existing["cancel_requested"] = True
                    existing["cancel_requested_at"] = time.time()
                    await cancel_order_async(existing["id"])
                wo[side_key] = None
            return

        if max_size <= 0:
            if existing is not None:
                if VERBOSE:
                    print(f"[MM] manage_side {side_key}: cancel (no capacity) id={existing['id']}")
                if not existing.get("cancel_requested"):
                    existing["cancel_requested"] = True
                    existing["cancel_requested_at"] = time.time()
                    await cancel_order_async(existing["id"])
                wo[side_key] = None
            return

        # Post-fill cooldown: don't place new order on this side if recently filled
        last_fill = getattr(global_state, "last_fill_time", {}).get(market_id, {}).get(side_key, 0.0)
        if now - last_fill < POST_FILL_COOLDOWN:
            if VERBOSE:
                print(f"[MM] manage_side {side_key}: in post-fill cooldown ({now - last_fill:.2f}s < {POST_FILL_COOLDOWN}s)")
            return

        # Use larger size in aggressive mode
        base_size = AGGRESSIVE_SIZE if aggressive_mode else quote_size
        size = min(base_size, max_size)

        if existing is not None:
            if abs(existing["price"] - desired_price) < PRICE_MOVE_TOL:
                if VERBOSE:
                    print(f"[MM] manage_side {side_key}: existing price close enough, doing nothing")
                return
            if VERBOSE:
                print(f"[MM] manage_side {side_key}: cancel stale {existing['id']} @ {existing['price']:.4f} -> {desired_price:.4f}")
            if not existing.get("cancel_requested"):
                existing["cancel_requested"] = True
                existing["cancel_requested_at"] = time.time()
                await cancel_order_async(existing["id"])
            wo[side_key] = None
            # CRITICAL: Skip this quote cycle after canceling to prevent race condition
            # If we place new order immediately, both orders could be live for 50-200ms
            # Next perform_trade() will place the new order with correct price
            return
        if VERBOSE:
            print(f"[MM] manage_side {side_key}: sending GTC {side_str} size={size} px={desired_price} token={token_id}")

        # Acquire lock to make order placement + wo[] update atomic
        # This prevents race where two manage_side() calls both place orders and overwrite wo[]
        async with global_state.position_check_lock:
            order_id = await _send_order_locked(
                token_id,
                side_str,
                desired_price,
                size,
                "GTC",
                market_id=market_id,
            )

            if order_id is None:
                if VERBOSE:
                    print(f"[MM] manage_side {side_key}: send_order returned None (throttled/error)")
                return

            # Update working orders dictionary inside lock to prevent race condition
            wo[side_key] = {
                "id": order_id,
                "price": desired_price,
                "size": size,
                "token": token_id,
                "side": side_str,
            }
        label = "YES" if token_id == yes_token else "NO"
        if VERBOSE:
            print(
                f"[MM] REST BUY {label} {size} @ {desired_price:.4f} "
                f"(fair_adj_yes={fair_adj_yes:.4f}, pos_yes={net_yes})"
            )

    # Run both sides in parallel for faster quote updates
    # Use Coinbase-RTDS z-score to skip vulnerable side when PRICE_SOURCE = "RTDS"
    price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')

    if price_source == "RTDS":
        z = getattr(global_state, 'coinbase_rtds_zscore', 0.0)
        z_skew_by_market = getattr(global_state, 'z_skew_by_market', {})
        z_skew = z_skew_by_market.get(market_id, 0.0)

        # Combined rule: Skip vulnerable side if EITHER:
        # 1) Z-score exceeds main threshold, OR
        # 2) Z-score exceeds combined threshold AND z-skew predicts significant option move
        should_skip_ask = (z > COINBASE_RTDS_ZSCORE_THRESHOLD) or \
                          (z > Z_SCORE_COMBINED_THRESHOLD and z_skew > Z_SKEW_COMBINED_THRESHOLD)
        should_skip_bid = (z < -COINBASE_RTDS_ZSCORE_THRESHOLD) or \
                          (z < -Z_SCORE_COMBINED_THRESHOLD and z_skew < -Z_SKEW_COMBINED_THRESHOLD)

        if should_skip_ask:
            # Coinbase HIGH → RTDS will rise → Cancel ASK if exists
            ask_order = wo.get("ask")
            if ask_order and isinstance(ask_order, dict) and ask_order.get("id"):
                if not ask_order.get("cancel_requested"):
                    reason = f"z={z:.2f}, z_skew={z_skew*100:.2f}¢"
                    print(f"[Z-SCORE] {reason} → Canceling ASK (RTDS will rise)")
                    ask_order["cancel_requested"] = True
                    ask_order["cancel_requested_at"] = time.time()
                    await cancel_order_async(ask_order["id"])
                    wo["ask"] = None
                    # Skip this cycle to prevent race (cancel needs time to process)
                    return

            # ASK already canceled or doesn't exist - quote BID only
            await manage_side("bid")

        elif should_skip_bid:
            # Coinbase LOW → RTDS will fall → Cancel BID if exists
            bid_order = wo.get("bid")
            if bid_order and isinstance(bid_order, dict) and bid_order.get("id"):
                if not bid_order.get("cancel_requested"):
                    reason = f"z={z:.2f}, z_skew={z_skew*100:.2f}¢"
                    print(f"[Z-SCORE] {reason} → Canceling BID (RTDS will fall)")
                    bid_order["cancel_requested"] = True
                    bid_order["cancel_requested_at"] = time.time()
                    await cancel_order_async(bid_order["id"])
                    wo["bid"] = None
                    # Skip this cycle to prevent race (cancel needs time to process)
                    return

            # BID already canceled or doesn't exist - quote ASK only
            await manage_side("ask")

        else:
            # Normal spread - quote both sides
            await asyncio.gather(
                manage_side("bid"),
                manage_side("ask")
            )
    else:
        # COINBASE or BLEND mode - always quote both sides
        await asyncio.gather(
            manage_side("bid"),
            manage_side("ask")
        )


def get_protected_mm_order_ids() -> set[str]:
    """
    Look at global_state.working_orders_by_market and collect the order IDs
    for the MM's actively-managed quotes (the 'bid'/'ask' entries used by manage_side).
    Does NOT modify any existing state.
    """
    protected = set()

    wom = getattr(global_state, "working_orders_by_market", {})

    for mkt_id, per_mkt in wom.items():
        if not isinstance(per_mkt, dict):
            continue

        # manage_side stores current quotes under 'bid' and 'ask'
        for side_key in ("bid", "ask"):
            entry = per_mkt.get(side_key)
            if isinstance(entry, dict):
                oid = entry.get("id")
                if oid:
                    protected.add(oid)

    return protected

async def sweep_zombie_orders_from_working():
    """
    Cancel orders that are in working_orders_by_market but are NOT the
    MM's actively-managed 'bid'/'ask' quotes.

    Uses ONLY local state (working_orders_by_market) and cancel_many_orders().
    Does not touch MM logic.
    """
    wom = getattr(global_state, "working_orders_by_market", {})

    if not isinstance(wom, dict) or not wom:
        return

    # 1) Find protected IDs: the current MM bid/ask that manage_side is using
    protected_ids = get_protected_mm_order_ids()
    if protected_ids and VERBOSE:
        print(f"[SWEEP] Protected MM order IDs (won't cancel): {protected_ids}")

    # 2) Collect zombie order IDs from working_orders_by_market
    zombie_ids = []

    for mkt_id, per_mkt in wom.items():
        if not isinstance(per_mkt, dict):
            continue

        for key, meta in per_mkt.items():
            # Skip the MM tracking slots themselves
            if key in ("bid", "ask"):
                continue

            if not isinstance(meta, dict):
                continue

            # Order id may be stored as 'order_id' or 'id' depending on your code
            oid = meta.get("order_id") or meta.get("id")
            if not oid:
                continue

            if oid in protected_ids:
                # Just in case, never cancel a protected quote
                continue

            status = (meta.get("status") or "").upper()
            # Only cancel live-ish orders
            if status in ("LIVE", "OPEN", "PARTIALLY_FILLED", "MATCHED"):
                zombie_ids.append(oid)

    if not zombie_ids:
        # Nothing to clean up
        return

    if VERBOSE:
        print(f"[SWEEP] Cancelling {len(zombie_ids)} zombie orders: {zombie_ids}")
    await cancel_many_orders(zombie_ids)