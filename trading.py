import global_state
import asyncio
import time
import math
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from markouts import record_fill
from global_state import MarketPosition
from py_clob_client.exceptions import PolyApiException
from py_clob_client.clob_types import OpenOrderParams
from util import bs_binary_call

# TODO: 12/17 WE STILL AREN'T CANCELLING ORDERS CORRECTLY- currently using print statements to debug the open_orders from api. vs. protected orders set logic
# TODO: My suspicion is that we are somehow protecting too many orders and this is leaving stale orders in the book

RECONCILE_INTERVAL = 0.50  # seconds (start here; tighten later if stable)
RECONCILE_MAX_CANCEL_BATCH = 50  # keep batches mo  dest to avoid Cloudflare pain

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
    for side_key in ("bid", "ask"):
        entry = wo.get(side_key)
        if isinstance(entry, dict):
            oid = entry.get("id")
            if oid and oid not in open_ids:
                wo[side_key] = None

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

IOC_STALE_THRESHOLD = 30.0  # seconds - clean up IOC deltas older than this

def cleanup_stale_ioc_deltas():
    """
    Remove IOC order deltas that are older than IOC_STALE_THRESHOLD seconds.
    This is a safety net in case websocket misses TRADE/ORDER events.
    """
    if not hasattr(global_state, "ioc_order_deltas"):
        return

    now = time.time()
    stale_orders = []

    for order_id, (market_id, delta, ts) in list(global_state.ioc_order_deltas.items()):
        age = now - ts
        if age > IOC_STALE_THRESHOLD:
            stale_orders.append((order_id, market_id, delta, age))

    for order_id, market_id, delta, age in stale_orders:
        global_state.ioc_order_deltas.pop(order_id, None)
        if hasattr(global_state, "pending_order_delta") and market_id in global_state.pending_order_delta:
            global_state.pending_order_delta[market_id] -= delta
            if VERBOSE:
                print(f"[IOC STALE CLEANUP] Removed stale IOC delta {delta:.1f} for order {order_id} (age={age:.1f}s), new pending_delta={global_state.pending_order_delta[market_id]:.1f}")

async def reconcile_loop_all():
    while True:
        t0 = time.time()
        try:
            # Clean up any stale IOC deltas (safety net for missed websocket events)
            cleanup_stale_ioc_deltas()

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
    """
    yes_token = global_state.condition_to_token_id[market_id]
    no_token  = global_state.REVERSE_TOKENS[yes_token]

    pending = 0.0

    for o in open_orders:
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
EDGE_TAKE_THRESHOLD = 0.050      # edge to justify crossing #.0275, was .04 12/19 night
TAKER_SIZE_MULT = 1.0
BASE_QUOTE_SPREAD = 0.035             # desired total spread # was .03 morning of 12/19, was .03 12/19 night
MAX_POSITION = 10
BASE_SIZE = 5.0
#INV_SKEW_PER_SHARE = 0.00050

SKEW_K = .60          # 0.3–1.0, start ~0.6
SKEW_CAP = 0.04       # max skew in price points (5c)

MIN_PRICE = 0.01
MAX_PRICE = 0.99
PRICE_MOVE_TOL = 0.0020          # don’t cancel/replace if existing quote is within 0.5c of target
TICK_SIZE = .01
MIN_TICKS_FROM_TOUCH = 0   # start with 2; try 1–3

MIN_ORDER_INTERVAL = .50  # seconds → max 5 orders/sec per market+side, # changed this back to 1

# Binance momentum adjustment
USE_BINANCE_MOMENTUM = False  # Toggle to use Binance momentum for predictive quoting
BINANCE_MOMENTUM_LOOKBACK = 0.5  # Seconds to look back for momentum calculation
MAX_MOMENTUM_ADJUSTMENT = 0.03  # Max price adjustment from momentum (caps at 3 cents)

# Dynamic spread based on momentum volatility
USE_DYNAMIC_SPREAD = True  # Toggle to widen spread when momentum is volatile
MOMENTUM_VOLATILITY_WINDOW = 5.0  # Seconds to measure momentum volatility
MAX_VOLATILITY_SPREAD_MULT = 4.0  # Max spread multiplier (e.g., 3x wider)

VERBOSE = False


async def send_order(token_id: str, side: str, price: float, size: float, tif: str, *, market_id: str | None = None):
    """
    Centralized order sender with:
      - global trading_enabled check
      - per (key, side, tif) throttle  (key = market_id if given else token_id)
      - DRY RUN mode
      - error handling (incl. 403)

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

    # Start new blocker

    if market_id is None:
        # If you ever call send_order without market_id, safest is to block
        #print("[RISK] send_order called without market_id; blocking to enforce MAX_POSITION.")
        return None

    try:
        yes_token = global_state.condition_to_token_id[market_id]
        no_token = global_state.REVERSE_TOKENS[yes_token]

        # filled net YES
        pos_obj = global_state.positions_by_market.get(market_id)
        filled_yes = pos_obj.net_yes if pos_obj is not None else 0.0

        # pending net YES from working_orders_by_market (GTC orders)
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

            if tok == yes_token:
                pending_yes += qty if sde == "BUY" else -qty
            elif tok == no_token:
                pending_yes += -qty if sde == "BUY" else +qty

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

    now = time.time()  # Cache time once for this call
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

    # Blend theo (75%) with book mid (25%) for quoting
    # Theo reacts to Binance, book mid anchors to market reality
    theo = info["fair"]  # global_state.fair_value[market_id]
    fair_yes = 0.75 * theo + 0.25 * book_mid

    # Add Binance momentum adjustment if enabled (skip entirely when disabled for speed)
    if not USE_BINANCE_MOMENTUM:
        pass  # Skip momentum calculation entirely
    elif USE_BINANCE_MOMENTUM:
        binance_momentum = 0.0
        if hasattr(global_state, 'binance_price_history') and len(global_state.binance_price_history) >= 2:
            current_binance = global_state.binance_price_history[-1][1]

            # Find price from BINANCE_MOMENTUM_LOOKBACK seconds ago
            for ts, price in reversed(list(global_state.binance_price_history)[:-1]):
                if now - ts >= BINANCE_MOMENTUM_LOOKBACK:
                    old_binance = price
                    binance_momentum = current_binance - old_binance
                    break

        # Reprice option with momentum-adjusted spot (includes gamma!)
        # Get current blended price and option parameters
        S_current = global_state.blended_price
        sigma = global_state.fair_vol.get(market_id)

        if S_current is not None and sigma is not None and binance_momentum != 0:
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
                S_after_momentum = S_current + binance_momentum
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

    abs_delta = abs(getattr(global_state, "binary_delta", {}).get(market_id, 0.0))
    # normalize: probability points per 1% BTC move (more stable than per $)
    #delta_risk = abs_delta * (global_state.mid_price or 0.0)

    # simple capped multiplier
    mult = 1.0 + min(2.0, abs_delta)  # tune numbers later
    mult = 1

    # Calculate momentum volatility multiplier
    volatility_mult = 1.0
    current_momentum_mult = 1.0
    momentum_adjustment_mult = 1.0  # Extra widening if using momentum adjustment

    if USE_DYNAMIC_SPREAD and hasattr(global_state, 'binance_price_history') and len(global_state.binance_price_history) >= 5:
        # Get current momentum (may not be set if USE_BINANCE_MOMENTUM=False)
        current_momentum = 0.0
        if len(global_state.binance_price_history) >= 2:
            current_binance = global_state.binance_price_history[-1][1]
            for ts, price in reversed(list(global_state.binance_price_history)[:-1]):
                if now - ts >= BINANCE_MOMENTUM_LOOKBACK:
                    current_momentum = abs(current_binance - price)
                    break

        # Widen for LARGE CURRENT MOMENTUM (smooth big moves)
        if current_momentum > 3:  # $3+ move (very sensitive!)
            current_momentum_mult = 1.0 + min(3.0, (current_momentum - 3) / 12.0)
            # Examples:
            # $3 momentum = 1.0x (no widening)
            # $5 momentum = 1.17x (wider!)
            # $9 momentum = 1.5x
            # $15+ momentum = 2.0x (max 4x total)

        # Calculate recent momentum values for VOLATILITY (choppiness)
        recent_momentums = []
        prices = global_state.binance_price_history

        # Get prices within volatility window
        recent_prices = [(t, p) for t, p in prices if now - t <= MOMENTUM_VOLATILITY_WINDOW]

        if len(recent_prices) >= 3:
            # Calculate momentum at multiple points to measure volatility
            for i in range(len(recent_prices) - 1):
                t1, p1 = recent_prices[i]
                t2, p2 = recent_prices[i + 1]
                if t2 - t1 > 0.1:  # At least 0.1s apart
                    momentum = p2 - p1
                    recent_momentums.append(momentum)

            if len(recent_momentums) >= 2:
                # Calculate std dev of momentum changes (volatility)
                momentum_std = np.std(recent_momentums)

                # Scale volatility to spread multiplier (CHOPPINESS)
                # High volatility (>$1 std) = wider spread
                # Low volatility (<$1 std) = normal spread
                if momentum_std > 1:  # Much more sensitive! (was 5)
                    volatility_mult = 1.0 + min(MAX_VOLATILITY_SPREAD_MULT - 1.0, (momentum_std - 1) / 10.0)
                    # Examples:
                    # $1 std = 1.0x (no widening)
                    # $6 std = 1.5x
                    # $11+ std = 2.0x (max)

                    # Debug output when spread widens significantly
                    if volatility_mult > 1.5 and VERBOSE:
                        print(f"[MM] High momentum volatility: std=${momentum_std:.2f}, widening spread by {volatility_mult:.2f}x")

        # Use the LARGER of the two multipliers (protect against both types of risk)
        if current_momentum_mult > 1.5 and VERBOSE:
            print(f"[MM] Large current momentum: ${current_momentum:.2f}, widening spread by {current_momentum_mult:.2f}x")

    # If using momentum adjustment, widen extra to compensate for VPN lag
    if USE_BINANCE_MOMENTUM:
        # Always widen 1.5x minimum when momentum strategy is active
        # This protects from VPN lag during quote adjustments
        momentum_adjustment_mult = 1.5

    # Apply the larger of volatility, current momentum, or momentum adjustment multiplier
    dynamic_mult = max(volatility_mult, current_momentum_mult, momentum_adjustment_mult)
    quote_spread = BASE_QUOTE_SPREAD * mult * mult_warm * dynamic_mult

    raw_size = BASE_SIZE / mult
    raw_size = BASE_SIZE
    # enforce minimum order size
    quote_size = int(max(5.0, raw_size))

    # --- dynamic inventory skew (scaled to spread, capped, stronger when "dangerous") ---
    # half spread should be the (possibly delta-adjusted) one you actually plan to quote
    half_spread = quote_spread / 2.0  # if you compute quote_spread dynamically
    # half_spread = QUOTE_SPREAD / 2.0 # if not

    inv_ratio = 0.0
    if MAX_POSITION > 0:
        inv_ratio = max(-1.0, min(1.0, net_yes / MAX_POSITION))  # net_yes = effective yes (filled+pending)

    skew = -inv_ratio * (SKEW_K * half_spread)  # long YES -> skew down, short YES -> skew up
    skew *= mult  # make skew stronger when market is dangerous (high delta / near expiry)
    skew = max(-SKEW_CAP, min(SKEW_CAP, skew))  # hard cap

    fair_adj_yes = fair_yes + skew

    half_spread = quote_spread / 2.0
    raw_bid_yes = fair_adj_yes - half_spread
    raw_ask_yes = fair_adj_yes + half_spread

    # 1) theo-based targets (what you already do)
    target_bid_yes = max(MIN_PRICE, min(MAX_PRICE, tick_down(fair_adj_yes - half_spread, TICK_SIZE)))
    target_ask_yes = max(MIN_PRICE, min(MAX_PRICE, tick_up(fair_adj_yes + half_spread, TICK_SIZE)))

    # 2) back off from touch (new)
    # bid: don't be closer than N ticks to the best bid
    target_bid_yes = min(target_bid_yes, best_bid_yes - MIN_TICKS_FROM_TOUCH * TICK_SIZE)

    # ask: don't be closer than N ticks to the best ask
    target_ask_yes = max(target_ask_yes, best_ask_yes + MIN_TICKS_FROM_TOUCH * TICK_SIZE)

    # 3) re-clip + re-tick (keep it clean)
    target_bid_yes = max(MIN_PRICE, min(MAX_PRICE, tick_down(target_bid_yes, TICK_SIZE)))
    target_ask_yes = max(MIN_PRICE, min(MAX_PRICE, tick_up(target_ask_yes, TICK_SIZE)))

    # 4) your "ask side" is BUY NO at 1 - (YES ask). Round NO bid down (never overpay)
    target_bid_no = max(MIN_PRICE, min(MAX_PRICE, tick_down(1.0 - target_ask_yes, TICK_SIZE)))

    edge_bid_yes = best_bid_yes - fair_adj_yes
    edge_ask_yes = fair_adj_yes - best_ask_yes
    if VERBOSE:
        print(f"[MM] edges: edge_bid_yes={edge_bid_yes:.4f}, edge_ask_yes={edge_ask_yes:.4f}")

    max_buy_yes  = max(0.0, MAX_POSITION - net_yes)
    max_buy_no   = max(0.0, net_yes + MAX_POSITION)
    if VERBOSE:
        print(f"[MM] capacity: max_buy_yes={max_buy_yes}, max_buy_no={max_buy_no}")

    # ----- aggressive IOC path -----
    if edge_ask_yes > EDGE_TAKE_THRESHOLD and max_buy_yes > 0:
        size = min(int((quote_size*TAKER_SIZE_MULT)), max_buy_yes)
        if VERBOSE:
            print(f"[MM] TAKE BUY YES: size={size}, px={best_ask_yes}")
        await send_order(
            yes_token,
            "BUY",
            best_ask_yes,
            size,
            "IOC",
            market_id=market_id,
        )

    if edge_bid_yes > EDGE_TAKE_THRESHOLD and max_buy_no > 0:
        size = min(int((quote_size*TAKER_SIZE_MULT)), max_buy_no)
        price_no = yes_to_no_price(best_bid_yes)
        if VERBOSE:
            print(f"[MM] TAKE BUY NO: size={size}, px={price_no}")
        await send_order(
            no_token,
            "BUY",
            price_no,
            size,
            "IOC",
            market_id=market_id,
        )

    # ----- resting GTC path -----
    async def manage_side(side_key: str):
        nonlocal wo

        existing = wo[side_key]

        if side_key == "bid":
            desired_price = target_bid_yes
            max_size = max_buy_yes
            token_id = yes_token
            side_str = "BUY"
        else:
            desired_price = target_bid_no
            max_size = max_buy_no
            token_id = no_token
            side_str = "BUY"
        if VERBOSE:
            print(f"[MM] manage_side {side_key}: desired_price={desired_price}, max_size={max_size}, existing={existing}")

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

        size = min(quote_size, max_size)

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
        if VERBOSE:
            print(f"[MM] manage_side {side_key}: sending GTC {side_str} size={size} px={desired_price} token={token_id}")
        order_id = await send_order(
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

    await manage_side("bid")
    await manage_side("ask")


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