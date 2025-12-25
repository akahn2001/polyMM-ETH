"""
Market Scheduler for 15-minute BTC Up/Down Markets

Automatically discovers, schedules, and transitions between markets.
Captures strike from RTDS at market start time.
"""

import re
import ast
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import global_state

ET = ZoneInfo("America/New_York")


def parse_market_time(question: str) -> tuple[datetime, datetime] | None:
    """
    Parse start and end times from market name.
    Example: "Bitcoin Up or Down - December 25, 2:45PM-3:00PM ET"
    Returns (start_dt, end_dt) in ET timezone, or None if parse fails.
    """
    # Pattern: "Month Day, StartTime-EndTime ET"
    pattern = r"(\w+)\s+(\d+),\s*(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)\s*ET"
    match = re.search(pattern, question, re.IGNORECASE)

    if not match:
        return None

    month_str, day, start_hr, start_min, start_ampm, end_hr, end_min, end_ampm = match.groups()

    # Map month name to number
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    month = months.get(month_str.lower())
    if month is None:
        return None

    day = int(day)

    # Convert to 24-hour
    start_hr = int(start_hr)
    start_min = int(start_min)
    if start_ampm.upper() == 'PM' and start_hr != 12:
        start_hr += 12
    elif start_ampm.upper() == 'AM' and start_hr == 12:
        start_hr = 0

    end_hr = int(end_hr)
    end_min = int(end_min)
    if end_ampm.upper() == 'PM' and end_hr != 12:
        end_hr += 12
    elif end_ampm.upper() == 'AM' and end_hr == 12:
        end_hr = 0

    # Assume current year (or next year if month has passed)
    now = datetime.now(ET)
    year = now.year

    # If the month is in the past, assume next year
    if month < now.month or (month == now.month and day < now.day):
        year += 1

    try:
        start_dt = datetime(year, month, day, start_hr, start_min, tzinfo=ET)
        end_dt = datetime(year, month, day, end_hr, end_min, tzinfo=ET)

        # Handle overnight (end time < start time means next day)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return start_dt, end_dt
    except ValueError:
        return None


def parse_tokens(tokens_str: str) -> tuple[str, str] | None:
    """
    Parse tokens field to extract YES and NO token IDs.
    tokens_str is a string representation of a list of dicts.
    Returns (yes_token, no_token) or None.
    """
    try:
        tokens = ast.literal_eval(tokens_str)
        yes_token = None
        no_token = None

        for t in tokens:
            outcome = t.get('outcome', '').lower()
            token_id = t.get('token_id')
            if outcome == 'up' or outcome == 'yes':
                yes_token = token_id
            elif outcome == 'down' or outcome == 'no':
                no_token = token_id

        if yes_token and no_token:
            return yes_token, no_token
    except:
        pass

    return None


def load_btc_15min_markets(csv_path: str) -> list[dict]:
    """
    Load all BTC 15-minute up/down markets from CSV.
    Returns list of market dicts sorted by start time.
    """
    print(f"[SCHEDULER] Reading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"[SCHEDULER] CSV loaded, {len(df)} total rows")

    # Filter to Bitcoin up/down markets (have time pattern in name)
    print(f"[SCHEDULER] Filtering to BTC up/down markets...")
    btc_markets = df[
        df['question'].str.contains('Bitcoin Up or Down', na=False) &
        df['question'].str.contains(r'\d+:\d+[AP]M-\d+:\d+[AP]M', na=False, regex=True)
    ].copy()
    print(f"[SCHEDULER] Found {len(btc_markets)} BTC markets with time patterns")

    markets = []
    print(f"[SCHEDULER] Parsing {len(btc_markets)} markets...")

    for idx, (_, row) in enumerate(btc_markets.iterrows()):
        if idx % 1000 == 0:
            print(f"[SCHEDULER] Parsed {idx}/{len(btc_markets)} markets...")

        question = row['question']
        times = parse_market_time(question)

        if times is None:
            continue

        start_dt, end_dt = times

        # Filter to only 15-minute markets (duration between 14-16 minutes to allow for slight variations)
        duration_minutes = (end_dt - start_dt).total_seconds() / 60
        if duration_minutes < 14 or duration_minutes > 16:
            continue

        tokens = parse_tokens(str(row.get('tokens', '')))
        if tokens is None:
            continue

        yes_token, no_token = tokens

        markets.append({
            'question': question,
            'condition_id': row['condition_id'],
            'yes_token': yes_token,
            'no_token': no_token,
            'start_time': start_dt,
            'end_time': end_dt,
        })

    print(f"[SCHEDULER] Parsing complete. Found {len(markets)} 15-min markets")
    # Sort by start time
    markets.sort(key=lambda m: m['start_time'])

    return markets


def get_next_market(markets: list[dict], now: datetime = None) -> dict | None:
    """
    Get the next market that hasn't ended yet.
    Prefers markets that are currently active, otherwise the next upcoming one.
    """
    if now is None:
        now = datetime.now(ET)

    # First, check for currently active market
    for m in markets:
        if m['start_time'] <= now < m['end_time']:
            return m

    # Otherwise, find next upcoming market
    for m in markets:
        if m['start_time'] > now:
            return m

    return None


def get_markets_for_next_hours(markets: list[dict], hours: int = 24) -> list[dict]:
    """Get markets starting within the next N hours."""
    now = datetime.now(ET)
    cutoff = now + timedelta(hours=hours)

    return [m for m in markets if now <= m['start_time'] <= cutoff]


async def wait_until(target_time: datetime):
    """Sleep until the target time."""
    now = datetime.now(ET)
    delta = (target_time - now).total_seconds()

    if delta > 0:
        print(f"[SCHEDULER] Waiting {delta:.1f}s until {target_time.strftime('%I:%M:%S %p ET')}")
        await asyncio.sleep(delta)


async def capture_strike_from_rtds() -> float:
    """
    Capture current RTDS price as the strike.
    Waits for RTDS to be connected (up to 30 seconds).
    """
    # Wait for RTDS to be connected (up to 30 seconds)
    for i in range(60):
        if global_state.rtds_connected:
            break
        if i % 10 == 0:
            print(f"[SCHEDULER] Waiting for RTDS connection... ({i}/60)")
        await asyncio.sleep(0.5)

    if not global_state.rtds_connected:
        raise RuntimeError("RTDS not connected after 30 seconds - cannot capture strike")

    # Now get the RTDS price
    for _ in range(10):  # Try for up to 5 seconds
        rtds_price = getattr(global_state, 'mid_price', None)
        if rtds_price is not None and rtds_price > 0:
            print(f"[SCHEDULER] Captured strike from RTDS: ${rtds_price:.2f}")
            return rtds_price
        await asyncio.sleep(0.5)

    raise RuntimeError("RTDS connected but no price available")


def configure_market(market: dict, strike: float):
    """
    Configure global_state for trading this market.
    """
    print(f"[SCHEDULER] Configuring market: {market['question']}")
    print(f"[SCHEDULER] Strike: ${strike:.2f}")
    print(f"[SCHEDULER] YES token: {market['yes_token']}")
    print(f"[SCHEDULER] End time: {market['end_time'].strftime('%I:%M:%S %p ET')}")

    # Set strike and expiry
    global_state.strike = strike
    global_state.exp = market['end_time']

    # Set token mappings
    condition_id = market['condition_id']
    yes_token = market['yes_token']
    no_token = market['no_token']

    global_state.condition_to_token_id[condition_id] = yes_token
    global_state.token_to_condition_id[yes_token] = condition_id
    global_state.token_to_condition_id[no_token] = condition_id
    global_state.REVERSE_TOKENS[yes_token] = no_token
    global_state.REVERSE_TOKENS[no_token] = yes_token

    # Update all_tokens to point to current market's YES token
    global_state.all_tokens = [yes_token]
    # btc_markets should contain condition_ids (not token_ids) for fair value updates
    global_state.btc_markets = {condition_id}

    # Initialize position tracking
    if condition_id not in global_state.positions_by_market:
        global_state.positions_by_market[condition_id] = global_state.MarketPosition()

    # Initialize order tracking
    if not hasattr(global_state, 'working_orders_by_market'):
        global_state.working_orders_by_market = {}
    if condition_id not in global_state.working_orders_by_market:
        global_state.working_orders_by_market[condition_id] = {'bid': None, 'ask': None}

    # Set as active market
    global_state.active_market_id = condition_id

    print(f"[SCHEDULER] Market configured and ready to trade")


async def flatten_and_cancel(market_id: str):
    """
    Cancel all orders and optionally flatten position before market ends.
    """
    from trading import cancel_many_orders

    print(f"[SCHEDULER] Flattening market {market_id}")

    # Cancel all orders for this market
    wo = global_state.working_orders_by_market.get(market_id, {})
    order_ids = []

    for side in ['bid', 'ask']:
        entry = wo.get(side)
        if isinstance(entry, dict) and entry.get('id'):
            order_ids.append(entry['id'])

    if order_ids:
        await cancel_many_orders(order_ids)

    # Clear working orders
    global_state.working_orders_by_market[market_id] = {'bid': None, 'ask': None}

    print(f"[SCHEDULER] Cancelled {len(order_ids)} orders")


async def run_scheduler(csv_path: str, stop_before_end_seconds: int = 60, preloaded_markets: list = None):
    """
    Main scheduler loop.

    Args:
        csv_path: Path to markets CSV
        stop_before_end_seconds: Stop trading this many seconds before market ends
        preloaded_markets: Optional pre-loaded markets list to avoid re-parsing CSV
    """
    import traceback

    print(f"[SCHEDULER] Starting market scheduler")

    if preloaded_markets is not None:
        markets = preloaded_markets
        print(f"[SCHEDULER] Using {len(markets)} pre-loaded markets")
    else:
        print(f"[SCHEDULER] Loading markets from {csv_path}")
        try:
            markets = load_btc_15min_markets(csv_path)
        except Exception as e:
            print(f"[SCHEDULER] ERROR loading markets: {e}")
            traceback.print_exc()
            return
        print(f"[SCHEDULER] Found {len(markets)} BTC 15-min markets")

    if not markets:
        print(f"[SCHEDULER] ERROR: No markets found! Check CSV file and parsing.")
        return

    # Show upcoming markets
    now = datetime.now(ET)
    print(f"[SCHEDULER] Current time: {now.strftime('%Y-%m-%d %I:%M:%S %p ET')}")

    upcoming = get_markets_for_next_hours(markets, hours=2)
    print(f"[SCHEDULER] Next {len(upcoming)} markets in the next 2 hours:")
    for m in upcoming[:5]:
        print(f"  - {m['start_time'].strftime('%I:%M %p')} - {m['end_time'].strftime('%I:%M %p')}: {m['question'][:50]}...")

    if not upcoming:
        print(f"[SCHEDULER] WARNING: No markets in next 2 hours. Showing next 5 markets regardless of time:")
        for m in markets[:5]:
            print(f"  - {m['start_time'].strftime('%Y-%m-%d %I:%M %p')} - {m['end_time'].strftime('%I:%M %p')}: {m['question'][:50]}...")

    first_run = True
    while True:
        now = datetime.now(ET)
        print(f"[SCHEDULER] Looking for next market at {now.strftime('%I:%M:%S %p ET')}")
        market = get_next_market(markets, now)

        if market is None:
            print(f"[SCHEDULER] get_next_market returned None")
            print(f"[SCHEDULER] No more markets today. Reloading CSV in 1 hour...")
            await asyncio.sleep(3600)
            markets = load_btc_15min_markets(csv_path)

            # Update subscription tokens with any new markets
            old_tokens = set(global_state.all_subscription_tokens)
            new_tokens = [m['yes_token'] for m in markets]
            global_state.all_subscription_tokens = new_tokens

            # Register new token mappings
            for m in markets:
                global_state.condition_to_token_id[m['condition_id']] = m['yes_token']
                global_state.token_to_condition_id[m['yes_token']] = m['condition_id']
                global_state.token_to_condition_id[m['no_token']] = m['condition_id']
                global_state.REVERSE_TOKENS[m['yes_token']] = m['no_token']
                global_state.REVERSE_TOKENS[m['no_token']] = m['yes_token']

            # Check if we have new tokens that need subscription
            new_token_set = set(new_tokens)
            if new_token_set - old_tokens:
                print(f"[SCHEDULER] Found {len(new_token_set - old_tokens)} new tokens, signaling websocket reconnect")
                global_state.websocket_reconnect_needed = True

            print(f"[SCHEDULER] Reloaded {len(markets)} markets")
            continue

        print(f"[SCHEDULER] Found market: {market['question'][:50]}...")
        print(f"[SCHEDULER] Market starts at {market['start_time'].strftime('%I:%M:%S %p ET')}, ends at {market['end_time'].strftime('%I:%M:%S %p ET')}")

        # Wait for market start if not yet started
        if now < market['start_time']:
            print(f"[SCHEDULER] Market hasn't started yet, waiting...")
            await wait_until(market['start_time'])
        elif first_run:
            # Started mid-market - we don't know the true strike, skip to next
            print(f"[SCHEDULER] Started mid-market ({market['question'][:50]}...)")
            print(f"[SCHEDULER] Skipping - strike unknown. Waiting for next market...")
            remaining = (market['end_time'] - now).total_seconds()
            await asyncio.sleep(remaining + 5)
            first_run = False
            continue

        first_run = False

        # Capture strike from RTDS at market start
        try:
            strike = await capture_strike_from_rtds()
        except RuntimeError as e:
            print(f"[SCHEDULER] ERROR: {e}")
            print(f"[SCHEDULER] Skipping market {market['question'][:40]}...")
            # Wait for this market to end, then try next
            remaining = (market['end_time'] - datetime.now(ET)).total_seconds()
            if remaining > 0:
                await asyncio.sleep(remaining + 5)
            continue

        # Configure global_state for this market
        configure_market(market, strike)

        # Calculate when to stop trading
        stop_time = market['end_time'] - timedelta(seconds=stop_before_end_seconds)

        # Trade until stop time (unless dry_run mode)
        if global_state.dry_run:
            print(f"[SCHEDULER] DRY RUN - would trade until {stop_time.strftime('%I:%M:%S %p ET')}")
        else:
            print(f"[SCHEDULER] Trading until {stop_time.strftime('%I:%M:%S %p ET')}")
            global_state.trading_enabled = True

        while datetime.now(ET) < stop_time:
            await asyncio.sleep(1)

        # Stop trading and flatten
        print(f"[SCHEDULER] Stopping trading for {market['question'][:40]}...")
        global_state.trading_enabled = False

        await flatten_and_cancel(market['condition_id'])

        # Wait for market to actually end
        remaining = (market['end_time'] - datetime.now(ET)).total_seconds()
        if remaining > 0:
            print(f"[SCHEDULER] Waiting {remaining:.0f}s for market to end...")
            await asyncio.sleep(remaining + 5)  # Add 5s buffer

        print(f"[SCHEDULER] Market ended. Moving to next market...")


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python market_scheduler.py <csv_path>")
        print("\nExample: python market_scheduler.py all_markets_df_12_22.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    markets = load_btc_15min_markets(csv_path)

    print(f"Found {len(markets)} BTC 15-min markets\n")

    now = datetime.now(ET)
    upcoming = [m for m in markets if m['end_time'] > now]

    print(f"Upcoming markets ({len(upcoming)}):")
    for m in upcoming[:10]:
        print(f"  {m['start_time'].strftime('%b %d %I:%M %p')} - {m['end_time'].strftime('%I:%M %p')}")
        print(f"    condition_id: {m['condition_id']}")
        print(f"    yes_token: {m['yes_token'][:20]}...")
        print()
