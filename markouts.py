import os
import csv
import time
import asyncio
import global_state
from datetime import datetime
from zoneinfo import ZoneInfo

MARKOUT_HORIZONS = [1, 5, 15, 30, 60]          # seconds
MARKOUT_DUMP_INTERVAL = 1 * 60             # 10 minutes
MARKOUT_OUTDIR = "markouts"                 # folder to write into

def current_yes_mid(market_id: str) -> float | None:
    book = global_state.all_data.get(market_id)
    if not book:
        return None
    bids = book.get("bids")
    asks = book.get("asks")
    if not bids or not asks:
        return None
    best_bid = bids.peekitem(-1)[0]
    best_ask = asks.peekitem(0)[0]
    return 0.5 * (best_bid + best_ask)

async def markout_loop():
    # ensure list exists
    if not hasattr(global_state, "markouts"):
        global_state.markouts = []

    while True:
        now = time.time()
        fills = global_state.markouts

        for rec in fills:
            mkt = rec.get("market_id")
            ts  = rec.get("ts")
            if not mkt or ts is None:
                continue

            mid_yes = current_yes_mid(mkt)
            if mid_yes is None:
                continue

            age = now - ts
            done = rec.get("done")
            if done is None:
                rec["done"] = set()
                done = rec["done"]

            m = rec.get("m")
            if m is None:
                rec["m"] = {}
                m = rec["m"]

            fill_yes = float(rec["fill_yes"])
            qty      = float(rec["qty"])
            dir_yes  = int(rec["dir_yes"])

            for h in MARKOUT_HORIZONS:
                if h in done:
                    continue
                if age >= h:
                    # +dir_yes means you got longer YES; markout is move after fill in your favor
                    pnl = (mid_yes - fill_yes) * qty if dir_yes > 0 else (fill_yes - mid_yes) * qty
                    m[h] = pnl
                    done.add(h)

        # Clean up old fills to prevent memory leak
        # Remove fills older than 120s that have been written to CSV
        max_age = 120  # 2x the longest markout horizon (60s)
        before_len = len(global_state.markouts)
        global_state.markouts = [
            rec for rec in global_state.markouts
            if (now - rec.get("ts", now)) < max_age or not rec.get("written_to_csv", False)
        ]
        cleaned = before_len - len(global_state.markouts)
        # Only print cleanup messages in verbose mode
        # if cleaned > 0:
        #     print(f"[MARKOUT] Cleaned {cleaned} old fills from memory, {len(global_state.markouts)} remaining")

        await asyncio.sleep(0.25)

def record_fill(market_id, token_id, side, price, size, ts=None, order_type="GTC"):
    ts = ts or time.time()

    yes_token = global_state.condition_to_token_id[market_id]
    no_token  = global_state.REVERSE_TOKENS[yes_token]

    # map fill to YES price + direction
    if token_id == yes_token:
        fill_yes = float(price)
        dir_yes  = +1 if side.upper() == "BUY" else -1   # +1 means you got longer YES
    else:
        # token is NO
        fill_yes = 1.0 - float(price)
        dir_yes  = -1 if side.upper() == "BUY" else +1   # BUY NO makes you shorter YES

    # Capture diagnostic info at time of fill
    momentum = 0.0
    price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
    if price_source in ("COINBASE", "RTDS"):
        price_history = global_state.coinbase_price_history
    else:  # BLEND
        price_history = global_state.binance_price_history

    if len(price_history) >= 2:
        try:
            current_price = price_history[-1][1]
            for t, p in reversed(list(price_history)[:-1]):
                if time.time() - t >= 0.5:  # Match MOMENTUM_LOOKBACK (0.5s lookback)
                    momentum = current_price - p
                    break
        except:
            pass

    # Get current state
    position_before = global_state.positions_by_market.get(market_id)
    net_yes_before = position_before.net_yes if position_before else 0.0

    delta = global_state.binary_delta.get(market_id, 0.0)
    theo = global_state.fair_value.get(market_id, 0.0)
    binance_theo = global_state.binance_fair_value.get(market_id, 0.0)
    book_imbalance = getattr(global_state, "book_imbalance", {}).get(market_id, 0.0)

    # Volatility tracking
    implied_vol = global_state.fair_vol.get(market_id)
    realized_vol_5m = getattr(global_state, 'realized_vol_5m', None)
    realized_vol_15m = getattr(global_state, 'realized_vol_15m', None)

    # Vol edge = realized - implied (positive means market underpricing vol)
    vol_edge_5m = (realized_vol_5m - implied_vol) if (realized_vol_5m is not None and implied_vol is not None) else None
    vol_edge_15m = (realized_vol_15m - implied_vol) if (realized_vol_15m is not None and implied_vol is not None) else None

    # Calculate theo using realized vol instead of implied vol
    realized_vol_theo = None
    if price_source == "COINBASE":
        S_current = global_state.coinbase_mid_price
    elif price_source == "RTDS":
        S_current = global_state.mid_price
    else:  # BLEND
        S_current = global_state.blended_price

    if S_current is not None and realized_vol_15m is not None:
        from util import bs_binary_call
        K = global_state.strike
        T = (global_state.exp - datetime.now(ZoneInfo("America/New_York"))).total_seconds() / (60 * 60 * 24 * 365)
        if T > 0:
            try:
                realized_vol_theo = bs_binary_call(S_current, K, T, 0.0, realized_vol_15m, 0.0, 1.0)
            except:
                pass

    # Get market state
    book = global_state.all_data.get(market_id)
    market_mid = None
    if book:
        bids = book.get("bids")
        asks = book.get("asks")
        if bids and asks:
            best_bid = bids.peekitem(-1)[0]
            best_ask = asks.peekitem(0)[0]
            market_mid = 0.5 * (best_bid + best_ask)

    # Calculate fair_yes used for quoting (momentum-adjusted)
    fair_yes = market_mid if market_mid else 0.0

    # Reprice option with momentum (includes gamma, matching trading.py)
    if price_source == "COINBASE":
        S_current = global_state.coinbase_mid_price
    elif price_source == "RTDS":
        S_current = global_state.mid_price
    else:  # BLEND
        S_current = global_state.blended_price

    sigma = global_state.fair_vol.get(market_id)

    if S_current is not None and sigma is not None and momentum != 0:
        from util import bs_binary_call

        K = global_state.strike
        T = (global_state.exp - datetime.now(ZoneInfo("America/New_York"))).total_seconds() / (60 * 60 * 24 * 365)
        r = 0.0
        q = 0.0
        payoff = 1.0

        if T > 0:
            # Price option at current spot
            current_option_price = bs_binary_call(S_current, K, T, r, sigma, q, payoff)

            # Price option at spot + momentum
            S_after_momentum = S_current + momentum
            new_option_price = bs_binary_call(S_after_momentum, K, T, r, sigma, q, payoff)

            # Predicted move (includes gamma!)
            predicted_option_move = new_option_price - current_option_price

            # Cap adjustment (match trading.py MAX_MOMENTUM_ADJUSTMENT)
            predicted_option_move = max(-0.03, min(0.03, predicted_option_move))

            fair_yes = fair_yes + predicted_option_move

    # Calculate edge metrics
    # For buys (dir_yes=+1): positive edge = bought below value
    # For sells (dir_yes=-1): positive edge = sold above value
    edge_vs_theo = (theo - fill_yes) if dir_yes > 0 else (fill_yes - theo)
    edge_vs_fair = (fair_yes - fill_yes) if dir_yes > 0 else (fill_yes - fair_yes)
    model_vs_market = theo - market_mid if market_mid else 0.0

    # Calculate momentum volatility (same logic as trading.py)
    momentum_volatility = 0.0
    if price_source in ("COINBASE", "RTDS"):
        price_history_vol = global_state.coinbase_price_history
    else:  # BLEND
        price_history_vol = global_state.binance_price_history

    if len(price_history_vol) >= 5:
        recent_momentums = []
        prices = price_history_vol
        recent_prices = [(t, p) for t, p in prices if time.time() - t <= 5.0]
        if len(recent_prices) >= 3:
            for i in range(len(recent_prices) - 1):
                t1, p1 = recent_prices[i]
                t2, p2 = recent_prices[i + 1]
                if t2 - t1 > 0.1:
                    recent_momentums.append(p2 - p1)
            if len(recent_momentums) >= 2:
                try:
                    import numpy as np
                    momentum_volatility = np.std(recent_momentums)
                except:
                    pass

    global_state.markouts.append({
        "ts": ts,
        "market_id": market_id,
        "fill_yes": fill_yes,
        "dir_yes": dir_yes,
        "qty": float(size),
        "done": set(),   # horizons already computed
        "m": {},         # horizon -> markout pnl
        # Diagnostic fields
        "momentum": momentum,  # Price momentum (Coinbase or Binance based on config)
        "momentum_volatility": momentum_volatility,  # How choppy is price?
        "delta": delta,
        "theo": theo,
        "binance_theo": binance_theo,
        "market_mid": market_mid,
        "fair_yes": fair_yes,  # What we actually quoted around
        "edge_vs_theo": edge_vs_theo,  # Did we get edge vs model?
        "edge_vs_fair": edge_vs_fair,  # Did we get filled better than target?
        "model_vs_market": model_vs_market,  # Model disagreement with market
        "net_yes_before": net_yes_before,
        "side": side.upper(),
        "token_type": "YES" if token_id == yes_token else "NO",
        "order_type": order_type,  # "GTC" for maker, "IOC" for taker
        "book_imbalance": book_imbalance,  # Order book imbalance at time of fill
        # Volatility edge tracking
        "implied_vol": implied_vol,
        "realized_vol_5m": realized_vol_5m,
        "realized_vol_15m": realized_vol_15m,
        "vol_edge_5m": vol_edge_5m,  # realized_5m - implied (positive = market underpricing vol)
        "vol_edge_15m": vol_edge_15m,  # realized_15m - implied
        "realized_vol_theo": realized_vol_theo,  # theo priced with realized vol instead of implied
    })

def _pct(vals, p):
    if not vals:
        return None
    vs = sorted(vals)
    k = int(round((p/100.0) * (len(vs)-1)))
    return vs[max(0, min(len(vs)-1, k))]

def _summarize_pnls(pnls):
    if not pnls:
        return None
    vs = sorted(pnls)
    n = len(vs)
    avg = sum(vs) / n
    med = vs[n//2]
    hit = sum(1 for v in vs if v > 0) / n
    return {
        "n": n,
        "avg_pnl": avg,
        "median_pnl": med,
        "hit_rate": hit,
        "p10": _pct(vs, 10),
        "p90": _pct(vs, 90),
        "total_pnl": sum(vs),
    }

def _collect_markouts():
    """
    Returns:
      overall[h] -> list[pnl]
      by_market[(market_id, h)] -> list[pnl]
    Expects global_state.markouts entries like:
      {"market_id": ..., "m": {h: pnl, ...}}
    """
    overall = {h: [] for h in MARKOUT_HORIZONS}
    by_market = {}  # (market_id, h) -> list[pnl]

    for rec in getattr(global_state, "markouts", []) or []:
        mkt = rec.get("market_id")
        m = rec.get("m", {}) or {}
        for h in MARKOUT_HORIZONS:
            if h in m:
                pnl = float(m[h])
                overall[h].append(pnl)
                by_market.setdefault((mkt, h), []).append(pnl)

    return overall, by_market

def _write_summary_csvs(ts_label: str):
    os.makedirs(MARKOUT_OUTDIR, exist_ok=True)

    overall, by_market = _collect_markouts()

    # --- overall file ---
    overall_path = os.path.join(MARKOUT_OUTDIR, f"markout_summary_overall_{ts_label}.csv")
    with open(overall_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["timestamp", "horizon_s", "n", "avg_pnl", "median_pnl", "hit_rate", "p10", "p90", "total_pnl"]
        )
        w.writeheader()
        for h in MARKOUT_HORIZONS:
            s = _summarize_pnls(overall[h]) or {"n": 0, "avg_pnl": None, "median_pnl": None, "hit_rate": None, "p10": None, "p90": None, "total_pnl": 0.0}
            w.writerow({
                "timestamp": ts_label,
                "horizon_s": h,
                **s
            })

    # --- by-market file ---
    bym_path = os.path.join(MARKOUT_OUTDIR, f"markout_summary_by_market_{ts_label}.csv")
    with open(bym_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["timestamp", "market_id", "horizon_s", "n", "avg_pnl", "median_pnl", "hit_rate", "p10", "p90", "total_pnl"]
        )
        w.writeheader()
        # stable output order
        for (mkt, h) in sorted(by_market.keys(), key=lambda x: (str(x[0]), x[1])):
            s = _summarize_pnls(by_market[(mkt, h)])
            if not s:
                continue
            w.writerow({
                "timestamp": ts_label,
                "market_id": mkt,
                "horizon_s": h,
                **s
            })

    print(f"[MARKOUT] wrote {overall_path} and {bym_path}")

def _write_detailed_fills_csv():
    """
    Write NEW fills (not yet written) with full diagnostic info to a CSV.
    Only writes fills that have been marked as not yet written.
    """
    os.makedirs(MARKOUT_OUTDIR, exist_ok=True)
    detailed_path = os.path.join(MARKOUT_OUTDIR, "detailed_fills.csv")

    fills = getattr(global_state, "markouts", []) or []
    if not fills:
        return

    # Check if file exists to determine if we need header
    file_exists = os.path.exists(detailed_path)

    # Count new fills to write
    new_fills_count = 0

    with open(detailed_path, "a", newline="") as f:
        fieldnames = [
            "timestamp", "order_type", "fill_yes", "dir_yes", "side", "token_type", "qty",
            "momentum", "momentum_volatility", "delta",
            "theo", "binance_theo", "market_mid", "fair_yes",
            "edge_vs_theo", "edge_vs_fair", "model_vs_market",
            "net_yes_before", "net_yes_after", "book_imbalance",
            "implied_vol", "realized_vol_5m", "realized_vol_15m", "vol_edge_5m", "vol_edge_15m", "realized_vol_theo",
            "markout_1s", "markout_5s", "markout_15s", "markout_30s", "markout_60s"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            w.writeheader()

        for rec in fills:
            # Skip if already written
            if rec.get("written_to_csv"):
                continue

            # Only write fills that have at least 1s markout computed
            m = rec.get("m", {})
            if 1 not in m:
                continue

            # Calculate net_yes after this fill
            net_after = rec.get("net_yes_before", 0.0) + (rec.get("dir_yes", 0) * rec.get("qty", 0))

            ts_str = datetime.fromtimestamp(rec.get("ts", 0), ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")

            w.writerow({
                "timestamp": ts_str,
                "order_type": rec.get("order_type", "GTC"),
                "fill_yes": rec.get("fill_yes"),
                "dir_yes": rec.get("dir_yes"),
                "side": rec.get("side"),
                "token_type": rec.get("token_type"),
                "qty": rec.get("qty"),
                "momentum": rec.get("momentum"),
                "momentum_volatility": rec.get("momentum_volatility"),
                "delta": rec.get("delta"),
                "theo": rec.get("theo"),
                "binance_theo": rec.get("binance_theo"),
                "market_mid": rec.get("market_mid"),
                "fair_yes": rec.get("fair_yes"),
                "edge_vs_theo": rec.get("edge_vs_theo"),
                "edge_vs_fair": rec.get("edge_vs_fair"),
                "model_vs_market": rec.get("model_vs_market"),
                "net_yes_before": rec.get("net_yes_before"),
                "net_yes_after": net_after,
                "book_imbalance": rec.get("book_imbalance"),
                "implied_vol": rec.get("implied_vol"),
                "realized_vol_5m": rec.get("realized_vol_5m"),
                "realized_vol_15m": rec.get("realized_vol_15m"),
                "vol_edge_5m": rec.get("vol_edge_5m"),
                "vol_edge_15m": rec.get("vol_edge_15m"),
                "realized_vol_theo": rec.get("realized_vol_theo"),
                "markout_1s": m.get(1),
                "markout_5s": m.get(5),
                "markout_15s": m.get(15),
                "markout_30s": m.get(30),
                "markout_60s": m.get(60),
            })

            # Mark as written
            rec["written_to_csv"] = True
            new_fills_count += 1

    if new_fills_count > 0:
        print(f"[MARKOUT] Wrote {new_fills_count} new fills to CSV")

async def markout_dump_loop():
    """
    Every 10 minutes, dump markout summary CSVs.
    """
    while True:
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            ts_label = now.strftime("%Y%m%d_%H%M")
            _write_summary_csvs(ts_label)
            _write_detailed_fills_csv()  # Also write detailed fills
            print(f"[MARKOUT] Wrote summary and detailed fills CSV")
        except Exception as e:
            print(f"[MARKOUT][DUMP] error: {e}")
        await asyncio.sleep(MARKOUT_DUMP_INTERVAL)