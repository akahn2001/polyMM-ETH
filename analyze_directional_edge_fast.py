"""
Binance -> RTDS latency/edge test (event-based, leakage-resistant).

Question:
  When Binance makes a big move (signal), does RTDS subsequently move in the same direction
  within MAX_WAIT_TIME seconds?

Key properties:
  - Denominator = number of Binance signals
  - Avoids inflated success from RTDS sparse updates + overlapping credit
  - Requires RTDS to be "flat" into the signal
  - Uses first-crossing after signal (not max-in-window)
  - Consumes RTDS events so one RTDS jump can't satisfy many signals
"""

import pandas as pd
import numpy as np
from datetime import timedelta

try:
    import global_state
except ImportError:
    global_state = None

# -----------------------------
# Config
# -----------------------------
CSV_FILE = "price_lag_data.csv"

SAMPLE_INTERVAL_SECONDS = 1

# Binance signal definition
SIGNAL_WINDOW_S = 10            # lookback window to define "big move burst"
BINANCE_MOVE_THRESHOLD = 25      # dollars of move over SIGNAL_WINDOW_S window
SIGNAL_COOLDOWN_S = 10           # minimum spacing between signals (burst dedupe)

# RTDS confirmation definition
RTDS_MOVE_THRESHOLD = 20         # dollars move required to count as "follow"
MAX_WAIT_TIME = 5               # seconds after signal to wait for RTDS confirmation

# Leakage guards
PRE_FLAT_S = 2                   # RTDS must be flat in the PRE_FLAT_S seconds BEFORE signal
RTDS_BASE_MAX_STALENESS_S = 5    # baseline RTDS price must be at most this old at signal time

# Null test
RUN_PERMUTATION_TEST = False
N_PERMUTATIONS = 200


# -----------------------------
# Loading / prep
# -----------------------------
def load_and_prepare_data(csv_file: str):
    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["event_time_iso"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)

    usdtusd = 0.999425
    if global_state is not None and hasattr(global_state, "usdtusd"):
        usdtusd = float(global_state.usdtusd)
        print(f"Using USDT/USD rate: {usdtusd:.6f}")
    else:
        print("WARNING: global_state.usdtusd not available. Using hardcoded rate.")

    df.loc[df["source"] == "binance", "price"] = df.loc[df["source"] == "binance", "price"] * usdtusd

    binance_df = df[df["source"] == "binance"][["timestamp", "price"]].copy()
    binance_df.columns = ["timestamp", "binance_price"]

    rtds_df = df[df["source"] == "polymarket"][["timestamp", "price"]].copy()
    rtds_df.columns = ["timestamp", "rtds_price"]

    # Resample Binance to regular cadence
    binance_resampled = (
        binance_df.set_index("timestamp")
        .resample(f"{SAMPLE_INTERVAL_SECONDS}s")["binance_price"]
        .last()
        .dropna()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    rtds_df = rtds_df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(binance_df)} Binance ticks -> {len(binance_resampled)} resampled points")
    print(f"Loaded {len(rtds_df)} RTDS updates")
    return binance_resampled, rtds_df


# -----------------------------
# Binance signals (burst-based)
# -----------------------------
def make_binance_signals(binance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Define a Binance 'signal' at time t if:
      net_change over last SIGNAL_WINDOW_S seconds has abs >= BINANCE_MOVE_THRESHOLD.

    Then dedupe using SIGNAL_COOLDOWN_S (one signal per burst).
    """
    b = binance_df.copy().sort_values("timestamp").reset_index(drop=True)
    b = b.set_index("timestamp")

    # Rolling start value (first in window), then net change
    roll_start = b["binance_price"].rolling(f"{SIGNAL_WINDOW_S}s").apply(lambda x: x.iloc[0], raw=False)
    b["net_change"] = b["binance_price"] - roll_start
    b = b.dropna().reset_index()

    b["is_signal"] = b["net_change"].abs() >= BINANCE_MOVE_THRESHOLD
    b["direction"] = np.where(b["net_change"] >= 0, "up", "down")

    signals = []
    last_signal_time = None

    for _, row in b[b["is_signal"]].iterrows():
        t = row["timestamp"]
        if last_signal_time is not None:
            if (t - last_signal_time).total_seconds() < SIGNAL_COOLDOWN_S:
                continue
        signals.append(
            {
                "signal_time": t,
                "signal_direction": row["direction"],
                "binance_net_change": float(row["net_change"]),
                "binance_price": float(row["binance_price"]),
            }
        )
        last_signal_time = t

    return pd.DataFrame(signals)


# -----------------------------
# RTDS helpers
# -----------------------------
def _rtds_baseline_at_or_before(rtds_df: pd.DataFrame, t0: pd.Timestamp):
    """
    Return (base_idx, base_time, base_price) where base_time <= t0 is the last RTDS update.
    Return None if no baseline exists.
    """
    # rtds_df must be sorted by timestamp
    idx = np.searchsorted(rtds_df["timestamp"].values, np.array(t0.to_datetime64()), side="right") - 1
    if idx < 0:
        return None
    base_time = rtds_df.iloc[idx]["timestamp"]
    base_price = float(rtds_df.iloc[idx]["rtds_price"])
    return int(idx), base_time, base_price


def _rtds_was_flat_before(rtds_df: pd.DataFrame, t0: pd.Timestamp, base_price: float) -> bool:
    """
    Require RTDS to be "flat" into the signal:
      within (t0 - PRE_FLAT_S, t0], RTDS should not have moved >= RTDS_MOVE_THRESHOLD from base_price.
    """
    t_start = t0 - timedelta(seconds=PRE_FLAT_S)
    window = rtds_df[(rtds_df["timestamp"] > t_start) & (rtds_df["timestamp"] <= t0)]
    if window.empty:
        return True

    max_dev = float((window["rtds_price"] - base_price).abs().max())
    return max_dev < RTDS_MOVE_THRESHOLD


def _first_crossing_after(
    rtds_df: pd.DataFrame,
    start_idx: int,
    t0: pd.Timestamp,
    direction: str,
    base_price: float,
):
    """
    Find the FIRST RTDS update after t0 (and at index >= start_idx) that crosses threshold in direction.
    Returns (hit, hit_idx, hit_time, hit_price, lag_seconds) or (False, None, None, None, None).
    """
    t_end = t0 + timedelta(seconds=MAX_WAIT_TIME)

    # Slice forward from start_idx for speed
    forward = rtds_df.iloc[start_idx:].copy()
    forward = forward[(forward["timestamp"] > t0) & (forward["timestamp"] <= t_end)]
    if forward.empty:
        return False, None, None, None, None

    if direction == "up":
        thresh = base_price + RTDS_MOVE_THRESHOLD
        crossed = forward[forward["rtds_price"] >= thresh]
    else:
        thresh = base_price - RTDS_MOVE_THRESHOLD
        crossed = forward[forward["rtds_price"] <= thresh]

    if crossed.empty:
        return False, None, None, None, None

    hit_row = crossed.iloc[0]
    hit_time = hit_row["timestamp"]
    hit_price = float(hit_row["rtds_price"])
    hit_idx = int(hit_row.name)  # name corresponds to original index in rtds_df
    lag = (hit_time - t0).total_seconds()
    return True, hit_idx, hit_time, hit_price, lag


# -----------------------------
# Main test
# -----------------------------
def run_test(binance_df: pd.DataFrame, rtds_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame | None:
    if signals_df.empty:
        print("\nNo Binance signals at this threshold/cooldown.")
        print("SUCCESS RATE: N/A (0 predictions)\n")
        return None

    # Ensure RTDS sorted
    rtds_df = rtds_df.sort_values("timestamp").reset_index(drop=True)

    rows = []
    # Consumption pointer: next RTDS index we allow ourselves to use
    consume_from_idx = 0

    for _, s in signals_df.iterrows():
        t0 = s["signal_time"]
        direction = s["signal_direction"]

        baseline = _rtds_baseline_at_or_before(rtds_df, t0)
        if baseline is None:
            continue

        base_idx, base_time, base_price = baseline

        # Baseline staleness guard
        staleness = (t0 - base_time).total_seconds()
        if staleness > RTDS_BASE_MAX_STALENESS_S:
            # RTDS hasn't updated recently; baseline could be too old -> misleading
            continue

        # Flatness guard: RTDS must not already be moving into the signal
        if not _rtds_was_flat_before(rtds_df, t0, base_price):
            continue

        # Start searching from max(consumption pointer, baseline index) so we don't "reuse" old RTDS moves
        search_start_idx = max(consume_from_idx, base_idx)

        hit, hit_idx, hit_time, hit_price, lag = _first_crossing_after(
            rtds_df=rtds_df,
            start_idx=search_start_idx,
            t0=t0,
            direction=direction,
            base_price=base_price,
        )

        if hit:
            # Consume through the hit idx so it can't satisfy later signals
            consume_from_idx = hit_idx + 1

        rows.append(
            {
                "signal_time": t0,
                "signal_direction": direction,
                "binance_net_change": float(s["binance_net_change"]),
                "binance_price": float(s["binance_price"]),
                "rtds_base_time": base_time,
                "rtds_base_price": base_price,
                "baseline_staleness_s": staleness,
                "success": bool(hit),
                "hit_time": hit_time,
                "hit_price": hit_price,
                "lag_seconds": lag,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        print("\nAll signals got filtered out by leakage guards (baseline staleness / flatness / etc.).")
        return None

    n = len(results)
    wins = int(results["success"].sum())
    rate = 100.0 * wins / n if n else np.nan

    print("\n" + "=" * 70)
    print("BINANCE → RTDS EDGE TEST (LEAKAGE-RESISTANT)")
    print("=" * 70)
    print(f"Signals evaluated:            {n}")
    print(f"Successes:                   {wins}")
    print(f"Failures:                    {n - wins}")
    print(f"SUCCESS RATE:                {rate:.1f}%")
    hits = results[results["success"] & results["lag_seconds"].notna()]
    if not hits.empty:
        print(f"Median lag (sec):            {hits['lag_seconds'].median():.2f}")
        print(f"Mean lag (sec):              {hits['lag_seconds'].mean():.2f}")
    print(f"Filtered constraints:")
    print(f"  - RTDS baseline staleness <= {RTDS_BASE_MAX_STALENESS_S}s")
    print(f"  - RTDS flat pre-window     = {PRE_FLAT_S}s")
    print(f"  - First-cross within       = {MAX_WAIT_TIME}s")
    print("=" * 70 + "\n")

    # by direction
    for d in ["up", "down"]:
        sub = results[results["signal_direction"] == d]
        if len(sub) > 0:
            print(f"{d.title()} signals: {len(sub)}, success {sub['success'].mean()*100:.1f}%")
    print()
    return results


# -----------------------------
# Permutation null test
# -----------------------------
def permutation_test(signals_df: pd.DataFrame, rtds_df: pd.DataFrame, n_perm: int = 200, seed: int = 0):
    """
    Shuffle signal times among themselves (keeps directions and count),
    rerun evaluation to estimate 'no-edge' success rate distribution.

    If your observed success is not meaningfully above this distribution,
    your edge is likely fake/co-movement.
    """
    rng = np.random.default_rng(seed)
    times = signals_df["signal_time"].values

    rates = []
    for _ in range(n_perm):
        shuffled = signals_df.copy()
        shuffled["signal_time"] = rng.permutation(times)
        res = run_test(binance_df=None, rtds_df=rtds_df, signals_df=shuffled)  # binance_df unused in run_test
        if res is None or res.empty:
            continue
        rates.append(res["success"].mean())

    if not rates:
        print("Permutation test produced no valid samples (likely too strict filters).")
        return

    rates = np.array(rates)
    print("Permutation (null) success rate:")
    print(f"  mean:  {rates.mean()*100:.1f}%")
    print(f"  p95:   {np.quantile(rates, 0.95)*100:.1f}%")
    print(f"  p99:   {np.quantile(rates, 0.99)*100:.1f}%")
    print("If your observed success isn't meaningfully above p95, it's probably not edge.\n")


def main():
    binance_df, rtds_df = load_and_prepare_data(CSV_FILE)
    signals_df = make_binance_signals(binance_df)

    print(f"\nGenerated {len(signals_df)} Binance signals "
          f"(window={SIGNAL_WINDOW_S}s, thresh={BINANCE_MOVE_THRESHOLD}, cooldown={SIGNAL_COOLDOWN_S}s)\n")

    results = run_test(binance_df, rtds_df, signals_df)
    if results is None:
        return

    if RUN_PERMUTATION_TEST:
        permutation_test(signals_df, rtds_df, n_perm=N_PERMUTATIONS)


if __name__ == "__main__":
    main()