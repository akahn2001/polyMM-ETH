"""
Measure the lag between Binance and Polymarket RTDS.

Analyzes historical data to determine how far behind Binance the RTDS updates lag.
Uses global_state.usdtusd to convert BTCUSDT → BTCUSD for proper comparison.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import global_state for usdtusd conversion rate
try:
    import global_state
except ImportError:
    global_state = None

# Configuration
CSV_FILE = "price_lag_data.csv"


def load_data(csv_file):
    """Load and prepare data."""
    print(f"Loading data from {csv_file}...")

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['event_time_iso'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    binance_df = df[df['source'] == 'binance'][['timestamp', 'price']].copy()
    binance_df.columns = ['timestamp', 'binance_price']

    rtds_df = df[df['source'] == 'polymarket'][['timestamp', 'price']].copy()
    rtds_df.columns = ['timestamp', 'rtds_price']

    print(f"Loaded {len(binance_df)} Binance ticks and {len(rtds_df)} RTDS updates")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")

    return binance_df, rtds_df


def adjust_binance_for_usdtusd(binance_df):
    """
    Adjust Binance prices from BTCUSDT to BTCUSD using global_state.usdtusd.

    BTCUSD = BTCUSDT * usdtusd

    Returns adjusted binance_df and the usdtusd rate used.
    """
    if global_state is None or not hasattr(global_state, 'usdtusd'):
        print("WARNING: global_state.usdtusd not available. Using 1.0 (no adjustment).\n")
        return binance_df, 1.0

    usdtusd = global_state.usdtusd

    print("="*60)
    print("ADJUSTING BINANCE PRICES FOR USDT/USD CONVERSION")
    print("="*60)
    print(f"USDT/USD rate: {usdtusd:.6f}")

    if usdtusd > 1.0:
        premium_pct = (usdtusd - 1.0) * 100
        print(f"USDT at {premium_pct:.4f}% premium")
    elif usdtusd < 1.0:
        discount_pct = (1.0 - usdtusd) * 100
        print(f"USDT at {discount_pct:.4f}% discount")
    else:
        print(f"USDT at parity")

    binance_df_adjusted = binance_df.copy()
    binance_df_adjusted['binance_price'] = binance_df_adjusted['binance_price'] * usdtusd

    print(f"Converted BTCUSDT → BTCUSD by multiplying by {usdtusd:.6f}\n")

    return binance_df_adjusted, usdtusd


def method1_cross_correlation(binance_df, rtds_df, max_lag_seconds=60):
    """
    Method 1: Cross-correlation analysis.

    Resample both series to regular intervals and compute cross-correlation
    at different lags to find the lag that maximizes correlation.
    """
    print("="*60)
    print("METHOD 1: Cross-Correlation Analysis")
    print("="*60)

    # Resample to 1-second intervals
    binance_resampled = binance_df.set_index('timestamp').resample('1S')['binance_price'].last().fillna(method='ffill')
    rtds_resampled = rtds_df.set_index('timestamp').resample('1S')['rtds_price'].last().fillna(method='ffill')

    # Align the series
    common_index = binance_resampled.index.intersection(rtds_resampled.index)
    binance_series = binance_resampled.loc[common_index]
    rtds_series = rtds_resampled.loc[common_index]

    # Compute cross-correlation at different lags
    lags = range(0, max_lag_seconds + 1)
    correlations = []

    for lag in lags:
        if lag == 0:
            corr = binance_series.corr(rtds_series)
        else:
            # Shift RTDS forward (meaning Binance leads by 'lag' seconds)
            corr = binance_series[:-lag].corr(rtds_series[lag:])
        correlations.append(corr)

    # Find lag with maximum correlation
    max_corr_idx = np.argmax(correlations)
    optimal_lag = lags[max_corr_idx]
    max_correlation = correlations[max_corr_idx]

    print(f"Optimal lag: {optimal_lag} seconds")
    print(f"Correlation at optimal lag: {max_correlation:.4f}")
    print(f"Correlation at 0 lag: {correlations[0]:.4f}\n")

    return {
        'lags': list(lags),
        'correlations': correlations,
        'optimal_lag': optimal_lag,
        'max_correlation': max_correlation
    }


def method2_price_matching(binance_df, rtds_df, price_tolerance=10.0):
    """
    Method 2: Price matching.

    For each RTDS update, find when Binance was at a similar price level,
    and calculate the time difference.
    """
    print("="*60)
    print("METHOD 2: Price Matching Analysis")
    print("="*60)
    print(f"Using price tolerance: ${price_tolerance:.2f}\n")

    lags = []

    for idx, rtds_row in rtds_df.iterrows():
        rtds_time = rtds_row['timestamp']
        rtds_price = rtds_row['rtds_price']

        # Find Binance ticks before this RTDS update that were within tolerance of this price
        binance_before = binance_df[binance_df['timestamp'] < rtds_time].copy()

        if len(binance_before) == 0:
            continue

        # Calculate price difference
        binance_before['price_diff'] = abs(binance_before['binance_price'] - rtds_price)

        # Find closest price match
        closest_match = binance_before.loc[binance_before['price_diff'].idxmin()]

        if closest_match['price_diff'] <= price_tolerance:
            lag_seconds = (rtds_time - closest_match['timestamp']).total_seconds()

            # Only consider positive lags (Binance leading RTDS)
            if 0 <= lag_seconds <= 120:  # Ignore unreasonable lags
                lags.append({
                    'rtds_time': rtds_time,
                    'rtds_price': rtds_price,
                    'binance_time': closest_match['timestamp'],
                    'binance_price': closest_match['binance_price'],
                    'lag_seconds': lag_seconds,
                    'price_diff': closest_match['price_diff']
                })

    if len(lags) == 0:
        print("WARNING: No matching prices found. Try increasing price_tolerance.\n")
        return None

    lags_df = pd.DataFrame(lags)

    print(f"Found {len(lags_df)} matching price points")
    print(f"\nLag Statistics:")
    print(f"  Mean: {lags_df['lag_seconds'].mean():.2f} seconds")
    print(f"  Median: {lags_df['lag_seconds'].median():.2f} seconds")
    print(f"  Std Dev: {lags_df['lag_seconds'].std():.2f} seconds")
    print(f"  Min: {lags_df['lag_seconds'].min():.2f} seconds")
    print(f"  Max: {lags_df['lag_seconds'].max():.2f} seconds")

    print(f"\nPercentiles:")
    for pct in [25, 50, 75, 90, 95]:
        val = np.percentile(lags_df['lag_seconds'], pct)
        print(f"  {pct}th: {val:.2f} seconds")

    print()

    return lags_df


def method3_update_delay(binance_df, rtds_df):
    """
    Method 3: Direct update comparison.

    For each RTDS update, compare to the most recent Binance price at that moment,
    and measure the gap.
    """
    print("="*60)
    print("METHOD 3: Update Delay Analysis")
    print("="*60)

    delays = []

    for idx, rtds_row in rtds_df.iterrows():
        rtds_time = rtds_row['timestamp']
        rtds_price = rtds_row['rtds_price']

        # Get most recent Binance price at this RTDS update time
        binance_at_time = binance_df[binance_df['timestamp'] <= rtds_time]

        if len(binance_at_time) == 0:
            continue

        latest_binance = binance_at_time.iloc[-1]

        price_gap = rtds_price - latest_binance['binance_price']
        time_gap = (rtds_time - latest_binance['timestamp']).total_seconds()

        delays.append({
            'rtds_time': rtds_time,
            'rtds_price': rtds_price,
            'binance_price': latest_binance['binance_price'],
            'price_gap': price_gap,
            'time_since_binance_tick': time_gap
        })

    delays_df = pd.DataFrame(delays)

    print(f"Analyzed {len(delays_df)} RTDS updates\n")
    print(f"Price Gap Statistics (RTDS - Binance at same moment):")
    print(f"  Mean: ${delays_df['price_gap'].mean():.2f}")
    print(f"  Median: ${delays_df['price_gap'].median():.2f}")
    print(f"  Std Dev: ${delays_df['price_gap'].std():.2f}")
    print(f"  Mean Absolute Gap: ${delays_df['price_gap'].abs().mean():.2f}")

    print(f"\nTime since last Binance tick:")
    print(f"  Mean: {delays_df['time_since_binance_tick'].mean():.2f} seconds")
    print(f"  Median: {delays_df['time_since_binance_tick'].median():.2f} seconds\n")

    return delays_df


def method4_momentum_lag(binance_df, rtds_df):
    """
    Method 4: Momentum-based lag estimation.

    Look at RTDS changes and see how long ago Binance made similar changes.
    """
    print("="*60)
    print("METHOD 4: Momentum Lag Analysis")
    print("="*60)

    # Calculate changes in RTDS
    rtds_df = rtds_df.sort_values('timestamp').reset_index(drop=True)
    rtds_df['prev_price'] = rtds_df['rtds_price'].shift(1)
    rtds_df['price_change'] = rtds_df['rtds_price'] - rtds_df['prev_price']
    rtds_df['prev_time'] = rtds_df['timestamp'].shift(1)

    momentum_lags = []

    for idx, rtds_row in rtds_df.iterrows():
        if idx == 0 or pd.isna(rtds_row['price_change']):
            continue

        rtds_time = rtds_row['timestamp']
        rtds_change = rtds_row['price_change']

        if abs(rtds_change) < 1.0:  # Skip tiny changes
            continue

        # Look for similar change in Binance before this RTDS update
        prev_rtds_time = rtds_row['prev_time']

        # Get Binance prices between previous and current RTDS update
        binance_window = binance_df[
            (binance_df['timestamp'] >= prev_rtds_time) &
            (binance_df['timestamp'] <= rtds_time)
        ].copy()

        if len(binance_window) < 2:
            continue

        # Find when Binance crossed the new RTDS price
        new_rtds_price = rtds_row['rtds_price']
        old_rtds_price = rtds_row['prev_price']

        if rtds_change > 0:  # RTDS went up
            # Find when Binance crossed above new_rtds_price
            crosses = binance_window[binance_window['binance_price'] >= new_rtds_price]
        else:  # RTDS went down
            # Find when Binance crossed below new_rtds_price
            crosses = binance_window[binance_window['binance_price'] <= new_rtds_price]

        if len(crosses) > 0:
            first_cross = crosses.iloc[0]
            lag_seconds = (rtds_time - first_cross['timestamp']).total_seconds()

            if 0 <= lag_seconds <= 60:
                momentum_lags.append({
                    'rtds_time': rtds_time,
                    'rtds_change': rtds_change,
                    'binance_cross_time': first_cross['timestamp'],
                    'lag_seconds': lag_seconds
                })

    if len(momentum_lags) == 0:
        print("No momentum patterns detected.\n")
        return None

    momentum_df = pd.DataFrame(momentum_lags)

    print(f"Found {len(momentum_df)} RTDS updates with matching Binance movements\n")
    print(f"Lag from Binance crossing price to RTDS updating:")
    print(f"  Mean: {momentum_df['lag_seconds'].mean():.2f} seconds")
    print(f"  Median: {momentum_df['lag_seconds'].median():.2f} seconds")
    print(f"  Std Dev: {momentum_df['lag_seconds'].std():.2f} seconds\n")

    return momentum_df


def plot_results(cross_corr_result, price_match_df, momentum_df):
    """Plot all lag analysis results."""
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cross-correlation vs lag
    ax = axes[0, 0]
    ax.plot(cross_corr_result['lags'], cross_corr_result['correlations'], 'o-', linewidth=2)
    ax.axvline(cross_corr_result['optimal_lag'], color='red', linestyle='--',
               label=f'Optimal lag: {cross_corr_result["optimal_lag"]}s')
    ax.set_xlabel('Lag (seconds)')
    ax.set_ylabel('Correlation')
    ax.set_title('Cross-Correlation: Binance Leading RTDS')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Price matching lag distribution
    ax = axes[0, 1]
    if price_match_df is not None and len(price_match_df) > 0:
        ax.hist(price_match_df['lag_seconds'], bins=50, edgecolor='black', alpha=0.7)
        median_lag = price_match_df['lag_seconds'].median()
        ax.axvline(median_lag, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_lag:.1f}s')
        ax.set_xlabel('Lag (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Matching Lag Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No price matching data', ha='center', va='center')
        ax.set_title('Price Matching Lag Distribution')

    # Plot 3: Momentum lag distribution
    ax = axes[1, 0]
    if momentum_df is not None and len(momentum_df) > 0:
        ax.hist(momentum_df['lag_seconds'], bins=30, edgecolor='black', alpha=0.7, color='green')
        median_lag = momentum_df['lag_seconds'].median()
        ax.axvline(median_lag, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_lag:.1f}s')
        ax.set_xlabel('Lag (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Momentum Lag Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No momentum data', ha='center', va='center')
        ax.set_title('Momentum Lag Distribution')

    # Plot 4: Lag over time (if we have price matching data)
    ax = axes[1, 1]
    if price_match_df is not None and len(price_match_df) > 0:
        ax.scatter(price_match_df['rtds_time'], price_match_df['lag_seconds'],
                  alpha=0.5, s=20)
        # Add rolling median
        window_size = min(50, len(price_match_df) // 5)
        if window_size > 0:
            rolling_median = price_match_df.set_index('rtds_time')['lag_seconds'].rolling(
                window=window_size, center=True).median()
            ax.plot(rolling_median.index, rolling_median.values, 'r-', linewidth=2,
                   label=f'Rolling median (window={window_size})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Lag (seconds)')
        ax.set_title('Lag Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No time series data', ha='center', va='center')
        ax.set_title('Lag Over Time')

    plt.tight_layout()
    plt.savefig('lag_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved plot to lag_analysis.png\n")


def main():
    """Main analysis pipeline."""
    print("\n" + "="*60)
    print("BINANCE -> RTDS LAG MEASUREMENT")
    print("="*60 + "\n")

    # Load data
    binance_df, rtds_df = load_data(CSV_FILE)

    if len(rtds_df) < 10:
        print("ERROR: Not enough RTDS data for analysis. Need at least 10 updates.")
        return

    # Adjust Binance prices using global_state.usdtusd
    binance_df, usdtusd = adjust_binance_for_usdtusd(binance_df)

    # Run all methods
    cross_corr_result = method1_cross_correlation(binance_df, rtds_df, max_lag_seconds=60)
    price_match_df = method2_price_matching(binance_df, rtds_df, price_tolerance=10.0)
    delays_df = method3_update_delay(binance_df, rtds_df)
    momentum_df = method4_momentum_lag(binance_df, rtds_df)

    # Summary and recommendations
    print("="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)

    recommendations = []

    # Cross-correlation recommendation
    recommendations.append(cross_corr_result['optimal_lag'])
    print(f"Cross-correlation suggests: {cross_corr_result['optimal_lag']} seconds")

    # Price matching recommendation
    if price_match_df is not None and len(price_match_df) > 0:
        median_price_lag = price_match_df['lag_seconds'].median()
        recommendations.append(median_price_lag)
        print(f"Price matching suggests: {median_price_lag:.1f} seconds")

    # Momentum recommendation
    if momentum_df is not None and len(momentum_df) > 0:
        median_momentum_lag = momentum_df['lag_seconds'].median()
        recommendations.append(median_momentum_lag)
        print(f"Momentum analysis suggests: {median_momentum_lag:.1f} seconds")

    if recommendations:
        avg_recommendation = np.mean(recommendations)
        print(f"\n{'='*60}")
        print(f"RECOMMENDED HORIZON: {int(round(avg_recommendation))} seconds")
        print(f"{'='*60}")
        print(f"\nYou should update HORIZON in analyze_price_lag.py to: {int(round(avg_recommendation))}")
        print(f"\nAlso consider testing horizons around this value:")
        for h in [max(1, int(avg_recommendation - 5)),
                  int(avg_recommendation),
                  int(avg_recommendation + 5)]:
            print(f"  - {h} seconds")

    # Generate plots
    plot_results(cross_corr_result, price_match_df, momentum_df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
