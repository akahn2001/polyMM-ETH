"""
Visualize Binance and Polymarket price streams to check for lag.

Creates time series plots to optically determine if Binance leads RTDS.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

# Import global_state for usdtusd conversion rate
try:
    import global_state
except ImportError:
    global_state = None

# Configuration
CSV_FILE = "price_lag_data.csv"


def load_and_adjust_data(csv_file):
    """Load CSV and adjust Binance prices for USDT/USD conversion."""
    print(f"Loading data from {csv_file}...")

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['event_time_iso'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Total records: {len(df)}")
    print(f"  - Binance: {len(df[df['source'] == 'binance'])}")
    print(f"  - Polymarket: {len(df[df['source'] == 'polymarket'])}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}\n")

    # Get USDT/USD rate
    usdtusd = 1.0
    if global_state is not None and hasattr(global_state, 'usdtusd'):
        usdtusd = global_state.usdtusd
        print(f"Using USDT/USD rate: {usdtusd:.6f}")
        if usdtusd > 1.0:
            premium_pct = (usdtusd - 1.0) * 100
            print(f"USDT at {premium_pct:.4f}% premium")
        elif usdtusd < 1.0:
            discount_pct = (1.0 - usdtusd) * 100
            print(f"USDT at {discount_pct:.4f}% discount")
    else:
        print("WARNING: global_state.usdtusd not available. Using 1.0")

    # Adjust Binance prices
    df.loc[df['source'] == 'binance', 'price'] = df.loc[df['source'] == 'binance', 'price'] * usdtusd
    print(f"Adjusted Binance prices: BTCUSDT × {usdtusd:.6f} = BTCUSD\n")

    return df


def plot_full_timeseries(df):
    """Plot the entire time series."""
    print("Creating full time series plot...")

    binance_df = df[df['source'] == 'binance'].copy()
    rtds_df = df[df['source'] == 'polymarket'].copy()

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot Binance (dots)
    ax.scatter(binance_df['timestamp'], binance_df['price'],
              s=1, alpha=0.5, label='Binance (BTCUSD)', color='blue')

    # Plot RTDS (dots, larger)
    ax.scatter(rtds_df['timestamp'], rtds_df['price'],
              s=10, alpha=0.8, label='Polymarket RTDS (BTCUSD)', color='red', marker='o')

    ax.set_xlabel('Time')
    ax.set_ylabel('BTC Price ($)')
    ax.set_title('Binance vs Polymarket RTDS Price Streams (Full Range)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_streams_full.png', dpi=150, bbox_inches='tight')
    print("Saved: price_streams_full.png\n")


def plot_zoomed_windows(df, num_windows=4, window_duration_seconds=120):
    """Plot several zoomed-in windows to see lag detail."""
    print(f"Creating {num_windows} zoomed windows ({window_duration_seconds}s each)...")

    binance_df = df[df['source'] == 'binance'].copy()
    rtds_df = df[df['source'] == 'polymarket'].copy()

    # Select evenly spaced windows
    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    interval = duration / (num_windows + 1)

    fig, axes = plt.subplots(num_windows, 1, figsize=(16, 4 * num_windows))
    if num_windows == 1:
        axes = [axes]

    for i in range(num_windows):
        start_time = df['timestamp'].min() + timedelta(seconds=(i + 1) * interval)
        end_time = start_time + timedelta(seconds=window_duration_seconds)

        # Filter data for this window
        binance_window = binance_df[
            (binance_df['timestamp'] >= start_time) &
            (binance_df['timestamp'] <= end_time)
        ]
        rtds_window = rtds_df[
            (rtds_df['timestamp'] >= start_time) &
            (rtds_df['timestamp'] <= end_time)
        ]

        ax = axes[i]

        # Plot Binance (line + dots)
        if len(binance_window) > 0:
            ax.plot(binance_window['timestamp'], binance_window['price'],
                   'o-', markersize=3, alpha=0.7, label='Binance', color='blue', linewidth=1)

        # Plot RTDS (line + dots, larger)
        if len(rtds_window) > 0:
            ax.plot(rtds_window['timestamp'], rtds_window['price'],
                   'o-', markersize=6, alpha=0.8, label='Polymarket RTDS', color='red', linewidth=1.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('BTC Price ($)')
        ax.set_title(f'Window {i+1}: {start_time.strftime("%H:%M:%S")} - {end_time.strftime("%H:%M:%S")}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_streams_zoomed.png', dpi=150, bbox_inches='tight')
    print("Saved: price_streams_zoomed.png\n")


def plot_price_differences(df):
    """Plot the price difference between Binance and RTDS over time."""
    print("Creating price difference plot...")

    binance_df = df[df['source'] == 'binance'][['timestamp', 'price']].copy()
    binance_df.columns = ['timestamp', 'binance_price']

    rtds_df = df[df['source'] == 'polymarket'][['timestamp', 'price']].copy()
    rtds_df.columns = ['timestamp', 'rtds_price']

    # For each RTDS update, find closest Binance price
    differences = []

    for idx, rtds_row in rtds_df.iterrows():
        rtds_time = rtds_row['timestamp']
        rtds_price = rtds_row['rtds_price']

        # Find closest Binance tick
        time_diff = abs(binance_df['timestamp'] - rtds_time)
        closest_idx = time_diff.idxmin()

        if time_diff.loc[closest_idx].total_seconds() <= 1.0:
            binance_price = binance_df.loc[closest_idx, 'binance_price']
            diff = binance_price - rtds_price

            differences.append({
                'timestamp': rtds_time,
                'diff': diff
            })

    diff_df = pd.DataFrame(differences)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Plot 1: Price difference over time
    ax = axes[0]
    ax.scatter(diff_df['timestamp'], diff_df['diff'], s=10, alpha=0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(diff_df['diff'].mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: ${diff_df["diff"].mean():.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Binance - RTDS Price Difference ($)')
    ax.set_title('Price Difference: Binance vs Polymarket RTDS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Histogram of differences
    ax = axes[1]
    ax.hist(diff_df['diff'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(diff_df['diff'].mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: ${diff_df["diff"].mean():.2f}')
    ax.axvline(diff_df['diff'].median(), color='green', linestyle='--', linewidth=2,
              label=f'Median: ${diff_df["diff"].median():.2f}')
    ax.set_xlabel('Binance - RTDS Price Difference ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Price Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_differences.png', dpi=150, bbox_inches='tight')
    print("Saved: price_differences.png\n")

    print(f"Price Difference Statistics:")
    print(f"  Mean: ${diff_df['diff'].mean():.2f}")
    print(f"  Median: ${diff_df['diff'].median():.2f}")
    print(f"  Std Dev: ${diff_df['diff'].std():.2f}")
    print(f"  Min: ${diff_df['diff'].min():.2f}")
    print(f"  Max: ${diff_df['diff'].max():.2f}\n")


def analyze_rtds_updates(df):
    """Analyze RTDS update patterns."""
    print("Analyzing RTDS update patterns...")

    rtds_df = df[df['source'] == 'polymarket'].copy()
    rtds_df = rtds_df.sort_values('timestamp').reset_index(drop=True)

    # Calculate time between updates
    rtds_df['time_since_last'] = rtds_df['timestamp'].diff().dt.total_seconds()

    # Calculate price changes
    rtds_df['price_change'] = rtds_df['price'].diff()

    print(f"\nRTDS Update Frequency:")
    print(f"  Mean time between updates: {rtds_df['time_since_last'].mean():.2f} seconds")
    print(f"  Median time between updates: {rtds_df['time_since_last'].median():.2f} seconds")
    print(f"  Min time between updates: {rtds_df['time_since_last'].min():.2f} seconds")
    print(f"  Max time between updates: {rtds_df['time_since_last'].max():.2f} seconds")

    print(f"\nRTDS Price Change per Update:")
    print(f"  Mean: ${rtds_df['price_change'].abs().mean():.2f}")
    print(f"  Median: ${rtds_df['price_change'].abs().median():.2f}")
    print(f"  Max: ${rtds_df['price_change'].abs().max():.2f}")


def main():
    """Main visualization pipeline."""
    print("\n" + "="*60)
    print("BINANCE vs POLYMARKET PRICE STREAM VISUALIZATION")
    print("="*60 + "\n")

    # Load data
    df = load_and_adjust_data(CSV_FILE)

    # Create visualizations
    plot_full_timeseries(df)
    plot_zoomed_windows(df, num_windows=4, window_duration_seconds=120)
    plot_price_differences(df)

    # Analyze patterns
    analyze_rtds_updates(df)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - price_streams_full.png (full time series)")
    print("  - price_streams_zoomed.png (4 detailed windows)")
    print("  - price_differences.png (Binance - RTDS differences)")
    print("\nExamine these plots to determine if Binance leads RTDS.")
    print("If Binance truly leads, you should see:")
    print("  1. Binance dots moving before RTDS dots in zoomed windows")
    print("  2. Consistent price difference patterns (not random noise)")
    print("  3. RTDS 'catching up' to Binance price levels\n")


if __name__ == "__main__":
    main()
