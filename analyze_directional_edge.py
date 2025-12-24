"""
Test if Binance price movements predict RTDS direction.

Tests: When Binance moves significantly, does RTDS eventually move in the same direction?
This is more robust to variable lag than fixed-horizon price prediction.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# Import global_state for usdtusd conversion rate
try:
    import global_state
except ImportError:
    global_state = None

# Configuration
CSV_FILE = "price_lag_data.csv"

# Movement detection thresholds
BINANCE_MOVE_THRESHOLD = 50  # Detect Binance moves of $50+
RTDS_MOVE_THRESHOLD = 30     # Count RTDS moves of $30+
MAX_WAIT_TIME = 120          # Wait up to 120 seconds for RTDS to respond


def load_and_adjust_data(csv_file):
    """Load CSV and adjust Binance prices for USDT/USD conversion."""
    print(f"Loading data from {csv_file}...")

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['event_time_iso'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Get USDT/USD rate
    usdtusd = 1.0
    if global_state is not None and hasattr(global_state, 'usdtusd'):
        usdtusd = global_state.usdtusd
        print(f"Using USDT/USD rate: {usdtusd:.6f}\n")
    else:
        print("WARNING: global_state.usdtusd not available. Using 1.0\n")

    # Adjust Binance prices
    df.loc[df['source'] == 'binance', 'price'] = df.loc[df['source'] == 'binance', 'price'] * usdtusd

    binance_df = df[df['source'] == 'binance'][['timestamp', 'price']].copy()
    binance_df.columns = ['timestamp', 'binance_price']

    rtds_df = df[df['source'] == 'polymarket'][['timestamp', 'price']].copy()
    rtds_df.columns = ['timestamp', 'rtds_price']

    print(f"Loaded {len(binance_df)} Binance ticks and {len(rtds_df)} RTDS updates\n")

    return binance_df, rtds_df


def detect_binance_moves(binance_df, move_threshold=50, lookback_seconds=10):
    """
    Detect significant Binance price movements.

    Returns list of moves with direction and magnitude.
    """
    print(f"Detecting Binance moves >= ${move_threshold} (lookback: {lookback_seconds}s)...")

    binance_df = binance_df.sort_values('timestamp').reset_index(drop=True)
    moves = []

    for i in range(len(binance_df)):
        current_time = binance_df.iloc[i]['timestamp']
        current_price = binance_df.iloc[i]['binance_price']

        # Look back N seconds
        lookback_start = current_time - timedelta(seconds=lookback_seconds)
        window = binance_df[
            (binance_df['timestamp'] >= lookback_start) &
            (binance_df['timestamp'] <= current_time)
        ]

        if len(window) < 2:
            continue

        # Check for significant move from min or max in window
        window_min = window['binance_price'].min()
        window_max = window['binance_price'].max()

        move_from_low = current_price - window_min
        move_from_high = window_max - current_price

        # Detect move up
        if move_from_low >= move_threshold:
            # Avoid duplicate detections (skip if last move was within 5 seconds)
            if len(moves) > 0 and (current_time - moves[-1]['time']).total_seconds() < 5:
                continue

            moves.append({
                'time': current_time,
                'price': current_price,
                'direction': 'up',
                'magnitude': move_from_low,
                'start_price': window_min
            })

        # Detect move down
        elif move_from_high >= move_threshold:
            if len(moves) > 0 and (current_time - moves[-1]['time']).total_seconds() < 5:
                continue

            moves.append({
                'time': current_time,
                'price': current_price,
                'direction': 'down',
                'magnitude': move_from_high,
                'start_price': window_max
            })

    print(f"Detected {len(moves)} significant Binance moves")
    up_moves = sum(1 for m in moves if m['direction'] == 'up')
    down_moves = sum(1 for m in moves if m['direction'] == 'down')
    print(f"  - Up moves: {up_moves}")
    print(f"  - Down moves: {down_moves}\n")

    return moves


def check_rtds_response(binance_move, rtds_df, max_wait_seconds=120, rtds_threshold=30):
    """
    Check if RTDS moved in the same direction as Binance within the wait time.

    Returns dict with success/failure and timing info.
    """
    move_time = binance_move['time']
    move_direction = binance_move['direction']
    binance_price = binance_move['price']

    # Get RTDS price at the time of Binance move
    rtds_at_move = rtds_df[rtds_df['timestamp'] <= move_time]
    if len(rtds_at_move) == 0:
        return None

    rtds_before = rtds_at_move.iloc[-1]['rtds_price']

    # Look for RTDS updates in the next max_wait_seconds
    end_time = move_time + timedelta(seconds=max_wait_seconds)
    rtds_after = rtds_df[
        (rtds_df['timestamp'] > move_time) &
        (rtds_df['timestamp'] <= end_time)
    ]

    if len(rtds_after) == 0:
        return {
            'success': False,
            'reason': 'no_rtds_update',
            'wait_time': None,
            'rtds_move': 0.0
        }

    # Check each subsequent RTDS update
    for idx, rtds_row in rtds_after.iterrows():
        rtds_new_price = rtds_row['rtds_price']
        rtds_move = rtds_new_price - rtds_before
        wait_time = (rtds_row['timestamp'] - move_time).total_seconds()

        # Check if RTDS moved in same direction with sufficient magnitude
        if move_direction == 'up' and rtds_move >= rtds_threshold:
            return {
                'success': True,
                'wait_time': wait_time,
                'rtds_move': rtds_move,
                'rtds_new_price': rtds_new_price,
                'binance_rtds_diff': binance_price - rtds_new_price
            }
        elif move_direction == 'down' and rtds_move <= -rtds_threshold:
            return {
                'success': True,
                'wait_time': wait_time,
                'rtds_move': rtds_move,
                'rtds_new_price': rtds_new_price,
                'binance_rtds_diff': binance_price - rtds_new_price
            }

    # RTDS updated but didn't move in the same direction
    final_rtds = rtds_after.iloc[-1]['rtds_price']
    final_move = final_rtds - rtds_before

    return {
        'success': False,
        'reason': 'wrong_direction',
        'wait_time': (rtds_after.iloc[-1]['timestamp'] - move_time).total_seconds(),
        'rtds_move': final_move
    }


def analyze_directional_edge(binance_moves, rtds_df):
    """Analyze if Binance moves predict RTDS direction."""
    print("Analyzing directional predictive power...")

    results = []

    for move in binance_moves:
        response = check_rtds_response(move, rtds_df, MAX_WAIT_TIME, RTDS_MOVE_THRESHOLD)
        if response is not None:
            results.append({
                'binance_time': move['time'],
                'binance_direction': move['direction'],
                'binance_magnitude': move['magnitude'],
                'success': response['success'],
                'wait_time': response.get('wait_time'),
                'rtds_move': response.get('rtds_move', 0.0)
            })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No results to analyze!\n")
        return None

    # Overall statistics
    success_rate = results_df['success'].mean() * 100

    print("\n" + "="*60)
    print("DIRECTIONAL PREDICTION RESULTS")
    print("="*60)
    print(f"Total Binance moves analyzed: {len(results_df)}")
    print(f"Successful predictions: {results_df['success'].sum()}")
    print(f"Failed predictions: {(~results_df['success']).sum()}")
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")

    # Success breakdown by direction
    up_moves = results_df[results_df['binance_direction'] == 'up']
    down_moves = results_df[results_df['binance_direction'] == 'down']

    if len(up_moves) > 0:
        up_success_rate = up_moves['success'].mean() * 100
        print(f"\nUp moves: {len(up_moves)} total, {up_success_rate:.1f}% success")

    if len(down_moves) > 0:
        down_success_rate = down_moves['success'].mean() * 100
        print(f"Down moves: {len(down_moves)} total, {down_success_rate:.1f}% success")

    # Timing analysis (for successful predictions)
    successful = results_df[results_df['success']]
    if len(successful) > 0:
        print(f"\nTiming of successful predictions:")
        print(f"  Mean wait time: {successful['wait_time'].mean():.1f} seconds")
        print(f"  Median wait time: {successful['wait_time'].median():.1f} seconds")
        print(f"  Min wait time: {successful['wait_time'].min():.1f} seconds")
        print(f"  Max wait time: {successful['wait_time'].max():.1f} seconds")

        print(f"\nRTDS movement magnitude (successful predictions):")
        print(f"  Mean: ${successful['rtds_move'].abs().mean():.2f}")
        print(f"  Median: ${successful['rtds_move'].abs().median():.2f}")

    print("="*60 + "\n")

    return results_df


def plot_results(results_df):
    """Plot directional prediction results."""
    if results_df is None or len(results_df) == 0:
        return

    print("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Success rate over time
    ax = axes[0, 0]
    window = min(20, len(results_df) // 4)
    if window > 0:
        rolling_success = results_df['success'].rolling(window=window, center=True).mean() * 100
        ax.plot(results_df['binance_time'], rolling_success, linewidth=2)
        ax.axhline(50, color='red', linestyle='--', label='Random (50%)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Rolling Success Rate (window={window})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')

    # Plot 2: Wait time distribution (successful predictions)
    ax = axes[0, 1]
    successful = results_df[results_df['success']]
    if len(successful) > 0:
        ax.hist(successful['wait_time'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(successful['wait_time'].median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {successful["wait_time"].median():.1f}s')
        ax.set_xlabel('Wait Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Time Until RTDS Responds (Successful)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No successful predictions', ha='center', va='center')

    # Plot 3: Success by move magnitude
    ax = axes[1, 0]
    ax.scatter(results_df['binance_magnitude'], results_df['success'],
              alpha=0.5, s=30)
    ax.set_xlabel('Binance Move Magnitude ($)')
    ax.set_ylabel('Success (1=Yes, 0=No)')
    ax.set_title('Success vs Binance Move Size')
    ax.grid(True, alpha=0.3)

    # Plot 4: Success rate by direction
    ax = axes[1, 1]
    up_success = results_df[results_df['binance_direction'] == 'up']['success'].mean() * 100
    down_success = results_df[results_df['binance_direction'] == 'down']['success'].mean() * 100
    overall_success = results_df['success'].mean() * 100

    bars = ax.bar(['Up Moves', 'Down Moves', 'Overall'],
                  [up_success, down_success, overall_success],
                  color=['green', 'red', 'blue'], alpha=0.7)
    ax.axhline(50, color='black', linestyle='--', linewidth=2, label='Random (50%)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('directional_edge.png', dpi=150, bbox_inches='tight')
    print("Saved: directional_edge.png\n")


def main():
    """Main analysis pipeline."""
    print("\n" + "="*60)
    print("DIRECTIONAL EDGE ANALYSIS")
    print("="*60 + "\n")

    print(f"Configuration:")
    print(f"  Binance move threshold: ${BINANCE_MOVE_THRESHOLD}")
    print(f"  RTDS move threshold: ${RTDS_MOVE_THRESHOLD}")
    print(f"  Max wait time: {MAX_WAIT_TIME} seconds\n")

    # Load data
    binance_df, rtds_df = load_and_adjust_data(CSV_FILE)

    # Detect Binance moves
    binance_moves = detect_binance_moves(binance_df, BINANCE_MOVE_THRESHOLD)

    if len(binance_moves) == 0:
        print("No significant Binance moves detected. Try lowering BINANCE_MOVE_THRESHOLD.")
        return

    # Analyze directional edge
    results_df = analyze_directional_edge(binance_moves, rtds_df)

    # Plot results
    if results_df is not None:
        plot_results(results_df)

    print("INTERPRETATION:")
    print("  >70% success rate: Strong directional edge - exploit this!")
    print("  50-70% success rate: Moderate edge - might be tradeable")
    print("  <50% success rate: No edge or inverse relationship")
    print("\nIf success rate is high, you can trade on Binance moves without")
    print("worrying about exact timing - just know RTDS will eventually follow.\n")


if __name__ == "__main__":
    main()
