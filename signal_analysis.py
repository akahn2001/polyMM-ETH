"""
Signal Interaction Analysis

Analyzes how trading signals interact and perform across different market conditions.
Uses existing detailed_fills.csv data to identify optimal signal combinations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
MARKOUTS_FILE = "markouts/detailed_fills.csv"


def load_data():
    """Load fills data from CSV."""
    csv_path = Path(MARKOUTS_FILE)
    if not csv_path.exists():
        print(f"ERROR: {MARKOUTS_FILE} not found!")
        print("Run the bot to generate fill data first.")
        return None

    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # Calculate per-share markouts (CSV contains total PNL, need to divide by qty)
    for horizon in [1, 5, 15, 30, 60]:
        col = f'markout_{horizon}s'
        per_share_col = f'markout_{horizon}s_per_share'
        if col in df.columns and 'qty' in df.columns:
            df[per_share_col] = df[col] / df['qty']

    # Filter to fills with required columns
    required = ['zscore', 'z_skew', 'book_imbalance', 'dir_yes', 'markout_5s_per_share']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None

    # Check for optional columns (residual data)
    has_residual = 'z_skew_residual' in df.columns
    if not has_residual:
        print("WARNING: z_skew_residual column not found - residual analysis will be skipped")
        print("Delete detailed_fills.csv and restart bot with updated code to get this data\n")

    # Filter to fills with signal data (non-zero z-score or imbalance)
    df_with_signals = df[(df['zscore'].abs() > 0.001) | (df['book_imbalance'].abs() > 0.001)].copy()

    if len(df_with_signals) == 0:
        print("ERROR: No fills with signal data found!")
        return None

    print(f"Loaded {len(df_with_signals)} fills with signal data\n")
    return df_with_signals


def create_2d_heatmap(df, signal1_name, signal1_bins, signal2_name, signal2_bins):
    """
    Create 2D heatmap showing how two signals interact.

    Returns DataFrame with columns: signal1_bucket, signal2_bucket, count, avg_markout, win_rate
    """
    # Create bucket labels
    signal1_labels = [f"{signal1_bins[i]:.1f} to {signal1_bins[i+1]:.1f}"
                      for i in range(len(signal1_bins)-1)]
    signal2_labels = [f"{signal2_bins[i]:.1f} to {signal2_bins[i+1]:.1f}"
                      for i in range(len(signal2_bins)-1)]

    # Bucket the data
    df['sig1_bucket'] = pd.cut(df[signal1_name], bins=signal1_bins, labels=signal1_labels, include_lowest=True, ordered=False)
    df['sig2_bucket'] = pd.cut(df[signal2_name], bins=signal2_bins, labels=signal2_labels, include_lowest=True, ordered=False)

    # Group by both buckets
    results = []
    for s1_label in signal1_labels:
        for s2_label in signal2_labels:
            subset = df[(df['sig1_bucket'] == s1_label) & (df['sig2_bucket'] == s2_label)]

            if len(subset) == 0:
                continue

            # Calculate metrics when FOLLOWING the combined signal
            # Following = trade direction matches signal direction
            # For z-score: positive z-score = bullish (dir_yes=1), negative = bearish (dir_yes=-1)
            # For z-skew: positive = bullish, negative = bearish
            # For imbalance: positive (bid pressure) = bearish for YES (shorts want to sell YES), negative (ask pressure) = bullish

            # Determine if each fill followed the signal
            following_mask = np.ones(len(subset), dtype=bool)

            # Z-score direction
            if signal1_name == 'zscore':
                zscore_bullish = subset['zscore'] > 0
                following_mask &= ((zscore_bullish & (subset['dir_yes'] == 1)) |
                                  (~zscore_bullish & (subset['dir_yes'] == -1)))

            # Z-skew direction
            if signal1_name == 'z_skew':
                zskew_bullish = subset['z_skew'] > 0
                following_mask &= ((zskew_bullish & (subset['dir_yes'] == 1)) |
                                  (~zskew_bullish & (subset['dir_yes'] == -1)))

            # Z-skew residual direction
            if signal1_name == 'z_skew_residual':
                resid_bullish = subset['z_skew_residual'] > 0
                following_mask &= ((resid_bullish & (subset['dir_yes'] == 1)) |
                                  (~resid_bullish & (subset['dir_yes'] == -1)))

            # Book imbalance direction (inverted: positive imbalance = bid pressure = bearish for YES)
            if signal1_name == 'book_imbalance':
                imb_bearish = subset['book_imbalance'] > 0
                following_mask &= ((imb_bearish & (subset['dir_yes'] == -1)) |
                                  (~imb_bearish & (subset['dir_yes'] == 1)))

            # Apply same logic for signal 2
            if signal2_name == 'zscore':
                zscore_bullish = subset['zscore'] > 0
                following_mask &= ((zscore_bullish & (subset['dir_yes'] == 1)) |
                                  (~zscore_bullish & (subset['dir_yes'] == -1)))

            if signal2_name == 'z_skew':
                zskew_bullish = subset['z_skew'] > 0
                following_mask &= ((zskew_bullish & (subset['dir_yes'] == 1)) |
                                  (~zskew_bullish & (subset['dir_yes'] == -1)))

            # Z-skew residual direction
            if signal2_name == 'z_skew_residual':
                resid_bullish = subset['z_skew_residual'] > 0
                following_mask &= ((resid_bullish & (subset['dir_yes'] == 1)) |
                                  (~resid_bullish & (subset['dir_yes'] == -1)))

            if signal2_name == 'book_imbalance':
                imb_bearish = subset['book_imbalance'] > 0
                following_mask &= ((imb_bearish & (subset['dir_yes'] == -1)) |
                                  (~imb_bearish & (subset['dir_yes'] == 1)))

            following = subset[following_mask]

            count = len(subset)
            following_count = len(following)
            avg_markout = following['markout_5s_per_share'].mean() if len(following) > 0 else 0
            win_rate = (following['markout_5s_per_share'] > 0).sum() / len(following) if len(following) > 0 else 0

            results.append({
                'sig1_bucket': s1_label,
                'sig2_bucket': s2_label,
                'total_fills': count,
                'following_fills': following_count,
                'avg_markout': avg_markout,
                'win_rate': win_rate
            })

    return pd.DataFrame(results)


def print_2d_heatmap(df, signal1_name, signal2_name):
    """Print 2D heatmap in readable format."""
    print(f"\n{'='*80}")
    print(f"2D HEATMAP: {signal1_name.upper()} (rows) vs {signal2_name.upper()} (cols)")
    print(f"{'='*80}\n")

    if len(df) == 0:
        print("No data in any buckets\n")
        return

    # Get unique buckets and sort them
    sig1_buckets = sorted(df['sig1_bucket'].unique(), key=str)
    sig2_buckets = sorted(df['sig2_bucket'].unique(), key=str)

    # Print as a simple list instead of grid (more readable)
    print(f"{'ROW ('+signal1_name+')':<20} {'COL ('+signal2_name+')':<20} {'N':>5} {'Markout':>10} {'WinRate':>8}")
    print("-" * 70)

    for s1 in sig1_buckets:
        for s2 in sig2_buckets:
            cell = df[(df['sig1_bucket'] == s1) & (df['sig2_bucket'] == s2)]
            if len(cell) > 0:
                row = cell.iloc[0]
                if row['following_fills'] > 0:
                    print(f"{str(s1):<20} {str(s2):<20} {row['following_fills']:>5} {row['avg_markout']:>+10.4f} {row['win_rate']:>8.1%}")

    print()


def conditional_analysis(df, filter_signal, filter_ranges, analyze_signals):
    """
    Conditional analysis: filter by one signal, analyze performance of others.

    Args:
        df: DataFrame with fills
        filter_signal: Signal to filter on ('zscore', 'z_skew', 'book_imbalance')
        filter_ranges: List of (min, max, label) tuples for filtering
        analyze_signals: List of signals to analyze within each filter range
    """
    print(f"\n{'='*100}")
    print(f"CONDITIONAL ANALYSIS: Filter by {filter_signal.upper()}")
    print(f"{'='*100}\n")

    for min_val, max_val, label in filter_ranges:
        # Filter data to this range
        if min_val == -np.inf:
            subset = df[df[filter_signal] < max_val]
        elif max_val == np.inf:
            subset = df[df[filter_signal] >= min_val]
        else:
            subset = df[(df[filter_signal] >= min_val) & (df[filter_signal] < max_val)]

        if len(subset) == 0:
            continue

        print(f"\n{filter_signal.upper()} {label}: {len(subset)} fills")
        print("-" * 80)

        # Analyze each signal within this filter
        for signal_name in analyze_signals:
            # Calculate performance when following this signal
            if signal_name == 'zscore':
                bullish = subset['zscore'] > 0
                following = subset[((bullish & (subset['dir_yes'] == 1)) |
                                   (~bullish & (subset['dir_yes'] == -1)))]
            elif signal_name == 'z_skew':
                bullish = subset['z_skew'] > 0
                following = subset[((bullish & (subset['dir_yes'] == 1)) |
                                   (~bullish & (subset['dir_yes'] == -1)))]
            elif signal_name == 'book_imbalance':
                bearish = subset['book_imbalance'] > 0  # Inverted
                following = subset[((bearish & (subset['dir_yes'] == -1)) |
                                   (~bearish & (subset['dir_yes'] == 1)))]

            if len(following) > 0:
                avg_signal = subset[signal_name].abs().mean()
                avg_markout = following['markout_5s_per_share'].mean()
                win_rate = (following['markout_5s_per_share'] > 0).sum() / len(following)

                print(f"  {signal_name:<20s}: avg={avg_signal:+.3f}  "
                      f"following={len(following):3d} fills  "
                      f"markout={avg_markout:+.5f}  "
                      f"wr={win_rate:.2%}")

    print()


def sigmoid_effectiveness_analysis(df):
    """
    Analyze if sigmoid confidence scaling is improving z_skew performance.

    Bucket fills by |z_score| magnitude (which determines sigmoid confidence).
    Within each bucket, show z_skew performance.
    If sigmoid is working: high |z_score| buckets should have better z_skew markouts.
    """
    print(f"\n{'='*100}")
    print(f"SIGMOID CONFIDENCE SCALING EFFECTIVENESS")
    print(f"{'='*100}")
    print("Analyzing if z_skew performs better when z_score is high (sigmoid near 1.0)")
    print("vs when z_score is low (sigmoid dampens z_skew)\n")

    # Filter to fills with z_skew signal
    df_with_zskew = df[df['z_skew'].abs() > 0.001].copy()

    if len(df_with_zskew) == 0:
        print("No fills with z_skew signal\n")
        return

    # Calculate sigmoid confidence for each fill (matches trading_config.py)
    MIDPOINT = 0.4
    STEEPNESS = 5.0
    df_with_zskew['z_confidence'] = 1.0 / (1.0 + np.exp(-STEEPNESS * (df_with_zskew['zscore'].abs() - MIDPOINT)))

    # Bucket by z_score magnitude (determines sigmoid strength)
    zscore_ranges = [
        (0, 0.2, "Very Low (conf ~7%)"),
        (0.2, 0.4, "Low (conf ~18-50%)"),
        (0.4, 0.6, "Medium (conf ~50-73%)"),
        (0.6, 1.0, "High (conf ~73-95%)"),
        (1.0, np.inf, "Very High (conf ~95%+)")
    ]

    print(f"{'Z-Score Magnitude':<25} {'Fills':<8} {'Avg Conf':<10} {'Avg |Z-Skew|':<15} {'Markout':<12} {'Win Rate':<10}")
    print("-" * 100)

    for min_z, max_z, label in zscore_ranges:
        subset = df_with_zskew[(df_with_zskew['zscore'].abs() >= min_z) &
                               (df_with_zskew['zscore'].abs() < max_z)]

        if len(subset) == 0:
            continue

        # Get fills that followed z_skew direction
        zskew_bullish = subset['z_skew'] > 0
        following = subset[((zskew_bullish & (subset['dir_yes'] == 1)) |
                           (~zskew_bullish & (subset['dir_yes'] == -1)))]

        if len(following) > 0:
            avg_conf = subset['z_confidence'].mean()
            avg_zskew = subset['z_skew'].abs().mean()
            avg_markout = following['markout_5s_per_share'].mean()
            win_rate = (following['markout_5s_per_share'] > 0).sum() / len(following)

            print(f"{label:<25} {len(following):<8} {avg_conf:<10.2f} ${avg_zskew*100:<14.2f}¢ "
                  f"${avg_markout:<11.5f} {win_rate:<10.2%}")

    print("\n✅ If sigmoid is working: Higher z_score buckets should show better markouts")
    print("⚠️  If not working: Markouts similar or worse at high z_score\n")


def residual_analysis(df):
    """
    Compare z_skew (full prediction) vs z_skew_residual (after subtracting market move).

    Shows whether the residual approach improves edge by avoiding double-counting
    when the market has already priced in the predicted move.
    """
    if 'z_skew_residual' not in df.columns:
        print("\n" + "="*100)
        print("RESIDUAL ANALYSIS - SKIPPED (z_skew_residual column not found)")
        print("="*100)
        print("Restart bot with updated code to collect this data\n")
        return

    print(f"\n{'='*100}")
    print(f"Z-SKEW RESIDUAL ANALYSIS")
    print(f"{'='*100}")
    print("Comparing full z_skew vs residual approach (subtracting market-priced move)\n")

    # Filter to fills with z_skew data
    df_with_zskew = df[(df['z_skew'].abs() > 0.001) | (df['z_skew_residual'].abs() > 0.001)].copy()

    if len(df_with_zskew) == 0:
        print("No fills with z_skew data\n")
        return

    # Overall comparison
    print("Overall Performance:")
    print("-" * 80)

    # Full z_skew approach
    zskew_bullish = df_with_zskew['z_skew'] > 0
    zskew_following = df_with_zskew[((zskew_bullish & (df_with_zskew['dir_yes'] == 1)) |
                                     (~zskew_bullish & (df_with_zskew['dir_yes'] == -1)))]

    # Residual approach
    resid_bullish = df_with_zskew['z_skew_residual'] > 0
    resid_following = df_with_zskew[((resid_bullish & (df_with_zskew['dir_yes'] == 1)) |
                                     (~resid_bullish & (df_with_zskew['dir_yes'] == -1)))]

    if len(zskew_following) > 0:
        zskew_markout = zskew_following['markout_5s_per_share'].mean()
        zskew_wr = (zskew_following['markout_5s_per_share'] > 0).sum() / len(zskew_following)
        print(f"Full Z-Skew:      {len(zskew_following):3d} fills  markout=${zskew_markout:+.5f}  wr={zskew_wr:.2%}")

    if len(resid_following) > 0:
        resid_markout = resid_following['markout_5s_per_share'].mean()
        resid_wr = (resid_following['markout_5s_per_share'] > 0).sum() / len(resid_following)
        print(f"Residual Approach: {len(resid_following):3d} fills  markout=${resid_markout:+.5f}  wr={resid_wr:.2%}")

    if len(zskew_following) > 0 and len(resid_following) > 0:
        diff = resid_markout - zskew_markout
        if diff > 0.0005:
            print(f"\n✅ Residual approach is BETTER by ${diff:.5f} per share")
        elif diff < -0.0005:
            print(f"\n❌ Full z_skew is BETTER by ${-diff:.5f} per share")
        else:
            print(f"\n≈  Similar performance (within 0.05¢/share)")

    # Breakdown by market_implied_move magnitude
    print("\n\nPerformance by Market-Priced Move:")
    print("-" * 80)

    if 'market_implied_move' in df_with_zskew.columns:
        market_move_ranges = [
            (0, 0.01, "<1¢ market move"),
            (0.01, 0.02, "1-2¢ market move"),
            (0.02, 0.04, "2-4¢ market move"),
            (0.04, np.inf, ">4¢ market move")
        ]

        print(f"{'Market Move':<20} {'Fills':<8} {'Full Z-Skew':<15} {'Residual':<15} {'Improvement':<15}")
        print("-" * 80)

        for min_move, max_move, label in market_move_ranges:
            subset = df_with_zskew[(df_with_zskew['market_implied_move'].abs() >= min_move) &
                                   (df_with_zskew['market_implied_move'].abs() < max_move)]

            if len(subset) == 0:
                continue

            # Full z_skew
            zskew_b = subset['z_skew'] > 0
            zskew_f = subset[((zskew_b & (subset['dir_yes'] == 1)) | (~zskew_b & (subset['dir_yes'] == -1)))]
            zskew_m = zskew_f['markout_5s_per_share'].mean() if len(zskew_f) > 0 else 0

            # Residual
            resid_b = subset['z_skew_residual'] > 0
            resid_f = subset[((resid_b & (subset['dir_yes'] == 1)) | (~resid_b & (subset['dir_yes'] == -1)))]
            resid_m = resid_f['markout_5s_per_share'].mean() if len(resid_f) > 0 else 0

            improvement = resid_m - zskew_m

            print(f"{label:<20} {len(subset):<8} ${zskew_m:<14.5f} ${resid_m:<14.5f} ${improvement:<14.5f}")

        print("\n✅ Residual should be better when market has already moved significantly")
        print("   (those are the cases where we're avoiding double-counting)\n")
    else:
        print("market_implied_move column not found - detailed breakdown skipped\n")


def main():
    """Run full signal interaction analysis."""
    df = load_data()
    if df is None:
        return

    # Define buckets for each signal
    zscore_bins = [-np.inf, -0.6, -0.3, 0, 0.3, 0.6, np.inf]
    zskew_bins = [-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf]  # in dollars (2¢, 1¢)
    imbalance_bins = [-1, -0.3, -0.1, 0.1, 0.3, 1]

    # ========== 2D HEATMAP ANALYSIS ==========

    # Z-score vs Z-skew
    heatmap_zs_zskew = create_2d_heatmap(df.copy(), 'zscore', zscore_bins, 'z_skew', zskew_bins)
    print_2d_heatmap(heatmap_zs_zskew, 'zscore', 'z_skew')

    # Z-score vs Book Imbalance
    heatmap_zs_imb = create_2d_heatmap(df.copy(), 'zscore', zscore_bins, 'book_imbalance', imbalance_bins)
    print_2d_heatmap(heatmap_zs_imb, 'zscore', 'book_imbalance')

    # Z-skew vs Book Imbalance
    heatmap_zskew_imb = create_2d_heatmap(df.copy(), 'z_skew', zskew_bins, 'book_imbalance', imbalance_bins)
    print_2d_heatmap(heatmap_zskew_imb, 'z_skew', 'book_imbalance')

    # Z-skew Residual vs Book Imbalance (if available)
    if 'z_skew_residual' in df.columns:
        heatmap_resid_imb = create_2d_heatmap(df.copy(), 'z_skew_residual', zskew_bins, 'book_imbalance', imbalance_bins)
        print_2d_heatmap(heatmap_resid_imb, 'z_skew_residual', 'book_imbalance')

    # ========== CONDITIONAL ANALYSIS ==========

    # Filter by Z-score, analyze other signals
    zscore_ranges = [
        (-np.inf, -0.6, "< -0.6 (strong bearish)"),
        (-0.6, -0.3, "-0.6 to -0.3 (moderate bearish)"),
        (-0.3, 0, "-0.3 to 0 (weak bearish)"),
        (0, 0.3, "0 to 0.3 (weak bullish)"),
        (0.3, 0.6, "0.3 to 0.6 (moderate bullish)"),
        (0.6, np.inf, "> 0.6 (strong bullish)")
    ]
    conditional_analysis(df.copy(), 'zscore', zscore_ranges, ['z_skew', 'book_imbalance'])

    # Filter by Z-skew, analyze other signals
    zskew_ranges = [
        (-np.inf, -0.02, "< -2¢ (strong bearish)"),
        (-0.02, -0.01, "-2¢ to -1¢ (moderate bearish)"),
        (-0.01, 0, "-1¢ to 0 (weak bearish)"),
        (0, 0.01, "0 to 1¢ (weak bullish)"),
        (0.01, 0.02, "1¢ to 2¢ (moderate bullish)"),
        (0.02, np.inf, "> 2¢ (strong bullish)")
    ]
    conditional_analysis(df.copy(), 'z_skew', zskew_ranges, ['zscore', 'book_imbalance'])

    # Filter by Book Imbalance, analyze other signals
    imbalance_ranges = [
        (-1, -0.3, "< -0.3 (strong ask pressure / bullish for YES)"),
        (-0.3, -0.1, "-0.3 to -0.1 (moderate ask pressure)"),
        (-0.1, 0.1, "-0.1 to 0.1 (balanced)"),
        (0.1, 0.3, "0.1 to 0.3 (moderate bid pressure)"),
        (0.3, 1, "> 0.3 (strong bid pressure / bearish for YES)")
    ]
    conditional_analysis(df.copy(), 'book_imbalance', imbalance_ranges, ['zscore', 'z_skew'])

    # ========== SIGMOID EFFECTIVENESS ANALYSIS ==========
    # Analyze if sigmoid confidence scaling improves z_skew performance
    sigmoid_effectiveness_analysis(df.copy())

    # ========== RESIDUAL ANALYSIS ==========
    # Compare full z_skew vs residual approach
    residual_analysis(df.copy())


if __name__ == "__main__":
    main()
