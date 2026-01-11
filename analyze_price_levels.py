"""
Price Level Analysis for Polymarket MM

Analyzes whether prices at different levels (especially tails near 0 or 100)
are systematically mispriced. Tests the "favorite-longshot bias" hypothesis.

Usage:
    python analyze_price_levels.py [path_to_fills.csv]

If no path provided, uses detailed_fills.csv in current directory.
"""

import pandas as pd
import numpy as np
import sys
from scipy import stats

# Minimum samples for statistical significance
MIN_SAMPLES = 10


def load_fills(filepath="markouts/detailed_fills.csv"):
    """Load fills data from CSV and compute per-share markouts."""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        print(f"Loaded {len(df)} fills from {filepath}")

        # Calculate per-share markouts (CSV contains total PNL, need to divide by qty)
        for horizon in [1, 5, 15, 30, 60]:
            col = f'markout_{horizon}s'
            per_share_col = f'markout_{horizon}s_per_share'
            if col in df.columns and 'qty' in df.columns:
                df[per_share_col] = df[col] / df['qty']

        # Validate required columns
        required = ['fill_yes', 'dir_yes', 'qty', 'markout_5s']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: Missing required columns: {missing}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

        print(f"Calculated per-share markouts")
        return df

    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)


def price_level_analysis(df):
    """Analyze markouts by price level buckets."""
    print("\n" + "=" * 80)
    print("PRICE LEVEL ANALYSIS")
    print("=" * 80)

    if 'fill_yes' not in df.columns:
        print("  Error: 'fill_yes' column not found in data")
        return df

    if 'markout_5s_per_share' not in df.columns:
        print("  Error: 'markout_5s_per_share' column not found in data")
        return df

    # Create price buckets (10 cent increments)
    df = df.copy()
    df['fill_yes_bucket'] = pd.cut(df['fill_yes'],
                                 bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                                 labels=['0-10¢', '10-20¢', '20-30¢', '30-40¢', '40-50¢',
                                        '50-60¢', '60-70¢', '70-80¢', '80-90¢', '90-100¢'],
                                 include_lowest=True)

    print(f"\nOverall by price level (all fills):")
    print(f"{'Price':<12} {'Fills':<8} {'Shares':<10} {'5s ¢/share':<14} {'15s ¢/share':<14} {'$ PNL':<12}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*14} {'-'*14} {'-'*12}")

    for bucket in df['fill_yes_bucket'].cat.categories:
        subset = df[df['fill_yes_bucket'] == bucket]
        if len(subset) > 0:
            shares = subset['qty'].sum() if 'qty' in subset.columns else len(subset)
            m5 = subset['markout_5s_per_share'].mean() * 100
            m15 = subset['markout_15s_per_share'].mean() * 100 if 'markout_15s_per_share' in subset.columns else 0
            pnl = subset['markout_5s'].sum() if 'markout_5s' in subset.columns else 0
            print(f"{bucket:<12} {len(subset):<8} {shares:<10.0f} {m5:>+.2f}¢         {m15:>+.2f}¢         ${pnl:>+.2f}")
        else:
            print(f"{bucket:<12} {'0':<8} {'-':<10} {'-':<14} {'-':<14} {'-':<12}")

    return df


def direction_by_price_analysis(df):
    """Analyze markouts by price level AND direction (buy vs sell YES)."""
    print("\n" + "=" * 80)
    print("PRICE LEVEL BY DIRECTION (Buy YES vs Sell YES)")
    print("=" * 80)

    if 'dir_yes' not in df.columns:
        print("  Error: 'dir_yes' column not found")
        return

    if 'fill_yes' not in df.columns:
        print("  Error: 'fill_yes' column not found")
        return

    if 'fill_yes_bucket' not in df.columns:
        df = df.copy()
        df['fill_yes_bucket'] = pd.cut(df['fill_yes'],
                                     bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                                     labels=['0-10¢', '10-20¢', '20-30¢', '30-40¢', '40-50¢',
                                            '50-60¢', '60-70¢', '70-80¢', '80-90¢', '90-100¢'],
                                     include_lowest=True)

    buy_yes = df[df['dir_yes'] == 1]
    sell_yes = df[df['dir_yes'] == -1]

    print(f"\n{'Price':<12} {'--- Buy YES ---':<30} {'--- Sell YES ---':<30}")
    print(f"{'Level':<12} {'Fills':<8} {'5s ¢/sh':<10} {'$ PNL':<12} {'Fills':<8} {'5s ¢/sh':<10} {'$ PNL':<12}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*12}")

    for bucket in df['fill_yes_bucket'].cat.categories:
        buy_sub = buy_yes[buy_yes['fill_yes_bucket'] == bucket]
        sell_sub = sell_yes[sell_yes['fill_yes_bucket'] == bucket]

        buy_fills = len(buy_sub)
        buy_m5 = buy_sub['markout_5s_per_share'].mean() * 100 if buy_fills > 0 else 0
        buy_pnl = buy_sub['markout_5s'].sum() if buy_fills > 0 and 'markout_5s' in buy_sub.columns else 0

        sell_fills = len(sell_sub)
        sell_m5 = sell_sub['markout_5s_per_share'].mean() * 100 if sell_fills > 0 else 0
        sell_pnl = sell_sub['markout_5s'].sum() if sell_fills > 0 and 'markout_5s' in sell_sub.columns else 0

        buy_str = f"{buy_fills:<8} {buy_m5:>+.2f}¢     ${buy_pnl:>+.2f}" if buy_fills > 0 else f"{'0':<8} {'-':<10} {'-':<12}"
        sell_str = f"{sell_fills:<8} {sell_m5:>+.2f}¢     ${sell_pnl:>+.2f}" if sell_fills > 0 else f"{'0':<8} {'-':<10} {'-':<12}"

        print(f"{bucket:<12} {buy_str:<30} {sell_str:<30}")


def tail_analysis(df):
    """Focused analysis on tail prices (near 0 or 100)."""
    print("\n" + "=" * 80)
    print("TAIL ANALYSIS (Extreme Prices)")
    print("=" * 80)

    if 'fill_yes' not in df.columns:
        print("  Error: 'fill_yes' column not found")
        return

    df = df.copy()

    # Define tails
    low_tail = df[df['fill_yes'] <= 0.15]  # 15¢ or less
    high_tail = df[df['fill_yes'] >= 0.85]  # 85¢ or more
    middle = df[(df['fill_yes'] > 0.15) & (df['fill_yes'] < 0.85)]

    print(f"\nFill distribution:")
    print(f"  Low tail (≤15¢):    {len(low_tail):>6} fills ({len(low_tail)/len(df)*100:.1f}%)")
    print(f"  Middle (15-85¢):    {len(middle):>6} fills ({len(middle)/len(df)*100:.1f}%)")
    print(f"  High tail (≥85¢):   {len(high_tail):>6} fills ({len(high_tail)/len(df)*100:.1f}%)")

    print(f"\n5s Markout by region:")
    for name, subset in [("Low tail ≤15¢", low_tail), ("Middle 15-85¢", middle), ("High tail ≥85¢", high_tail)]:
        if len(subset) > 0:
            m5 = subset['markout_5s_per_share'].mean() * 100
            m15 = subset['markout_15s_per_share'].mean() * 100 if 'markout_15s_per_share' in subset.columns else 0
            pnl = subset['markout_5s'].sum() if 'markout_5s' in subset.columns else 0
            print(f"  {name:<18} {len(subset):>5} fills, {m5:>+.2f}¢/share (5s), {m15:>+.2f}¢/share (15s), ${pnl:>+.2f} PNL")

    # Direction split for tails
    print(f"\nTail fills by direction:")
    print(f"  {'Region':<20} {'Buy YES':<25} {'Sell YES':<25}")
    print(f"  {'-'*20} {'-'*25} {'-'*25}")

    for name, subset in [("Low tail (≤15¢)", low_tail), ("High tail (≥85¢)", high_tail)]:
        if len(subset) > 0:
            buy = subset[subset['dir_yes'] == 1]
            sell = subset[subset['dir_yes'] == -1]

            buy_str = f"{len(buy)} fills, {buy['markout_5s_per_share'].mean()*100:>+.2f}¢" if len(buy) > 0 else "0 fills"
            sell_str = f"{len(sell)} fills, {sell['markout_5s_per_share'].mean()*100:>+.2f}¢" if len(sell) > 0 else "0 fills"

            print(f"  {name:<20} {buy_str:<25} {sell_str:<25}")

    # Test for systematic mispricing
    # Classic "favorite-longshot bias": longshots (low probability) are OVERPRICED
    # At 85¢ YES: the longshot is NO (15¢) - if overpriced, buying YES has edge
    # At 15¢ YES: the longshot is YES - if overpriced, selling YES has edge
    print(f"\nFavorite-Longshot Bias Test:")

    # At high YES prices (≥85¢), the longshot is NO
    # If tails overpriced → NO overpriced → YES underpriced → BUYING YES profitable
    high_tail_buys = high_tail[high_tail['dir_yes'] == 1]
    if len(high_tail_buys) >= MIN_SAMPLES:
        avg = high_tail_buys['markout_5s_per_share'].mean()
        t_stat, p_val = stats.ttest_1samp(high_tail_buys['markout_5s_per_share'], 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  Buying YES at ≥85¢:  {avg*100:+.2f}¢/share, n={len(high_tail_buys)}, p={p_val:.4f} {sig}")
        if avg > 0.005 and p_val < 0.05:
            print(f"    → NO tail (longshot) appears OVERPRICED - buying YES is profitable")
        elif avg < -0.005 and p_val < 0.05:
            print(f"    → YES appears OVERPRICED at high prices - avoid buying")

    high_tail_sells = high_tail[high_tail['dir_yes'] == -1]
    if len(high_tail_sells) >= MIN_SAMPLES:
        avg = high_tail_sells['markout_5s_per_share'].mean()
        t_stat, p_val = stats.ttest_1samp(high_tail_sells['markout_5s_per_share'], 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  Selling YES at ≥85¢: {avg*100:+.2f}¢/share, n={len(high_tail_sells)}, p={p_val:.4f} {sig}")
        if avg > 0.005 and p_val < 0.05:
            print(f"    → YES appears OVERPRICED at high prices - selling has edge")

    # At low YES prices (≤15¢), the longshot is YES
    # If tails overpriced → YES overpriced → SELLING YES profitable
    low_tail_sells = low_tail[low_tail['dir_yes'] == -1]
    if len(low_tail_sells) >= MIN_SAMPLES:
        avg = low_tail_sells['markout_5s_per_share'].mean()
        t_stat, p_val = stats.ttest_1samp(low_tail_sells['markout_5s_per_share'], 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  Selling YES at ≤15¢: {avg*100:+.2f}¢/share, n={len(low_tail_sells)}, p={p_val:.4f} {sig}")
        if avg > 0.005 and p_val < 0.05:
            print(f"    → YES tail (longshot) appears OVERPRICED - selling has edge")

    low_tail_buys = low_tail[low_tail['dir_yes'] == 1]
    if len(low_tail_buys) >= MIN_SAMPLES:
        avg = low_tail_buys['markout_5s_per_share'].mean()
        t_stat, p_val = stats.ttest_1samp(low_tail_buys['markout_5s_per_share'], 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  Buying YES at ≤15¢:  {avg*100:+.2f}¢/share, n={len(low_tail_buys)}, p={p_val:.4f} {sig}")
        if avg > 0.005 and p_val < 0.05:
            print(f"    → YES tail appears UNDERPRICED - buying has edge")


def extreme_tail_analysis(df):
    """Even more extreme tails (≤10¢ and ≥90¢)."""
    print("\n" + "=" * 80)
    print("EXTREME TAIL ANALYSIS (≤10¢ and ≥90¢)")
    print("=" * 80)

    if 'fill_yes' not in df.columns:
        print("  Error: 'fill_yes' column not found")
        return

    very_low = df[df['fill_yes'] <= 0.10]
    very_high = df[df['fill_yes'] >= 0.90]

    print(f"\nExtreme low (≤10¢): {len(very_low)} fills")
    if len(very_low) > 0:
        buy = very_low[very_low['dir_yes'] == 1]
        sell = very_low[very_low['dir_yes'] == -1]
        print(f"  Buy YES:  {len(buy):>4} fills, {buy['markout_5s_per_share'].mean()*100:>+.2f}¢/share" if len(buy) > 0 else "  Buy YES:  0 fills")
        print(f"  Sell YES: {len(sell):>4} fills, {sell['markout_5s_per_share'].mean()*100:>+.2f}¢/share" if len(sell) > 0 else "  Sell YES: 0 fills")

    print(f"\nExtreme high (≥90¢): {len(very_high)} fills")
    if len(very_high) > 0:
        buy = very_high[very_high['dir_yes'] == 1]
        sell = very_high[very_high['dir_yes'] == -1]
        print(f"  Buy YES:  {len(buy):>4} fills, {buy['markout_5s_per_share'].mean()*100:>+.2f}¢/share" if len(buy) > 0 else "  Buy YES:  0 fills")
        print(f"  Sell YES: {len(sell):>4} fills, {sell['markout_5s_per_share'].mean()*100:>+.2f}¢/share" if len(sell) > 0 else "  Sell YES: 0 fills")


def spread_by_price_analysis(df):
    """Analyze if spreads are wider at tails (less liquidity)."""
    print("\n" + "=" * 80)
    print("SPREAD BY PRICE LEVEL")
    print("=" * 80)

    if 'spread' not in df.columns:
        print("  'spread' column not in data - skipping")
        return

    df = df.copy()
    if 'fill_yes_bucket' not in df.columns:
        df['fill_yes_bucket'] = pd.cut(df['fill_yes'],
                                     bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                                     labels=['0-10¢', '10-20¢', '20-30¢', '30-40¢', '40-50¢',
                                            '50-60¢', '60-70¢', '70-80¢', '80-90¢', '90-100¢'],
                                     include_lowest=True)

    print(f"\n{'Price Level':<12} {'Avg Spread':<12} {'Fills':<10}")
    print(f"{'-'*12} {'-'*12} {'-'*10}")

    for bucket in df['fill_yes_bucket'].cat.categories:
        subset = df[df['fill_yes_bucket'] == bucket]
        if len(subset) > 0:
            avg_spread = subset['spread'].mean() * 100
            print(f"{bucket:<12} {avg_spread:.2f}¢        {len(subset):<10}")


def summary_and_recommendations(df):
    """Summarize findings and make recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    if 'fill_yes' not in df.columns:
        print("  Error: 'fill_yes' column not found")
        return

    low_tail = df[df['fill_yes'] <= 0.15]
    high_tail = df[df['fill_yes'] >= 0.85]
    middle = df[(df['fill_yes'] > 0.15) & (df['fill_yes'] < 0.85)]

    findings = []

    # Compare tails to middle
    if len(low_tail) >= MIN_SAMPLES and len(middle) >= MIN_SAMPLES:
        low_avg = low_tail['markout_5s_per_share'].mean()
        mid_avg = middle['markout_5s_per_share'].mean()
        if low_avg < mid_avg - 0.005:
            findings.append(f"Low tail (≤15¢) underperforms middle by {(mid_avg-low_avg)*100:.2f}¢/share")
        elif low_avg > mid_avg + 0.005:
            findings.append(f"Low tail (≤15¢) OUTperforms middle by {(low_avg-mid_avg)*100:.2f}¢/share")

    if len(high_tail) >= MIN_SAMPLES and len(middle) >= MIN_SAMPLES:
        high_avg = high_tail['markout_5s_per_share'].mean()
        mid_avg = middle['markout_5s_per_share'].mean()
        if high_avg < mid_avg - 0.005:
            findings.append(f"High tail (≥85¢) underperforms middle by {(mid_avg-high_avg)*100:.2f}¢/share")
        elif high_avg > mid_avg + 0.005:
            findings.append(f"High tail (≥85¢) OUTperforms middle by {(high_avg-mid_avg)*100:.2f}¢/share")

    # Check for directional edge at tails
    # At high prices: check both buy and sell
    high_buys = high_tail[high_tail['dir_yes'] == 1]
    high_sells = high_tail[high_tail['dir_yes'] == -1]
    if len(high_buys) >= MIN_SAMPLES:
        avg = high_buys['markout_5s_per_share'].mean()
        if avg > 0.01:
            findings.append(f"Buying YES at ≥85¢ has +{avg*100:.2f}¢ edge - NO tail overpriced, bias toward buying")
        elif avg < -0.01:
            findings.append(f"Buying YES at ≥85¢ loses {abs(avg)*100:.2f}¢ - YES overpriced at high prices, avoid buying")
    if len(high_sells) >= MIN_SAMPLES:
        avg = high_sells['markout_5s_per_share'].mean()
        if avg > 0.01:
            findings.append(f"Selling YES at ≥85¢ has +{avg*100:.2f}¢ edge - YES overpriced at high prices, bias toward selling")

    # At low prices: check both buy and sell
    low_buys = low_tail[low_tail['dir_yes'] == 1]
    low_sells = low_tail[low_tail['dir_yes'] == -1]
    if len(low_sells) >= MIN_SAMPLES:
        avg = low_sells['markout_5s_per_share'].mean()
        if avg > 0.01:
            findings.append(f"Selling YES at ≤15¢ has +{avg*100:.2f}¢ edge - YES tail overpriced, bias toward selling")
    if len(low_buys) >= MIN_SAMPLES:
        avg = low_buys['markout_5s_per_share'].mean()
        if avg > 0.01:
            findings.append(f"Buying YES at ≤15¢ has +{avg*100:.2f}¢ edge - YES tail underpriced, bias toward buying")
        elif avg < -0.01:
            findings.append(f"Buying YES at ≤15¢ loses {abs(avg)*100:.2f}¢ - YES tail overpriced, avoid buying longshots")

    if findings:
        print("\nKey findings:")
        for f in findings:
            print(f"  - {f}")
    else:
        print("\nNo significant tail mispricing detected with current data.")
        print("Need more fills at extreme prices for reliable conclusions.")

    print("\n" + "=" * 80)


def main():
    # Load data
    filepath = sys.argv[1] if len(sys.argv) > 1 else "markouts/detailed_fills.csv"
    df = load_fills(filepath)

    # Run analyses
    df = price_level_analysis(df)
    direction_by_price_analysis(df)
    tail_analysis(df)
    extreme_tail_analysis(df)
    spread_by_price_analysis(df)
    summary_and_recommendations(df)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
