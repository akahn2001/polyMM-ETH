"""
Price Level Analysis for Polymarket MM

Analyzes whether longshots vs favorites are systematically mispriced.
Tests the "favorite-longshot bias" hypothesis.

Usage:
    python analyze_price_levels.py [path_to_fills.csv]
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

        print(f"Calculated per-share markouts\n")
        return df

    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)


def compute_implied_prob(df):
    """
    Compute the implied probability of the side we're buying.

    - If we buy YES at 90¢ → we're buying a 90% favorite
    - If we sell YES at 90¢ → we're buying NO at 10¢ → buying a 10% longshot
    - If we buy YES at 10¢ → we're buying a 10% longshot
    - If we sell YES at 10¢ → we're buying NO at 90¢ → buying a 90% favorite
    """
    df = df.copy()

    # Implied probability of the side we're taking
    # dir_yes = 1 means we bought YES, so our implied prob is the YES price
    # dir_yes = -1 means we sold YES (bought NO), so our implied prob is 1 - YES price
    df['implied_prob'] = np.where(
        df['dir_yes'] == 1,
        df['fill_yes'],           # Bought YES: our side's prob = YES price
        1 - df['fill_yes']        # Sold YES (bought NO): our side's prob = NO price
    )

    return df


def significance_test(data, null_value=0):
    """Test if mean is significantly different from null_value. Returns (t_stat, p_val, sig_stars)."""
    clean_data = data.dropna()
    if len(clean_data) < MIN_SAMPLES:
        return None, None, ""
    try:
        t_stat, p_val = stats.ttest_1samp(clean_data, null_value)
        if np.isnan(p_val):
            return None, None, ""
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        return t_stat, p_val, sig
    except:
        return None, None, ""


def longshot_favorite_analysis(df):
    """Main analysis: markouts by implied probability of side we're buying."""
    print("=" * 80)
    print("LONGSHOT VS FAVORITE ANALYSIS")
    print("=" * 80)
    print("\nFor each fill, we compute the implied probability of the side we're buying:")
    print("  - Buy YES at 80¢ → buying 80% favorite")
    print("  - Sell YES at 80¢ → buying 20% longshot (the NO side)")
    print()

    df = compute_implied_prob(df)

    # Define buckets by implied probability
    buckets = [
        ("Extreme longshot (0-15%)", df[df['implied_prob'] <= 0.15]),
        ("Longshot (15-30%)", df[(df['implied_prob'] > 0.15) & (df['implied_prob'] <= 0.30)]),
        ("Slight underdog (30-45%)", df[(df['implied_prob'] > 0.30) & (df['implied_prob'] <= 0.45)]),
        ("Coin flip (45-55%)", df[(df['implied_prob'] > 0.45) & (df['implied_prob'] <= 0.55)]),
        ("Slight favorite (55-70%)", df[(df['implied_prob'] > 0.55) & (df['implied_prob'] <= 0.70)]),
        ("Favorite (70-85%)", df[(df['implied_prob'] > 0.70) & (df['implied_prob'] <= 0.85)]),
        ("Heavy favorite (85-100%)", df[df['implied_prob'] > 0.85]),
    ]

    print(f"{'Category':<28} {'Fills':<8} {'Shares':<10} {'5s ¢/share':<14} {'p-value':<12} {'$ PNL':<10}")
    print(f"{'-'*28} {'-'*8} {'-'*10} {'-'*14} {'-'*12} {'-'*10}")

    results = []
    for name, subset in buckets:
        if len(subset) > 0:
            shares = subset['qty'].sum()
            m5 = subset['markout_5s_per_share'].mean() * 100
            pnl = subset['markout_5s'].sum()

            # Significance test vs zero
            _, p_val, sig = significance_test(subset['markout_5s_per_share'], 0)
            p_str = f"p={p_val:.3f}{sig}" if p_val is not None else "n/a"

            print(f"{name:<28} {len(subset):<8} {shares:<10.0f} {m5:>+.2f}¢         {p_str:<12} ${pnl:>+.2f}")
            results.append((name, len(subset), m5, subset['markout_5s_per_share']))
        else:
            print(f"{name:<28} {'0':<8} {'-':<10} {'-':<14} {'-':<12} {'-':<10}")

    return df, results


def longshot_vs_favorite_comparison(df, results):
    """Compare longshots (< 30%) vs favorites (> 70%)."""
    print("\n" + "=" * 80)
    print("LONGSHOT VS FAVORITE COMPARISON")
    print("=" * 80)

    longshots = df[df['implied_prob'] <= 0.30]
    favorites = df[df['implied_prob'] >= 0.70]
    middle = df[(df['implied_prob'] > 0.30) & (df['implied_prob'] < 0.70)]

    print(f"\nAggregate comparison:")
    print(f"  {'Category':<20} {'Fills':<8} {'5s ¢/share':<14} {'p-value':<12} {'$ PNL':<12}")
    print(f"  {'-'*20} {'-'*8} {'-'*14} {'-'*12} {'-'*12}")

    for name, subset in [("Longshots (<30%)", longshots),
                          ("Middle (30-70%)", middle),
                          ("Favorites (>70%)", favorites)]:
        if len(subset) > 0:
            m5 = subset['markout_5s_per_share'].mean() * 100
            pnl = subset['markout_5s'].sum()
            _, p_val, sig = significance_test(subset['markout_5s_per_share'], 0)
            p_str = f"p={p_val:.3f}{sig}" if p_val is not None else "n/a"
            print(f"  {name:<20} {len(subset):<8} {m5:>+.2f}¢         {p_str:<12} ${pnl:>+.2f}")

    # Statistical test comparing longshots vs favorites
    long_clean = longshots['markout_5s_per_share'].dropna()
    fav_clean = favorites['markout_5s_per_share'].dropna()

    if len(long_clean) >= MIN_SAMPLES and len(fav_clean) >= MIN_SAMPLES:
        try:
            t_stat, p_val = stats.ttest_ind(long_clean, fav_clean)
            if np.isnan(p_val):
                print(f"\n  Statistical test: Could not compute (data issue)")
            else:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                long_avg = long_clean.mean() * 100
                fav_avg = fav_clean.mean() * 100

                print(f"\n  Statistical test (longshots vs favorites):")
                print(f"    Longshots: {long_avg:+.2f}¢, Favorites: {fav_avg:+.2f}¢")
                print(f"    Difference: {long_avg - fav_avg:+.2f}¢, p={p_val:.4f} {sig}")

                if long_avg < fav_avg - 0.5 and p_val < 0.05:
                    print(f"\n  --> LONGSHOTS UNDERPERFORM: Classic favorite-longshot bias detected!")
                    print(f"      Longshots are overpriced. Bias toward selling longshots / buying favorites.")
                elif long_avg > fav_avg + 0.5 and p_val < 0.05:
                    print(f"\n  --> FAVORITES UNDERPERFORM: Reverse bias detected!")
                    print(f"      Favorites are overpriced. Bias toward buying longshots / selling favorites.")
                else:
                    print(f"\n  --> No significant favorite-longshot bias detected.")
        except Exception as e:
            print(f"\n  Statistical test error: {e}")


def continuous_probability_analysis(df):
    """Analyze the continuous relationship between implied prob and markout."""
    print("\n" + "=" * 80)
    print("CONTINUOUS PROBABILITY ANALYSIS")
    print("=" * 80)

    df = compute_implied_prob(df)

    # Correlation between implied probability and markout
    corr = df['implied_prob'].corr(df['markout_5s_per_share'])
    print(f"\nCorrelation (implied_prob vs markout): {corr:.4f}")

    if corr > 0.05:
        print("  → Positive correlation: higher probability bets have better markouts")
        print("  → Suggests LONGSHOTS are OVERPRICED (buying favorites is better)")
    elif corr < -0.05:
        print("  → Negative correlation: lower probability bets have better markouts")
        print("  → Suggests FAVORITES are OVERPRICED (buying longshots is better)")
    else:
        print("  → No significant correlation between probability and markout")

    # Finer-grained buckets (10% increments)
    print(f"\nFine-grained analysis (10% buckets):")
    print(f"  {'Implied Prob':<15} {'Fills':<8} {'5s ¢/share':<14}")
    print(f"  {'-'*15} {'-'*8} {'-'*14}")

    for low in range(0, 100, 10):
        high = low + 10
        subset = df[(df['implied_prob'] > low/100) & (df['implied_prob'] <= high/100)]
        if len(subset) >= 5:
            m5 = subset['markout_5s_per_share'].mean() * 100
            print(f"  {low:>3}-{high:<3}%         {len(subset):<8} {m5:>+.2f}¢")
        elif len(subset) > 0:
            print(f"  {low:>3}-{high:<3}%         {len(subset):<8} (too few)")
        else:
            print(f"  {low:>3}-{high:<3}%         {'0':<8} -")


def extreme_longshot_analysis(df):
    """Deep dive on extreme longshots (< 15%)."""
    print("\n" + "=" * 80)
    print("EXTREME LONGSHOT ANALYSIS (<15% implied probability)")
    print("=" * 80)

    df = compute_implied_prob(df)
    extreme = df[df['implied_prob'] <= 0.15]

    if len(extreme) < 5:
        print(f"\n  Only {len(extreme)} fills at extreme longshot prices - insufficient data")
        return

    print(f"\n  Total extreme longshot fills: {len(extreme)}")
    print(f"  Avg markout: {extreme['markout_5s_per_share'].mean()*100:+.2f}¢/share")
    print(f"  Total PNL: ${extreme['markout_5s'].sum():+.2f}")

    # Break down further
    print(f"\n  {'Probability':<15} {'Fills':<8} {'5s ¢/share':<14}")
    print(f"  {'-'*15} {'-'*8} {'-'*14}")

    for low, high in [(0, 5), (5, 10), (10, 15)]:
        subset = extreme[(extreme['implied_prob'] > low/100) & (extreme['implied_prob'] <= high/100)]
        if len(subset) > 0:
            m5 = subset['markout_5s_per_share'].mean() * 100
            print(f"  {low:>3}-{high:<3}%         {len(subset):<8} {m5:>+.2f}¢")

    # Test if significantly different from zero
    if len(extreme) >= MIN_SAMPLES:
        t_stat, p_val = stats.ttest_1samp(extreme['markout_5s_per_share'], 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        avg = extreme['markout_5s_per_share'].mean() * 100
        print(f"\n  t-test vs zero: {avg:+.2f}¢, p={p_val:.4f} {sig}")

        if avg < -0.5 and p_val < 0.05:
            print(f"  --> Extreme longshots are OVERPRICED - avoid buying them!")
        elif avg > 0.5 and p_val < 0.05:
            print(f"  --> Extreme longshots are UNDERPRICED - edge in buying them!")


def summary(df):
    """Final summary and recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    df = compute_implied_prob(df)

    longshots = df[df['implied_prob'] <= 0.30]
    favorites = df[df['implied_prob'] >= 0.70]

    findings = []

    if len(longshots) >= MIN_SAMPLES:
        long_avg = longshots['markout_5s_per_share'].mean()
        if long_avg < -0.005:
            findings.append(f"Longshots (<30%) losing {abs(long_avg)*100:.2f}¢/share - they're OVERPRICED")
        elif long_avg > 0.005:
            findings.append(f"Longshots (<30%) making +{long_avg*100:.2f}¢/share - they're UNDERPRICED")

    if len(favorites) >= MIN_SAMPLES:
        fav_avg = favorites['markout_5s_per_share'].mean()
        if fav_avg < -0.005:
            findings.append(f"Favorites (>70%) losing {abs(fav_avg)*100:.2f}¢/share - they're OVERPRICED")
        elif fav_avg > 0.005:
            findings.append(f"Favorites (>70%) making +{fav_avg*100:.2f}¢/share - they're UNDERPRICED")

    if findings:
        print("\nKey findings:")
        for f in findings:
            print(f"  - {f}")
    else:
        print("\nNo significant mispricing detected at probability extremes.")

    print("\n" + "=" * 80)
    print("Analysis complete!")


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "markouts/detailed_fills.csv"
    df = load_fills(filepath)

    df, results = longshot_favorite_analysis(df)
    longshot_vs_favorite_comparison(df, results)
    continuous_probability_analysis(df)
    extreme_longshot_analysis(df)
    summary(df)


if __name__ == "__main__":
    main()
