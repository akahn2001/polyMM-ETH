"""
Parameter Analysis Script
=========================
Analyzes detailed_fills.csv to determine optimal values for:
- MAX_TOTAL_SIGNAL_ADJUSTMENT
- MAX_Z_SCORE_SKEW
- MAX_IMBALANCE_ADJUSTMENT

Run: python analyze_parameters.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
MARKOUTS_FILE = "markouts/detailed_fills.csv"

# Current parameter values (from trading_config.py)
CURRENT_MAX_Z_SCORE_SKEW = 0.035           # 3.5 cents
CURRENT_MAX_IMBALANCE_ADJUSTMENT = 0.035   # 3.5 cents
CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT = 0.025  # 2.5 cents (binding constraint)


def load_data():
    """Load markouts data."""
    if not Path(MARKOUTS_FILE).exists():
        print(f"ERROR: {MARKOUTS_FILE} not found!")
        return None

    try:
        df = pd.read_csv(MARKOUTS_FILE, on_bad_lines='skip')
        print(f"Loaded {len(df)} fills from {MARKOUTS_FILE}\n")

        # Calculate per-share markouts
        for horizon in [1, 5, 15, 30, 60]:
            col = f'markout_{horizon}s'
            per_share_col = f'markout_{horizon}s_per_share'
            if col in df.columns and 'qty' in df.columns:
                df[per_share_col] = df[col] / df['qty']

        return df

    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return None


def unfavorable_fill_analysis(df):
    """Identify fills trading AGAINST z_skew signal."""
    print("=" * 80)
    print("UNFAVORABLE FILL ANALYSIS")
    print("=" * 80)

    if 'z_skew' not in df.columns:
        print("  ERROR: z_skew column not found")
        return None, None

    # Filter to fills with z_skew data
    df_skew = df[df['z_skew'].notna()].copy()
    print(f"Fills with z_skew data: {len(df_skew)} / {len(df)}")

    # Define unfavorable: trading against the z_skew signal
    # z_skew > 0 means we expect RTDS to rise → YES worth more → should BUY
    # z_skew < 0 means we expect RTDS to fall → YES worth less → should SELL
    # Unfavorable = bought when z_skew < 0 OR sold when z_skew > 0

    # Use threshold of 0.003 (0.3 cents) to filter noise
    SIGNAL_THRESHOLD = 0.003

    unfavorable = df_skew[
        ((df_skew['z_skew'] > SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == -1)) |  # sold when should buy
        ((df_skew['z_skew'] < -SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == 1))    # bought when should sell
    ].copy()

    favorable = df_skew[
        ((df_skew['z_skew'] > SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == 1)) |   # bought when should buy
        ((df_skew['z_skew'] < -SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == -1))   # sold when should sell
    ].copy()

    neutral = df_skew[abs(df_skew['z_skew']) <= SIGNAL_THRESHOLD].copy()

    print(f"\nBy direction alignment (threshold={SIGNAL_THRESHOLD*100:.1f} cents):")
    print(f"  Favorable (with signal):     {len(favorable):4d} fills ({len(favorable)/len(df_skew)*100:.1f}%)")
    print(f"  Unfavorable (against signal):{len(unfavorable):4d} fills ({len(unfavorable)/len(df_skew)*100:.1f}%)")
    print(f"  Neutral (|z_skew| < threshold): {len(neutral):4d} fills ({len(neutral)/len(df_skew)*100:.1f}%)")

    if len(unfavorable) > 0 and 'markout_5s_per_share' in df.columns:
        unfav_markout = unfavorable['markout_5s_per_share'].mean()
        fav_markout = favorable['markout_5s_per_share'].mean() if len(favorable) > 0 else 0
        total_unfav_pnl = unfavorable['markout_5s'].sum() if 'markout_5s' in df.columns else 0

        print(f"\nPerformance:")
        print(f"  Unfavorable avg markout: {unfav_markout*100:+.2f} cents/share")
        print(f"  Favorable avg markout:   {fav_markout*100:+.2f} cents/share")
        print(f"  Difference:              {(fav_markout-unfav_markout)*100:+.2f} cents/share")
        print(f"  Total unfavorable PNL:   ${total_unfav_pnl:+.2f}")

    # Analyze z_skew distribution at unfavorable fills
    print(f"\n{'-'*60}")
    print("Z-SKEW DISTRIBUTION AT UNFAVORABLE FILLS")
    print(f"{'-'*60}")

    if len(unfavorable) > 0:
        # Calculate the effective z_skew cap (accounting for total signal cap)
        # If both signals fire in same direction, total cap limits z_skew to ~2.5 cents
        EFFECTIVE_CAP = CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT  # 0.025 = 2.5 cents

        at_cap = unfavorable[abs(unfavorable['z_skew']) >= EFFECTIVE_CAP - 0.001]
        near_cap = unfavorable[(abs(unfavorable['z_skew']) >= EFFECTIVE_CAP - 0.005) &
                               (abs(unfavorable['z_skew']) < EFFECTIVE_CAP - 0.001)]
        below_cap = unfavorable[abs(unfavorable['z_skew']) < EFFECTIVE_CAP - 0.005]

        print(f"  At cap (|z_skew| >= {(EFFECTIVE_CAP-0.001)*100:.1f} cents):   {len(at_cap):3d} fills ({len(at_cap)/len(unfavorable)*100:.1f}%)")
        print(f"  Near cap ({(EFFECTIVE_CAP-0.005)*100:.1f}-{(EFFECTIVE_CAP-0.001)*100:.1f} cents): {len(near_cap):3d} fills ({len(near_cap)/len(unfavorable)*100:.1f}%)")
        print(f"  Below cap (< {(EFFECTIVE_CAP-0.005)*100:.1f} cents):       {len(below_cap):3d} fills ({len(below_cap)/len(unfavorable)*100:.1f}%)")

        print(f"\n  Distribution stats:")
        print(f"    Mean |z_skew|:  {unfavorable['z_skew'].abs().mean()*100:.2f} cents")
        print(f"    Median |z_skew|:{unfavorable['z_skew'].abs().median()*100:.2f} cents")
        print(f"    Max |z_skew|:   {unfavorable['z_skew'].abs().max()*100:.2f} cents")

    return unfavorable, favorable


def cap_hit_analysis(df):
    """Analyze how often caps are being hit and performance when capped vs uncapped."""
    print(f"\n{'='*80}")
    print("CAP HIT ANALYSIS")
    print("='*80")

    if 'z_skew' not in df.columns:
        print("  ERROR: z_skew column not found")
        return

    df_skew = df[df['z_skew'].notna()].copy()

    # Z-skew cap analysis
    Z_CAP = CURRENT_MAX_Z_SCORE_SKEW  # 0.035 = 3.5 cents
    TOTAL_CAP = CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT  # 0.025 = 2.5 cents

    # The actual effective cap depends on whether book imbalance is also active
    # For now, assume both signals fire together so effective cap is TOTAL_CAP
    EFFECTIVE_CAP = TOTAL_CAP

    z_capped = df_skew[abs(df_skew['z_skew']) >= EFFECTIVE_CAP - 0.001]
    z_uncapped = df_skew[abs(df_skew['z_skew']) < EFFECTIVE_CAP - 0.001]

    print(f"\nZ-skew hitting effective cap (~{EFFECTIVE_CAP*100:.1f} cents):")
    print(f"  Capped fills:   {len(z_capped):4d} ({len(z_capped)/len(df_skew)*100:.1f}%)")
    print(f"  Uncapped fills: {len(z_uncapped):4d} ({len(z_uncapped)/len(df_skew)*100:.1f}%)")

    if 'markout_5s_per_share' in df.columns:
        if len(z_capped) >= 10:
            capped_markout = z_capped['markout_5s_per_share'].mean()
            print(f"  Capped avg markout:   {capped_markout*100:+.2f} cents/share")
        if len(z_uncapped) >= 10:
            uncapped_markout = z_uncapped['markout_5s_per_share'].mean()
            print(f"  Uncapped avg markout: {uncapped_markout*100:+.2f} cents/share")

    # Analyze by z_skew magnitude buckets
    print(f"\n{'-'*60}")
    print("PERFORMANCE BY Z-SKEW MAGNITUDE")
    print(f"{'-'*60}")

    buckets = [
        (0.000, 0.005, "0.0-0.5c"),
        (0.005, 0.010, "0.5-1.0c"),
        (0.010, 0.015, "1.0-1.5c"),
        (0.015, 0.020, "1.5-2.0c"),
        (0.020, 0.025, "2.0-2.5c"),
        (0.025, 0.030, "2.5-3.0c"),
        (0.030, 0.040, "3.0-4.0c"),
        (0.040, 0.100, "4.0c+"),
    ]

    print(f"{'Bucket':<12} {'Fills':>8} {'Avg mkout':>12} {'Total PNL':>12} {'Fav%':>8}")
    print("-" * 60)

    for low, high, label in buckets:
        bucket = df_skew[(abs(df_skew['z_skew']) >= low) & (abs(df_skew['z_skew']) < high)]
        if len(bucket) >= 5:
            avg_mkout = bucket['markout_5s_per_share'].mean() if 'markout_5s_per_share' in df.columns else 0
            total_pnl = bucket['markout_5s'].sum() if 'markout_5s' in df.columns else 0

            # Calculate favorable % for this bucket
            fav = bucket[
                ((bucket['z_skew'] > 0) & (bucket['dir_yes'] == 1)) |
                ((bucket['z_skew'] < 0) & (bucket['dir_yes'] == -1))
            ]
            fav_pct = len(fav) / len(bucket) * 100 if len(bucket) > 0 else 0

            print(f"{label:<12} {len(bucket):>8d} {avg_mkout*100:>+11.2f}c ${total_pnl:>+10.2f} {fav_pct:>7.1f}%")
        else:
            print(f"{label:<12} {len(bucket):>8d} (insufficient data)")


def estimate_required_skew(df):
    """For unfavorable fills, estimate what skew would have pushed quote far enough to avoid fill."""
    print(f"\n{'='*80}")
    print("REQUIRED SKEW ESTIMATION")
    print("='*80")

    if 'z_skew' not in df.columns:
        print("  ERROR: z_skew column not found")
        return

    df_skew = df[df['z_skew'].notna()].copy()

    # Unfavorable fills: trading against signal
    SIGNAL_THRESHOLD = 0.003
    unfavorable = df_skew[
        ((df_skew['z_skew'] > SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == -1)) |
        ((df_skew['z_skew'] < -SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == 1))
    ].copy()

    if len(unfavorable) == 0:
        print("  No unfavorable fills found")
        return

    # The key insight: if z_skew was X but we got filled on the wrong side,
    # we need skew > X + some buffer to avoid the fill
    # The buffer depends on how close our quote was to getting lifted

    # Calculate the "required skew" to push quote 1 tick away
    # Simplified: required_skew = |z_skew| + TICK_SIZE (0.01)
    TICK_SIZE = 0.01

    unfavorable['required_skew'] = unfavorable['z_skew'].abs() + TICK_SIZE

    print(f"Analyzing {len(unfavorable)} unfavorable fills\n")
    print("Required skew to avoid unfavorable fills (|z_skew| + 1 tick):")

    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(unfavorable['required_skew'], p)
        print(f"  {p}th percentile: {val*100:.2f} cents")

    print(f"\n  Mean required:   {unfavorable['required_skew'].mean()*100:.2f} cents")
    print(f"  Max required:    {unfavorable['required_skew'].max()*100:.2f} cents")

    # Show what % of unfavorable fills would be avoided at each cap level
    print(f"\n{'-'*60}")
    print("UNFAVORABLE FILLS AVOIDED BY CAP LEVEL")
    print(f"{'-'*60}")

    caps = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]
    print(f"{'Cap':>8} {'Avoided':>10} {'Pct':>8} {'Est PNL saved':>14}")
    print("-" * 45)

    # Get unfavorable PNL per fill for estimation
    avg_unfav_loss = unfavorable['markout_5s'].mean() if 'markout_5s' in df.columns else 0

    for cap in caps:
        # Fills avoided if required_skew <= cap (i.e., our skew would have been enough)
        avoided = unfavorable[unfavorable['required_skew'] <= cap]
        pct = len(avoided) / len(unfavorable) * 100 if len(unfavorable) > 0 else 0
        pnl_saved = -avoided['markout_5s'].sum() if 'markout_5s' in df.columns and len(avoided) > 0 else 0
        print(f"{cap*100:>7.1f}c {len(avoided):>10d} {pct:>7.1f}% ${pnl_saved:>+12.2f}")


def signal_effectiveness_by_magnitude(df):
    """Analyze signal effectiveness at different z_skew magnitudes."""
    print(f"\n{'='*80}")
    print("SIGNAL EFFECTIVENESS BY MAGNITUDE")
    print("='*80")

    if 'z_skew' not in df.columns or 'markout_5s_per_share' not in df.columns:
        print("  ERROR: Required columns not found")
        return

    df_skew = df[df['z_skew'].notna()].copy()

    # Create aligned_z_skew: positive when trading WITH signal
    df_skew['aligned_z_skew'] = df_skew['z_skew'] * df_skew['dir_yes']

    # Performance for fills following vs against the signal
    print("\nPerformance when FOLLOWING signal (aligned_z_skew > 0):")
    following = df_skew[df_skew['aligned_z_skew'] > 0]
    against = df_skew[df_skew['aligned_z_skew'] < 0]

    print(f"  Following signal: {len(following)} fills, avg markout {following['markout_5s_per_share'].mean()*100:+.2f}c/share")
    print(f"  Against signal:   {len(against)} fills, avg markout {against['markout_5s_per_share'].mean()*100:+.2f}c/share")

    # Break down by |z_skew| magnitude for following-signal fills only
    print(f"\n{'-'*60}")
    print("FOLLOWING-SIGNAL FILLS BY Z_SKEW MAGNITUDE")
    print(f"{'-'*60}")

    buckets = [
        (0.003, 0.010, "0.3-1.0c"),
        (0.010, 0.015, "1.0-1.5c"),
        (0.015, 0.020, "1.5-2.0c"),
        (0.020, 0.025, "2.0-2.5c"),
        (0.025, 0.035, "2.5-3.5c"),
        (0.035, 0.050, "3.5-5.0c"),
        (0.050, 0.100, "5.0c+"),
    ]

    print(f"{'|z_skew|':<12} {'Fills':>8} {'Avg mkout':>12} {'Sharpe':>10}")
    print("-" * 50)

    for low, high, label in buckets:
        bucket = following[(abs(following['z_skew']) >= low) & (abs(following['z_skew']) < high)]
        if len(bucket) >= 10:
            avg_mkout = bucket['markout_5s_per_share'].mean()
            std_mkout = bucket['markout_5s_per_share'].std()
            sharpe = avg_mkout / std_mkout if std_mkout > 0 else 0
            print(f"{label:<12} {len(bucket):>8d} {avg_mkout*100:>+11.2f}c {sharpe:>10.3f}")
        else:
            print(f"{label:<12} {len(bucket):>8d} (insufficient data)")


def book_imbalance_effectiveness(df):
    """Analyze book imbalance signal effectiveness."""
    print(f"\n{'='*80}")
    print("BOOK IMBALANCE SIGNAL EFFECTIVENESS")
    print("='*80")

    if 'book_imbalance' not in df.columns:
        print("  ERROR: book_imbalance column not found")
        return

    df_imb = df[df['book_imbalance'].notna()].copy()
    print(f"Fills with book_imbalance data: {len(df_imb)} / {len(df)}")

    if len(df_imb) < 50:
        print("  Insufficient data for analysis")
        return

    # aligned_imbalance: positive when trading WITH imbalance signal
    # book_imbalance > 0 means more bids → expect price up → should buy
    df_imb['aligned_imbalance'] = df_imb['book_imbalance'] * df_imb['dir_yes']

    # Performance by imbalance bucket
    print(f"\n{'-'*60}")
    print("PERFORMANCE BY BOOK IMBALANCE MAGNITUDE")
    print(f"{'-'*60}")

    buckets = [
        (0.0, 0.2, "0-0.2"),
        (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
        (0.8, 1.0, "0.8+"),
    ]

    print(f"{'|imb|':<12} {'Fills':>8} {'Avg mkout':>12} {'Follow%':>10}")
    print("-" * 50)

    for low, high, label in buckets:
        bucket = df_imb[(abs(df_imb['book_imbalance']) >= low) & (abs(df_imb['book_imbalance']) < high)]
        if len(bucket) >= 10:
            avg_mkout = bucket['markout_5s_per_share'].mean() if 'markout_5s_per_share' in df.columns else 0
            following = bucket[bucket['aligned_imbalance'] > 0]
            follow_pct = len(following) / len(bucket) * 100
            print(f"{label:<12} {len(bucket):>8d} {avg_mkout*100:>+11.2f}c {follow_pct:>9.1f}%")
        else:
            print(f"{label:<12} {len(bucket):>8d} (insufficient data)")


def recommendations(df):
    """Generate parameter recommendations based on analysis."""
    print(f"\n{'='*80}")
    print("PARAMETER RECOMMENDATIONS")
    print("='*80")

    if 'z_skew' not in df.columns:
        print("  ERROR: z_skew column not found")
        return

    df_skew = df[df['z_skew'].notna()].copy()

    # Unfavorable fills analysis
    SIGNAL_THRESHOLD = 0.003
    unfavorable = df_skew[
        ((df_skew['z_skew'] > SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == -1)) |
        ((df_skew['z_skew'] < -SIGNAL_THRESHOLD) & (df_skew['dir_yes'] == 1))
    ].copy()

    TICK_SIZE = 0.01
    if len(unfavorable) > 0:
        unfavorable['required_skew'] = unfavorable['z_skew'].abs() + TICK_SIZE
        p90_required = np.percentile(unfavorable['required_skew'], 90)
    else:
        p90_required = 0.035  # default

    # Find sweet spot from signal effectiveness
    df_skew['aligned_z_skew'] = df_skew['z_skew'] * df_skew['dir_yes']
    following = df_skew[df_skew['aligned_z_skew'] > 0]

    # Find magnitude where signal is most effective
    best_bucket_markout = -999
    best_bucket = None
    buckets = [(0.015, 0.025), (0.020, 0.030), (0.025, 0.035), (0.030, 0.040), (0.035, 0.050)]
    for low, high in buckets:
        bucket = following[(abs(following['z_skew']) >= low) & (abs(following['z_skew']) < high)]
        if len(bucket) >= 20:
            avg = bucket['markout_5s_per_share'].mean() if 'markout_5s_per_share' in df.columns else 0
            if avg > best_bucket_markout:
                best_bucket_markout = avg
                best_bucket = (low, high)

    print("\nCURRENT SETTINGS:")
    print(f"  MAX_Z_SCORE_SKEW:           {CURRENT_MAX_Z_SCORE_SKEW*100:.1f}c ({CURRENT_MAX_Z_SCORE_SKEW})")
    print(f"  MAX_IMBALANCE_ADJUSTMENT:   {CURRENT_MAX_IMBALANCE_ADJUSTMENT*100:.1f}c ({CURRENT_MAX_IMBALANCE_ADJUSTMENT})")
    print(f"  MAX_TOTAL_SIGNAL_ADJUSTMENT:{CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT*100:.1f}c ({CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT})")

    print(f"\n{'-'*60}")
    print("ANALYSIS-BASED RECOMMENDATIONS")
    print(f"{'-'*60}")

    # Recommendation for MAX_TOTAL_SIGNAL_ADJUSTMENT
    # Based on 90th percentile of required skew to avoid unfavorable fills
    rec_total = min(p90_required, 0.05)  # Cap at 5 cents
    rec_total = round(rec_total * 200) / 200  # Round to nearest 0.5 cent

    print(f"\n1. MAX_TOTAL_SIGNAL_ADJUSTMENT:")
    print(f"   Current: {CURRENT_MAX_TOTAL_SIGNAL_ADJUSTMENT*100:.1f}c")
    print(f"   Suggested: {rec_total*100:.1f}c ({rec_total})")
    print(f"   Rationale: 90th percentile of required skew to avoid unfavorable fills = {p90_required*100:.2f}c")

    # Recommendation for MAX_Z_SCORE_SKEW
    # Should be at least as high as MAX_TOTAL_SIGNAL_ADJUSTMENT
    rec_zskew = max(rec_total, CURRENT_MAX_Z_SCORE_SKEW)

    print(f"\n2. MAX_Z_SCORE_SKEW:")
    print(f"   Current: {CURRENT_MAX_Z_SCORE_SKEW*100:.1f}c")
    print(f"   Suggested: {rec_zskew*100:.1f}c ({rec_zskew})")
    print(f"   Rationale: Should be >= MAX_TOTAL_SIGNAL_ADJUSTMENT")

    # Recommendation for MAX_IMBALANCE_ADJUSTMENT
    # Keep current unless book imbalance analysis suggests otherwise
    rec_imb = CURRENT_MAX_IMBALANCE_ADJUSTMENT

    print(f"\n3. MAX_IMBALANCE_ADJUSTMENT:")
    print(f"   Current: {CURRENT_MAX_IMBALANCE_ADJUSTMENT*100:.1f}c")
    print(f"   Suggested: Keep at {rec_imb*100:.1f}c ({rec_imb})")
    print(f"   Rationale: Book imbalance is secondary signal; focus on z_skew first")

    # Expected improvement
    if len(unfavorable) > 0 and 'markout_5s' in df.columns:
        avoided_at_rec = unfavorable[unfavorable['required_skew'] <= rec_total]
        pnl_saved = -avoided_at_rec['markout_5s'].sum() if len(avoided_at_rec) > 0 else 0

        print(f"\n{'-'*60}")
        print("EXPECTED IMPROVEMENT")
        print(f"{'-'*60}")
        print(f"  Unfavorable fills that would be avoided: ~{len(avoided_at_rec)} ({len(avoided_at_rec)/len(unfavorable)*100:.1f}%)")
        print(f"  Estimated PNL saved: ~${pnl_saved:.2f}")
        print(f"  (Note: This is a rough estimate; actual results may vary)")

    print(f"\n{'='*80}")
    print("COPY-PASTE VALUES FOR trading_config.py")
    print("='*80")
    print(f"MAX_Z_SCORE_SKEW = {rec_zskew}           # {rec_zskew*100:.1f} cents")
    print(f"MAX_IMBALANCE_ADJUSTMENT = {rec_imb}    # {rec_imb*100:.1f} cents")
    print(f"MAX_TOTAL_SIGNAL_ADJUSTMENT = {rec_total}  # {rec_total*100:.1f} cents")


def main():
    print("=" * 80)
    print("PARAMETER ANALYSIS REPORT")
    print("=" * 80)

    df = load_data()
    if df is None:
        return

    # Run all analyses
    unfavorable_fill_analysis(df)
    cap_hit_analysis(df)
    estimate_required_skew(df)
    signal_effectiveness_by_magnitude(df)
    book_imbalance_effectiveness(df)
    recommendations(df)

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
