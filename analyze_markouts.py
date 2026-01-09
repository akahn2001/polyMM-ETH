"""
Markouts Analysis Script

Analyzes detailed_fills.csv to diagnose trading performance and identify issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
MARKOUTS_FILE = "markouts/detailed_fills.csv"
MIN_SAMPLES_FOR_SIGNIFICANCE = 10


def significance_test(data, null_value=0, name="metric"):
    """
    Run a t-test to check if data is significantly different from null_value.
    Returns (t_stat, p_value, is_significant, interpretation)
    """
    data = data.dropna()
    n = len(data)

    if n < MIN_SAMPLES_FOR_SIGNIFICANCE:
        return None, None, False, f"⚠️  Too few samples (n={n}) for significance test"

    t_stat, p_value = stats.ttest_1samp(data, null_value)
    mean = data.mean()
    std_err = data.std() / np.sqrt(n)
    ci_95 = (mean - 1.96 * std_err, mean + 1.96 * std_err)

    if p_value < 0.01:
        sig = "***"  # p < 0.01
        is_sig = True
    elif p_value < 0.05:
        sig = "**"   # p < 0.05
        is_sig = True
    elif p_value < 0.10:
        sig = "*"    # p < 0.10
        is_sig = False
    else:
        sig = ""
        is_sig = False

    direction = "positive" if mean > null_value else "negative"
    interp = f"mean={mean:.5f} {sig} (p={p_value:.4f}, n={n}, 95%CI=[{ci_95[0]:.5f}, {ci_95[1]:.5f}])"

    return t_stat, p_value, is_sig, interp


def compare_groups(group1, group2, name1="Group1", name2="Group2"):
    """
    Run a t-test comparing two groups.
    Returns interpretation string.
    """
    g1 = group1.dropna()
    g2 = group2.dropna()

    if len(g1) < MIN_SAMPLES_FOR_SIGNIFICANCE or len(g2) < MIN_SAMPLES_FOR_SIGNIFICANCE:
        return f"⚠️  Too few samples ({name1}={len(g1)}, {name2}={len(g2)}) for comparison"

    t_stat, p_value = stats.ttest_ind(g1, g2)
    diff = g1.mean() - g2.mean()

    if p_value < 0.01:
        sig = "***"
        verdict = "SIGNIFICANT difference"
    elif p_value < 0.05:
        sig = "**"
        verdict = "SIGNIFICANT difference"
    elif p_value < 0.10:
        sig = "*"
        verdict = "marginally significant"
    else:
        sig = ""
        verdict = "NOT significant"

    return f"Δ={diff:+.5f} {sig} (p={p_value:.4f}) - {verdict}"


def load_data():
    """Load markouts data."""
    if not Path(MARKOUTS_FILE).exists():
        print(f"ERROR: {MARKOUTS_FILE} not found!")
        print("Make sure the bot has been running and markout_dump_loop has executed.")
        return None

    try:
        # Read the header to check format
        with open(MARKOUTS_FILE, 'r') as f:
            header = f.readline().strip()

        has_momentum_volatility = 'momentum_volatility' in header

        if not has_momentum_volatility:
            print("⚠️  Old CSV format detected (no momentum_volatility column)")
            print("   Loading with old format, new fills will be skipped\n")
            # Use old format column names
            df = pd.read_csv(MARKOUTS_FILE, on_bad_lines='skip')
            # Add missing column with NaN
            df['momentum_volatility'] = float('nan')
        else:
            # New format
            df = pd.read_csv(MARKOUTS_FILE, on_bad_lines='skip')

        print(f"Loaded {len(df)} fills from {MARKOUTS_FILE}\n")

        # Check if we skipped any bad lines
        with open(MARKOUTS_FILE, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # -1 for header

        if len(df) < total_lines:
            skipped = total_lines - len(df)
            print(f"⚠️  WARNING: Skipped {skipped} malformed lines")
            print("   This happens when CSV format changed mid-run.")
            print("   Recommendation: Delete CSV and restart bot for clean data:")
            print(f"     del {MARKOUTS_FILE}\n")

        # CRITICAL: Calculate per-share markouts
        # The markout columns in CSV contain TOTAL PNL (already multiplied by qty)
        # We need per-share markouts for proper averaging
        for horizon in [1, 5, 15, 30, 60]:
            col = f'markout_{horizon}s'
            per_share_col = f'markout_{horizon}s_per_share'
            if col in df.columns and 'qty' in df.columns:
                df[per_share_col] = df[col] / df['qty']

        print(f"✓ Calculated per-share markouts (dividing by quantity)\n")

        return df

    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        print("\nThe CSV file is corrupted or has mixed formats.")
        print("Delete it and restart the bot:")
        print(f"  del {MARKOUTS_FILE}")
        return None

def overall_performance(df):
    """Overall P&L and performance metrics."""
    print("=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    total_shares = df['qty'].sum()
    print(f"Total fills: {len(df)}")
    print(f"Total shares traded: {total_shares:.0f}")
    print(f"Buy fills: {(df['dir_yes'] == 1).sum()}")
    print(f"Sell fills: {(df['dir_yes'] == -1).sum()}")

    print(f"\n{'='*80}")
    print(f"PER-SHARE MARKOUT PERFORMANCE (Edge in cents per share)")
    print(f"{'='*80}")
    print(f"  Legend: *** p<0.01, ** p<0.05, * p<0.10")
    print()
    for horizon in [1, 5, 15, 30, 60]:
        per_share_col = f'markout_{horizon}s_per_share'
        if per_share_col in df.columns:
            avg_per_share = df[per_share_col].mean()
            median_per_share = df[per_share_col].median()
            hit_rate = (df[per_share_col] > 0).sum() / len(df) * 100
            _, p_val, is_sig, interp = significance_test(df[per_share_col], null_value=0)

            # Handle None p_val (not enough samples)
            if p_val is not None:
                sig_stars = '*** ' if p_val < 0.01 else '** ' if p_val < 0.05 else '* ' if p_val < 0.10 else ''
                print(f"  {horizon}s: mean={avg_per_share:.5f} ({sig_stars}p={p_val:.4f})")
            else:
                print(f"  {horizon}s: mean={avg_per_share:.5f} (insufficient samples for significance test)")
            print(f"       median={median_per_share:.5f}, hit rate={hit_rate:.1f}%")

    print(f"\n{'='*80}")
    print(f"TOTAL PNL (Cumulative profit across all fills)")
    print(f"{'='*80}")
    for horizon in [1, 5, 15, 30, 60]:
        col = f'markout_{horizon}s'
        if col in df.columns:
            total_pnl = df[col].sum()
            per_share_col = f'markout_{horizon}s_per_share'
            avg_per_share = df[per_share_col].mean() if per_share_col in df.columns else 0
            print(f"  {horizon}s: Total PNL = ${total_pnl:+.2f}  (avg {avg_per_share*100:+.2f}¢/share × {total_shares:.0f} shares)")

    print(f"\nDelta Check:")
    print(f"  Avg delta: {df['delta'].mean():.6f}")
    print(f"  Delta = 0 count: {(df['delta'] == 0).sum()} ({(df['delta'] == 0).sum()/len(df)*100:.1f}%)")
    if df['delta'].mean() < 0.00001:
        print("  ⚠️  WARNING: Delta is near zero! Momentum adjustment not working!")

def momentum_analysis(df):
    """Analyze momentum strategy performance."""
    print("\n" + "=" * 80)
    print("MOMENTUM STRATEGY ANALYSIS")
    print("=" * 80)

    if 'momentum' not in df.columns:
        print("  ⚠️  momentum not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Directionally-adjusted correlation
    # aligned_momentum = momentum * dir_yes
    # Positive when trading WITH momentum (buy when rising, sell when falling)
    df['aligned_momentum'] = df['momentum'] * df['dir_yes']

    mom_corr = df['aligned_momentum'].corr(df['markout_5s_per_share'])
    print(f"Directionally-adjusted Momentum → Markout correlation: {mom_corr:.3f}")
    if mom_corr > 0.1:
        print("  ✅ Positive correlation - momentum has predictive power!")
    elif mom_corr > 0:
        print("  ⚠️  Weak positive correlation - momentum helps slightly")
    else:
        print("  ❌ Negative/zero correlation - momentum NOT working!")

    # Performance by momentum regime
    print(f"\nPerformance by momentum:")

    rising = df[df['momentum'] > 10]
    falling = df[df['momentum'] < -10]
    flat = df[abs(df['momentum']) <= 10]

    print(f"  Rising (>$10):   {len(rising):3d} fills, avg markout=${rising['markout_5s_per_share'].mean():.4f}")
    _, _, _, rising_sig = significance_test(rising['markout_5s_per_share']) if len(rising) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if rising_sig: print(f"                   {rising_sig}")

    print(f"  Falling (<-$10): {len(falling):3d} fills, avg markout=${falling['markout_5s_per_share'].mean():.4f}")
    _, _, _, falling_sig = significance_test(falling['markout_5s_per_share']) if len(falling) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if falling_sig: print(f"                   {falling_sig}")

    print(f"  Flat (±$10):     {len(flat):3d} fills, avg markout=${flat['markout_5s_per_share'].mean():.4f}")
    _, _, _, flat_sig = significance_test(flat['markout_5s_per_share']) if len(flat) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if flat_sig: print(f"                   {flat_sig}")

    # Should momentum fills have better markouts?
    momentum_fills = df[abs(df['momentum']) > 10]
    flat_fills = df[abs(df['momentum']) <= 10]

    print(f"\n  Momentum fills (|mom|>$10): avg markout=${momentum_fills['markout_5s_per_share'].mean():.4f}")
    print(f"  Flat fills (|mom|≤$10):     avg markout=${flat_fills['markout_5s_per_share'].mean():.4f}")

    # Statistical comparison
    comparison = compare_groups(momentum_fills['markout_5s_per_share'], flat_fills['markout_5s_per_share'], "Momentum", "Flat")
    print(f"  Comparison: {comparison}")

    if momentum_fills['markout_5s_per_share'].mean() > flat_fills['markout_5s_per_share'].mean():
        print("  ✅ Momentum fills are more profitable!")
    else:
        print("  ❌ Momentum fills are WORSE - strategy may be broken")

def theo_value_test(df):
    """Test if Black-Scholes theo is more valuable than market mid as reference price."""
    print("\n" + "=" * 80)
    print("THEO VALUE TEST: Black-Scholes Theo vs Market Mid")
    print("=" * 80)

    if 'theo' not in df.columns or 'market_mid' not in df.columns:
        print("  ⚠️  theo or market_mid not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Calculate edge vs theo (Black-Scholes)
    # For buys: positive = bought below theo value
    # For sells: positive = sold above theo value
    df['edge_vs_theo'] = df.apply(
        lambda row: (row['theo'] - row['fill_yes']) if row['dir_yes'] > 0 else (row['fill_yes'] - row['theo']),
        axis=1
    )

    # Calculate edge vs market mid
    # For buys: positive = bought below market mid
    # For sells: positive = sold above market mid
    df['edge_vs_market_mid'] = df.apply(
        lambda row: (row['market_mid'] - row['fill_yes']) if row['dir_yes'] > 0 else (row['fill_yes'] - row['market_mid']),
        axis=1
    )

    # Test correlation with markouts
    theo_corr = df['edge_vs_theo'].corr(df['markout_5s_per_share'])
    mid_corr = df['edge_vs_market_mid'].corr(df['markout_5s_per_share'])

    print(f"Which reference price is more predictive of markouts?")
    print(f"  Edge vs Theo (BS model)   → Markout correlation: {theo_corr:.3f}")
    print(f"  Edge vs Market Mid        → Markout correlation: {mid_corr:.3f}")

    if theo_corr > mid_corr + 0.05:
        print(f"  ✅ Black-Scholes theo is MORE valuable than market mid!")
    elif mid_corr > theo_corr + 0.05:
        print(f"  ⚠️  Market mid is MORE valuable than Black-Scholes theo")
    else:
        print(f"  ≈  Similar predictive value")

    # Compare average markouts for trades following theo vs market signals
    print(f"\nWhen theo disagrees with market (|theo - market_mid| > 2¢):")

    # Theo says cheap, market says rich → bought
    theo_cheap_bought = df[(df['theo'] - df['market_mid'] > 0.02) & (df['dir_yes'] == 1)]
    if len(theo_cheap_bought) > 0:
        print(f"  Bought when theo>mid by >2¢: {len(theo_cheap_bought):3d} fills, "
              f"avg markout=${theo_cheap_bought['markout_5s_per_share'].mean():.4f}")

    # Theo says rich, market says cheap → sold
    theo_rich_sold = df[(df['market_mid'] - df['theo'] > 0.02) & (df['dir_yes'] == -1)]
    if len(theo_rich_sold) > 0:
        print(f"  Sold when mid>theo by >2¢:   {len(theo_rich_sold):3d} fills, "
              f"avg markout=${theo_rich_sold['markout_5s_per_share'].mean():.4f}")

    # Performance by edge vs theo buckets
    print(f"\nMarkout performance by edge vs Black-Scholes theo:")
    df['edge_bucket'] = pd.cut(df['edge_vs_theo'], bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                 labels=['Large -ve', 'Small -ve', 'Neutral', 'Small +ve', 'Large +ve'])

    for bucket in df['edge_bucket'].cat.categories:
        subset = df[df['edge_bucket'] == bucket]
        if len(subset) > 0:
            avg_markout = subset['markout_5s_per_share'].mean()
            avg_theo_edge = subset['edge_vs_theo'].mean()
            print(f"  {bucket:12s}: {len(subset):3d} fills, "
                  f"avg edge=${avg_theo_edge:+.4f}, "
                  f"avg markout=${avg_markout:.4f}")

    print(f"\n  Expected: Large +ve edge → positive markouts")
    print(f"  Expected: Large -ve edge → negative markouts")

    # Summary stats
    avg_theo_vs_mid = (df['theo'] - df['market_mid']).mean()
    print(f"\nAverage theo - market_mid: ${avg_theo_vs_mid:+.4f}")
    if abs(avg_theo_vs_mid) > 0.01:
        if avg_theo_vs_mid > 0:
            print(f"  → Black-Scholes model consistently prices higher than market")
        else:
            print(f"  → Black-Scholes model consistently prices lower than market")

def model_vs_market_test(df):
    """Test if model disagreement with market is valuable."""
    print("\n" + "=" * 80)
    print("MODEL DISAGREEMENT TEST")
    print("=" * 80)

    # When model thinks market is wrong, and you trade accordingly
    model_thinks_cheap = df[(df['model_vs_market'] > 0.02) & (df['dir_yes'] == 1)]
    model_thinks_rich = df[(df['model_vs_market'] < -0.02) & (df['dir_yes'] == -1)]

    print(f"Bought when model said underpriced (model>market by >2¢):")
    print(f"  {len(model_thinks_cheap)} fills, avg markout=${model_thinks_cheap['markout_5s_per_share'].mean():.4f}")

    print(f"\nSold when model said overpriced (model<market by >2¢):")
    print(f"  {len(model_thinks_rich)} fills, avg markout=${model_thinks_rich['markout_5s_per_share'].mean():.4f}")

    if len(model_thinks_cheap) > 0 and model_thinks_cheap['markout_5s_per_share'].mean() > 0:
        print("  ✅ Model contrarian calls are profitable!")
    else:
        print("  ❌ Model contrarian calls are NOT profitable")

def adverse_selection_test(df):
    """Check if getting adverse selected."""
    print("\n" + "=" * 80)
    print("ADVERSE SELECTION TEST")
    print("=" * 80)

    # edge_vs_fair: negative = paid up (crossed spread), positive = got filled inside quote
    paid_up = df[df['edge_vs_fair'] < -0.005]  # Paid up >0.5¢
    got_edge = df[df['edge_vs_fair'] > 0.005]  # Got filled with edge
    neutral = df[abs(df['edge_vs_fair']) <= 0.005]

    print(f"Fills where you paid up (edge_vs_fair < -0.5¢):")
    print(f"  {len(paid_up)} fills, avg markout=${paid_up['markout_5s_per_share'].mean():.4f}")
    if paid_up['markout_5s_per_share'].mean() < 0:
        print("  ❌ Paid up fills have negative markouts - getting picked off!")

    print(f"\nFills where you got edge (edge_vs_fair > +0.5¢):")
    print(f"  {len(got_edge)} fills, avg markout=${got_edge['markout_5s_per_share'].mean():.4f}")

    print(f"\nFills at fair value (|edge_vs_fair| ≤ 0.5¢):")
    print(f"  {len(neutral)} fills, avg markout=${neutral['markout_5s_per_share'].mean():.4f}")

def inventory_analysis(df):
    """Analyze inventory and position management."""
    print("\n" + "=" * 80)
    print("INVENTORY ANALYSIS")
    print("=" * 80)

    print(f"Position statistics:")
    print(f"  Max long position: {df['net_yes_after'].max():.1f}")
    print(f"  Max short position: {df['net_yes_after'].min():.1f}")
    print(f"  Avg position: {df['net_yes_after'].mean():.1f}")
    print(f"  Final position: {df['net_yes_after'].iloc[-1]:.1f}")

    # Performance when building vs reducing position
    df['increasing_long'] = (df['dir_yes'] == 1) & (df['net_yes_before'] >= 0)
    df['increasing_short'] = (df['dir_yes'] == -1) & (df['net_yes_before'] <= 0)
    df['reducing'] = ((df['dir_yes'] == 1) & (df['net_yes_before'] < 0)) | \
                     ((df['dir_yes'] == -1) & (df['net_yes_before'] > 0))

    building = df[df['increasing_long'] | df['increasing_short']]
    reducing = df[df['reducing']]

    print(f"\nBuilding position: {len(building)} fills, avg markout=${building['markout_5s_per_share'].mean():.4f}")
    print(f"Reducing position: {len(reducing)} fills, avg markout=${reducing['markout_5s_per_share'].mean():.4f}")

    if building['markout_5s_per_share'].mean() < reducing['markout_5s_per_share'].mean():
        print("  ❌ Worse markouts when building - getting run over on entries")
    else:
        print("  ✅ Better markouts when building - good position management")

def directional_bias(df):
    """Check for directional bias in fills."""
    print("\n" + "=" * 80)
    print("DIRECTIONAL BIAS")
    print("=" * 80)

    buys = df[df['dir_yes'] == 1]
    sells = df[df['dir_yes'] == -1]

    print(f"Buy performance:  {len(buys)} fills, avg markout=${buys['markout_5s_per_share'].mean():.4f}")
    _, _, _, buy_sig = significance_test(buys['markout_5s_per_share']) if len(buys) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if buy_sig: print(f"                  {buy_sig}")

    print(f"Sell performance: {len(sells)} fills, avg markout=${sells['markout_5s_per_share'].mean():.4f}")
    _, _, _, sell_sig = significance_test(sells['markout_5s_per_share']) if len(sells) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if sell_sig: print(f"                  {sell_sig}")

    # Statistical comparison
    comparison = compare_groups(buys['markout_5s_per_share'], sells['markout_5s_per_share'], "Buys", "Sells")
    print(f"\nBuy vs Sell comparison: {comparison}")

    if abs(buys['markout_5s_per_share'].mean() - sells['markout_5s_per_share'].mean()) > 0.005:
        print(f"  ⚠️  Large difference - one side is much worse!")
    else:
        print(f"  ✅ Balanced performance on both sides")

def volatility_spread_analysis(df):
    """Analyze if dynamic spread based on volatility helps."""
    print("\n" + "=" * 80)
    print("VOLATILITY-BASED SPREAD ANALYSIS")
    print("=" * 80)

    if 'momentum_volatility' not in df.columns:
        print("  ⚠️  momentum_volatility not in data - run with new markouts code")
        return

    # Categorize by volatility
    high_vol = df[df['momentum_volatility'] > 20]
    med_vol = df[(df['momentum_volatility'] > 10) & (df['momentum_volatility'] <= 20)]
    low_vol = df[df['momentum_volatility'] <= 10]

    print(f"Performance by momentum volatility:")
    print(f"  Low volatility (≤$10):  {len(low_vol):3d} fills, avg markout=${low_vol['markout_5s_per_share'].mean():.4f}")
    print(f"  Med volatility ($10-20): {len(med_vol):3d} fills, avg markout=${med_vol['markout_5s_per_share'].mean():.4f}")
    print(f"  High volatility (>$20):  {len(high_vol):3d} fills, avg markout=${high_vol['markout_5s_per_share'].mean():.4f}")

    # Check if high volatility fills have worse markouts (justifying wider spreads)
    if len(high_vol) > 0 and len(low_vol) > 0:
        if high_vol['markout_5s_per_share'].mean() < low_vol['markout_5s_per_share'].mean():
            print(f"  ❌ High volatility fills are WORSE - dynamic spread helps!")
            print(f"     Widening spread during volatile times should improve performance")
        else:
            print(f"  ✅ High volatility fills are OK - dynamic spread may not be needed")

    # Check edge_vs_fair in high volatility (are you getting picked off?)
    if len(high_vol) > 0:
        paid_up_in_vol = high_vol[high_vol['edge_vs_fair'] < -0.005]
        print(f"\n  During high volatility:")
        print(f"    Paid up (edge<-0.5¢): {len(paid_up_in_vol)} fills, avg markout=${paid_up_in_vol['markout_5s_per_share'].mean():.4f}")
        if len(paid_up_in_vol) > 0:
            print(f"    ⚠️  Getting picked off during volatile periods - widen spreads!")

def book_imbalance_analysis(df):
    """Analyze if book imbalance signal has predictive value."""
    print("\n" + "=" * 80)
    print("BOOK IMBALANCE ANALYSIS")
    print("=" * 80)

    if 'book_imbalance' not in df.columns:
        print("  ⚠️  book_imbalance not in data - run with updated code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Filter out NaN values
    df_imb = df[df['book_imbalance'].notna()].copy()
    if len(df_imb) == 0:
        print("  ⚠️  No book imbalance data available")
        return

    # Directionally-adjusted correlation
    # aligned_imbalance = book_imbalance * dir_yes
    # Positive when trading WITH the flow (buy when bid>ask, sell when ask>bid)
    df_imb['aligned_imbalance'] = df_imb['book_imbalance'] * df_imb['dir_yes']

    imb_corr = df_imb['aligned_imbalance'].corr(df_imb['markout_5s_per_share'])
    print(f"Directionally-adjusted Book Imbalance → Markout correlation: {imb_corr:.3f}")
    if imb_corr > 0.1:
        print("  ✅ Positive correlation - imbalance has predictive power!")
    elif imb_corr > 0:
        print("  ⚠️  Weak positive correlation - imbalance helps slightly")
    else:
        print("  ❌ Negative/zero correlation - imbalance NOT predictive")

    # Performance by imbalance regime
    print(f"\nPerformance by book imbalance:")
    strong_bid = df_imb[df_imb['book_imbalance'] > 0.3]
    strong_ask = df_imb[df_imb['book_imbalance'] < -0.3]
    neutral = df_imb[abs(df_imb['book_imbalance']) <= 0.3]

    print(f"  Strong bid imbalance (>0.3):  {len(strong_bid):3d} fills, avg markout=${strong_bid['markout_5s_per_share'].mean():.4f}")
    print(f"  Neutral (±0.3):               {len(neutral):3d} fills, avg markout=${neutral['markout_5s_per_share'].mean():.4f}")
    print(f"  Strong ask imbalance (<-0.3): {len(strong_ask):3d} fills, avg markout=${strong_ask['markout_5s_per_share'].mean():.4f}")

    # Check if trading WITH imbalance is better than against
    # When imbalance > 0 and we buy (dir_yes = 1), we're trading WITH the flow
    # When imbalance < 0 and we sell (dir_yes = -1), we're trading WITH the flow
    with_flow = df_imb[((df_imb['book_imbalance'] > 0.2) & (df_imb['dir_yes'] == 1)) |
                       ((df_imb['book_imbalance'] < -0.2) & (df_imb['dir_yes'] == -1))]
    against_flow = df_imb[((df_imb['book_imbalance'] > 0.2) & (df_imb['dir_yes'] == -1)) |
                          ((df_imb['book_imbalance'] < -0.2) & (df_imb['dir_yes'] == 1))]

    print(f"\nTrading WITH order flow:     {len(with_flow):3d} fills, avg markout=${with_flow['markout_5s_per_share'].mean():.4f}")
    print(f"Trading AGAINST order flow:  {len(against_flow):3d} fills, avg markout=${against_flow['markout_5s_per_share'].mean():.4f}")

    if len(with_flow) > 0 and len(against_flow) > 0:
        comparison = compare_groups(with_flow['markout_5s_per_share'], against_flow['markout_5s_per_share'], "With Flow", "Against Flow")
        print(f"  Comparison: {comparison}")

        if with_flow['markout_5s_per_share'].mean() > against_flow['markout_5s_per_share'].mean():
            print("  ✅ Trading WITH order flow is more profitable!")
            print("     → Book imbalance adjustment is helping")
        else:
            print("  ❌ Trading WITH order flow is WORSE")
            print("     → Consider disabling USE_BOOK_IMBALANCE or reducing MAX_IMBALANCE_ADJUSTMENT")

    # Summary stats
    print(f"\nBook imbalance statistics:")
    print(f"  Mean imbalance: {df_imb['book_imbalance'].mean():.3f}")
    print(f"  Std imbalance:  {df_imb['book_imbalance'].std():.3f}")
    print(f"  Min/Max:        {df_imb['book_imbalance'].min():.3f} / {df_imb['book_imbalance'].max():.3f}")

def zscore_predictor_analysis(df):
    """Analyze if Coinbase-RTDS z-score predictor is working."""
    print("\n" + "=" * 80)
    print("Z-SCORE PREDICTOR ANALYSIS (Coinbase-RTDS Spread)")
    print("=" * 80)

    if 'zscore' not in df.columns:
        print("  ⚠️  zscore not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Filter out fills without z-score data
    df_z = df[df['zscore'].notna()].copy()
    if len(df_z) == 0:
        print("  ⚠️  No z-score data available")
        return

    print(f"Fills with z-score data: {len(df_z)} / {len(df)}")

    # Directionally-adjusted correlation
    # aligned_zscore = zscore * dir_yes
    # Positive when trading WITH the signal (buy when z>0, sell when z<0)
    df_z['aligned_zscore'] = df_z['zscore'] * df_z['dir_yes']

    z_corr = df_z['aligned_zscore'].corr(df_z['markout_5s_per_share'])
    print(f"\nDirectionally-adjusted Z-score → Markout correlation: {z_corr:.3f}")
    if z_corr > 0.1:
        print("  ✅ Positive correlation - z-score has predictive power!")
    elif z_corr > 0:
        print("  ⚠️  Weak positive correlation - z-score helps slightly")
    else:
        print("  ❌ Negative/zero correlation - z-score NOT predictive")

    # Key insight: We should see DIRECTIONAL effects
    # High z-score means RTDS will rise → we cancel ASK, keep BID
    # When we get filled on BID (dir_yes=+1) during high z-score, we should profit
    # Low z-score means RTDS will fall → we cancel BID, keep ASK
    # When we get filled on ASK (dir_yes=-1) during low z-score, we should profit

    # Performance by z-score regime
    print(f"\nPerformance by z-score regime:")
    high_pos_z = df_z[df_z['zscore'] > 0.8]
    med_pos_z = df_z[(df_z['zscore'] > 0.3) & (df_z['zscore'] <= 0.8)]
    neutral_z = df_z[abs(df_z['zscore']) <= 0.3]
    med_neg_z = df_z[(df_z['zscore'] < -0.3) & (df_z['zscore'] >= -0.8)]
    high_neg_z = df_z[df_z['zscore'] < -0.8]

    print(f"  High +z (>0.8):     {len(high_pos_z):3d} fills, avg markout=${high_pos_z['markout_5s_per_share'].mean():.4f}")
    print(f"  Med +z (0.3-0.8):   {len(med_pos_z):3d} fills, avg markout=${med_pos_z['markout_5s_per_share'].mean():.4f}")
    print(f"  Neutral (±0.3):     {len(neutral_z):3d} fills, avg markout=${neutral_z['markout_5s_per_share'].mean():.4f}")
    print(f"  Med -z (-0.8--0.3): {len(med_neg_z):3d} fills, avg markout=${med_neg_z['markout_5s_per_share'].mean():.4f}")
    print(f"  High -z (<-0.8):    {len(high_neg_z):3d} fills, avg markout=${high_neg_z['markout_5s_per_share'].mean():.4f}")

    # Z-score predictor logic test
    print(f"\nZ-score predictor effectiveness:")
    print(f"  (Testing if fills during extreme z-score have worse markouts)")

    extreme_z = df_z[abs(df_z['zscore']) > 0.8]
    neutral_z_fills = df_z[abs(df_z['zscore']) <= 0.3]

    if len(extreme_z) > 0 and len(neutral_z_fills) > 0:
        print(f"  Extreme z (|z|>0.8): {len(extreme_z)} fills, avg markout=${extreme_z['markout_5s_per_share'].mean():.4f}")
        print(f"  Neutral z (|z|≤0.3): {len(neutral_z_fills)} fills, avg markout=${neutral_z_fills['markout_5s_per_share'].mean():.4f}")

        comparison = compare_groups(extreme_z['markout_5s_per_share'], neutral_z_fills['markout_5s_per_share'],
                                   "Extreme Z", "Neutral Z")
        print(f"  {comparison}")

        if extreme_z['markout_5s_per_share'].mean() < neutral_z_fills['markout_5s_per_share'].mean():
            print(f"  ✅ Extreme z-score fills are WORSE - predictor is valuable!")
            print(f"     → Canceling during extreme z-score should improve performance")
        else:
            print(f"  ❌ Extreme z-score fills are NOT worse")
            print(f"     → Z-score predictor may not be working as intended")

    # Directional z-score test: Does the predictor work as intended?
    print(f"\nDirectional z-score test:")
    print(f"  Theory: High +z → RTDS will rise → Cancel ASK, keep BID")
    print(f"          Low -z → RTDS will fall → Cancel BID, keep ASK")

    # When z > 0.8 and we bought (dir_yes=+1), did we make money?
    bought_high_z = df_z[(df_z['zscore'] > 0.8) & (df_z['dir_yes'] == 1)]
    # When z < -0.8 and we sold (dir_yes=-1), did we make money?
    sold_low_z = df_z[(df_z['zscore'] < -0.8) & (df_z['dir_yes'] == -1)]

    # WRONG side fills (should have been canceled):
    # When z > 0.8 and we sold (dir_yes=-1), this is BAD (ASK should have been canceled)
    sold_high_z = df_z[(df_z['zscore'] > 0.8) & (df_z['dir_yes'] == -1)]
    # When z < -0.8 and we bought (dir_yes=+1), this is BAD (BID should have been canceled)
    bought_low_z = df_z[(df_z['zscore'] < -0.8) & (df_z['dir_yes'] == 1)]

    print(f"\n  CORRECT side (should be profitable):")
    print(f"    Bought when z>0.8 (RTDS will rise):  {len(bought_high_z)} fills, avg markout=${bought_high_z['markout_5s_per_share'].mean():.4f}")
    print(f"    Sold when z<-0.8 (RTDS will fall):   {len(sold_low_z)} fills, avg markout=${sold_low_z['markout_5s_per_share'].mean():.4f}")

    print(f"\n  WRONG side (should be canceled by z-score logic):")
    print(f"    Sold when z>0.8 (vulnerable ASK):    {len(sold_high_z)} fills, avg markout=${sold_high_z['markout_5s_per_share'].mean():.4f}")
    print(f"    Bought when z<-0.8 (vulnerable BID): {len(bought_low_z)} fills, avg markout=${bought_low_z['markout_5s_per_share'].mean():.4f}")

    if len(sold_high_z) > 0 or len(bought_low_z) > 0:
        print(f"\n  ⚠️  WARNING: Getting filled on VULNERABLE side despite cancel logic!")
        print(f"     → Either threshold (0.80) is too high, or cancels aren't fast enough")
        if len(sold_high_z) > 0:
            print(f"     → {len(sold_high_z)} fills on ASK when z>0.8 (should be canceled)")
        if len(bought_low_z) > 0:
            print(f"     → {len(bought_low_z)} fills on BID when z<-0.8 (should be canceled)")

    # Compare correct vs wrong side
    correct_side = pd.concat([bought_high_z, sold_low_z])
    wrong_side = pd.concat([sold_high_z, bought_low_z])

    if len(correct_side) > 0 and len(wrong_side) > 0:
        comparison = compare_groups(correct_side['markout_5s_per_share'], wrong_side['markout_5s_per_share'],
                                   "Correct Side", "Wrong Side")
        print(f"\n  Comparison: {comparison}")

        if correct_side['markout_5s_per_share'].mean() > wrong_side['markout_5s_per_share'].mean():
            print(f"  ✅ Correct side fills are MORE profitable!")
            print(f"     → Z-score predictor is working as intended")
        else:
            print(f"  ❌ Correct side fills are NOT better")
            print(f"     → Z-score predictor may be broken or threshold is wrong")

    # Z-score statistics
    print(f"\nZ-score statistics:")
    print(f"  Mean z-score:  {df_z['zscore'].mean():+.3f}")
    print(f"  Std z-score:   {df_z['zscore'].std():.3f}")
    print(f"  Min/Max:       {df_z['zscore'].min():+.3f} / {df_z['zscore'].max():+.3f}")
    print(f"  |z|>0.8 count: {(abs(df_z['zscore']) > 0.8).sum()} ({(abs(df_z['zscore']) > 0.8).sum()/len(df_z)*100:.1f}%)")
    print(f"  |z|>1.0 count: {(abs(df_z['zscore']) > 1.0).sum()} ({(abs(df_z['zscore']) > 1.0).sum()/len(df_z)*100:.1f}%)")
    print(f"  |z|>2.0 count: {(abs(df_z['zscore']) > 2.0).sum()} ({(abs(df_z['zscore']) > 2.0).sum()/len(df_z)*100:.1f}%)")


def z_skew_analysis(df):
    """Analyze if z-score skew (continuous fair value adjustment) improves performance."""
    print("\n" + "=" * 80)
    print("Z-SCORE SKEW ANALYSIS (Continuous Fair Value Adjustment)")
    print("=" * 80)

    if 'z_skew' not in df.columns:
        print("  ⚠️  z_skew not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Filter out fills without z_skew data
    df_skew = df[df['z_skew'].notna()].copy()
    if len(df_skew) == 0:
        print("  ⚠️  No z-score skew data available")
        return

    print(f"Fills with z_skew data: {len(df_skew)} / {len(df)}")

    # Directionally-adjusted correlation
    # aligned_z_skew = z_skew * dir_yes
    # Positive when trading WITH the signal (buy when z_skew>0, sell when z_skew<0)
    df_skew['aligned_z_skew'] = df_skew['z_skew'] * df_skew['dir_yes']

    skew_corr = df_skew['aligned_z_skew'].corr(df_skew['markout_5s_per_share'])
    print(f"\nDirectionally-adjusted z-skew → Markout correlation: {skew_corr:.3f}")
    if skew_corr > 0.15:
        print("  ✅ Strong positive correlation - z-skew is predictive!")
    elif skew_corr > 0.05:
        print("  ⚠️  Weak positive correlation - z-skew has some value")
    else:
        print("  ❌ No/negative correlation - z-skew is NOT predictive")

    # Key insight: Z-skew adjusts fair value based on predicted RTDS movement
    # Positive z_skew means we think RTDS will rise (YES worth more)
    # Negative z_skew means we think RTDS will fall (YES worth less)
    # Favorable direction: when sign(z_skew) == sign(dir_yes)

    # Performance by z_skew regime
    print(f"\nPerformance by z-skew regime:")
    high_pos = df_skew[df_skew['z_skew'] > 0.005]
    med_pos = df_skew[(df_skew['z_skew'] > 0.001) & (df_skew['z_skew'] <= 0.005)]
    neutral = df_skew[abs(df_skew['z_skew']) <= 0.001]
    med_neg = df_skew[(df_skew['z_skew'] < -0.001) & (df_skew['z_skew'] >= -0.005)]
    high_neg = df_skew[df_skew['z_skew'] < -0.005]

    print(f"  High +skew (>0.5¢):     {len(high_pos):3d} fills, avg markout=${high_pos['markout_5s_per_share'].mean():.4f}")
    print(f"  Med +skew (0.1-0.5¢):   {len(med_pos):3d} fills, avg markout=${med_pos['markout_5s_per_share'].mean():.4f}")
    print(f"  Neutral (±0.1¢):        {len(neutral):3d} fills, avg markout=${neutral['markout_5s_per_share'].mean():.4f}")
    print(f"  Med -skew (-0.5--0.1¢): {len(med_neg):3d} fills, avg markout=${med_neg['markout_5s_per_share'].mean():.4f}")
    print(f"  High -skew (<-0.5¢):    {len(high_neg):3d} fills, avg markout=${high_neg['markout_5s_per_share'].mean():.4f}")

    # Directional z_skew test
    print(f"\nDirectional z_skew test:")
    print(f"  Theory: +skew → RTDS will rise → Buy YES")
    print(f"          -skew → RTDS will fall → Sell YES")

    # Favorable direction: sign(z_skew) == sign(dir_yes)
    # When z_skew > 0 and we bought (dir_yes=+1), that's favorable
    bought_pos_skew = df_skew[(df_skew['z_skew'] > 0.003) & (df_skew['dir_yes'] == 1)]
    # When z_skew < 0 and we sold (dir_yes=-1), that's favorable
    sold_neg_skew = df_skew[(df_skew['z_skew'] < -0.003) & (df_skew['dir_yes'] == -1)]

    # Unfavorable direction: sign(z_skew) != sign(dir_yes)
    # When z_skew > 0 and we sold (dir_yes=-1), that's unfavorable
    sold_pos_skew = df_skew[(df_skew['z_skew'] > 0.003) & (df_skew['dir_yes'] == -1)]
    # When z_skew < 0 and we bought (dir_yes=+1), that's unfavorable
    bought_neg_skew = df_skew[(df_skew['z_skew'] < -0.003) & (df_skew['dir_yes'] == 1)]

    print(f"\n  FAVORABLE direction (trading WITH z-skew signal):")
    print(f"    Bought when z_skew>+0.3¢:  {len(bought_pos_skew)} fills, avg markout=${bought_pos_skew['markout_5s_per_share'].mean():.4f}")
    print(f"    Sold when z_skew<-0.3¢:    {len(sold_neg_skew)} fills, avg markout=${sold_neg_skew['markout_5s_per_share'].mean():.4f}")

    print(f"\n  UNFAVORABLE direction (trading AGAINST z-skew signal):")
    print(f"    Sold when z_skew>+0.3¢:    {len(sold_pos_skew)} fills, avg markout=${sold_pos_skew['markout_5s_per_share'].mean():.4f}")
    print(f"    Bought when z_skew<-0.3¢:  {len(bought_neg_skew)} fills, avg markout=${bought_neg_skew['markout_5s_per_share'].mean():.4f}")

    # Compare favorable vs unfavorable
    favorable = pd.concat([bought_pos_skew, sold_neg_skew])
    unfavorable = pd.concat([sold_pos_skew, bought_neg_skew])

    if len(favorable) > 0 and len(unfavorable) > 0:
        comparison = compare_groups(favorable['markout_5s_per_share'], unfavorable['markout_5s_per_share'],
                                   "Favorable", "Unfavorable")
        print(f"\n  Comparison: {comparison}")

        if favorable['markout_5s_per_share'].mean() > unfavorable['markout_5s_per_share'].mean():
            print(f"  ✅ Favorable direction fills are MORE profitable!")
            print(f"     → Z-score skew is working as intended")
        else:
            print(f"  ❌ Favorable direction fills are NOT better")
            print(f"     → Z-score skew may not be predictive")

    # Test if z_skew improves overall performance
    print(f"\nZ-skew effectiveness test:")
    extreme_skew = df_skew[abs(df_skew['z_skew']) > 0.005]
    neutral_skew = df_skew[abs(df_skew['z_skew']) <= 0.001]

    if len(extreme_skew) > 0 and len(neutral_skew) > 0:
        print(f"  Extreme skew (|skew|>0.5¢): {len(extreme_skew)} fills, avg markout=${extreme_skew['markout_5s_per_share'].mean():.4f}")
        print(f"  Neutral skew (|skew|≤0.1¢): {len(neutral_skew)} fills, avg markout=${neutral_skew['markout_5s_per_share'].mean():.4f}")

        comparison = compare_groups(extreme_skew['markout_5s_per_share'], neutral_skew['markout_5s_per_share'],
                                   "Extreme Skew", "Neutral Skew")
        print(f"  {comparison}")

        if extreme_skew['markout_5s_per_share'].mean() > neutral_skew['markout_5s_per_share'].mean():
            print(f"  ✅ Extreme skew fills are BETTER - z-skew adjustment is valuable!")
        else:
            print(f"  ⚠️  Extreme skew fills are NOT better - signal may be weak")

    # Z-skew statistics
    print(f"\nZ-skew statistics:")
    print(f"  Mean z_skew:   {df_skew['z_skew'].mean():+.4f} cents")
    print(f"  Std z_skew:    {df_skew['z_skew'].std():.4f} cents")
    print(f"  Min/Max:       {df_skew['z_skew'].min():+.4f} / {df_skew['z_skew'].max():+.4f} cents")
    print(f"  |skew|>0.5¢:   {(abs(df_skew['z_skew']) > 0.005).sum()} ({(abs(df_skew['z_skew']) > 0.005).sum()/len(df_skew)*100:.1f}%)")
    print(f"  |skew|>1.0¢:   {(abs(df_skew['z_skew']) > 0.010).sum()} ({(abs(df_skew['z_skew']) > 0.010).sum()/len(df_skew)*100:.1f}%)")
    print(f"  At cap (1.5¢): {(abs(df_skew['z_skew']) >= 0.0149).sum()} ({(abs(df_skew['z_skew']) >= 0.0149).sum()/len(df_skew)*100:.1f}%)")

    # Compare z_skew vs zscore
    if 'zscore' in df_skew.columns:
        df_both = df_skew[(df_skew['zscore'].notna()) & (df_skew['z_skew'].notna())]
        if len(df_both) > 30:
            print(f"\nZ-skew vs Z-score correlation:")
            skew_z_corr = df_both['z_skew'].corr(df_both['zscore'])
            print(f"  z_skew ↔ zscore correlation: {skew_z_corr:.3f}")
            if skew_z_corr > 0.7:
                print(f"  ✅ Strong correlation - z_skew properly derived from z-score")
            elif skew_z_corr > 0.3:
                print(f"  ⚠️  Moderate correlation - delta modulation working")
            else:
                print(f"  ❌ Weak correlation - check z_skew calculation")


def z_skew_residual_analysis(df):
    """Analyze z_skew residual approach - does subtracting market_implied_move improve edge?"""
    print("\n" + "=" * 80)
    print("Z-SKEW RESIDUAL ANALYSIS")
    print("=" * 80)

    required_cols = ['z_skew', 'market_implied_move', 'z_skew_residual']
    if not all(col in df.columns for col in required_cols):
        print("  ⚠️  Residual columns not in data - run with updated code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    print(f"Analyzing {len(df)} fills with z_skew residual data\n")

    # 1. How much does market typically price in?
    avg_z_skew = df['z_skew'].abs().mean()
    avg_market_implied = df['market_implied_move'].abs().mean()
    avg_residual = df['z_skew_residual'].abs().mean()

    print(f"Signal decomposition (absolute values):")
    print(f"  Avg |z_skew| (full prediction):     {avg_z_skew*100:.2f}¢")
    print(f"  Avg |market_implied| (already priced): {avg_market_implied*100:.2f}¢")
    print(f"  Avg |residual| (what we apply):     {avg_residual*100:.2f}¢")

    pct_applied = (avg_residual / avg_z_skew * 100) if avg_z_skew > 0 else 0
    print(f"  → We apply {pct_applied:.1f}% of the full z_skew on average\n")

    # 2. Agreement vs disagreement with market
    # Same sign = market moved in predicted direction but not fully
    # Opposite sign = market overshot or moved wrong direction
    df['z_market_agree'] = (df['z_skew'] * df['market_implied_move']) > 0
    df['z_residual_agree'] = (df['z_skew'] * df['z_skew_residual']) > 0

    agree = df[df['z_market_agree'] == True]
    disagree = df[df['z_market_agree'] == False]

    print(f"Market vs Z-score agreement:")
    print(f"  Agree (same direction):   {len(agree):3d} fills ({len(agree)/len(df)*100:.1f}%)")
    if len(agree) > 0:
        print(f"    → Avg |market_implied|: {agree['market_implied_move'].abs().mean()*100:.2f}¢")
        print(f"    → Avg markout: ${agree['markout_5s_per_share'].mean():.4f}")

    print(f"  Disagree (opposite dirs): {len(disagree):3d} fills ({len(disagree)/len(df)*100:.1f}%)")
    if len(disagree) > 0:
        print(f"    → Avg |market_implied|: {disagree['market_implied_move'].abs().mean()*100:.2f}¢")
        print(f"    → Avg markout: ${disagree['markout_5s_per_share'].mean():.4f}")

    # 3. When residual flips sign (market overshot), are we right to fade it?
    same_sign = df[df['z_residual_agree'] == True]
    flipped_sign = df[df['z_residual_agree'] == False]

    print(f"\nResidual sign flips:")
    print(f"  Residual same sign as z_skew:     {len(same_sign):3d} fills ({len(same_sign)/len(df)*100:.1f}%)")
    if len(same_sign) > 0:
        print(f"    → Avg residual: {same_sign['z_skew_residual'].mean()*100:+.2f}¢")
        print(f"    → Avg markout: ${same_sign['markout_5s_per_share'].mean():.4f}")

    print(f"  Residual flipped sign (fading mkt): {len(flipped_sign):3d} fills ({len(flipped_sign)/len(df)*100:.1f}%)")
    if len(flipped_sign) > 0:
        print(f"    → Avg residual: {flipped_sign['z_skew_residual'].mean()*100:+.2f}¢")
        print(f"    → Avg markout: ${flipped_sign['markout_5s_per_share'].mean():.4f}")
        if flipped_sign['markout_5s_per_share'].mean() > 0:
            print(f"    ✅ Profitable to fade market when it overshoots!")
        else:
            print(f"    ❌ Losing money when fading market - residual approach may be wrong")

    # 4. Predictive power comparison - what matters is following the signal
    # Full z_skew: when signal is positive, did buying work? when negative, did selling work?
    full_bullish_buys = df[(df['z_skew'] > 0) & (df['dir_yes'] == 1)]
    full_bearish_sells = df[(df['z_skew'] < 0) & (df['dir_yes'] == -1)]
    full_signal_following = pd.concat([full_bullish_buys, full_bearish_sells])

    # Residual z_skew: same logic
    res_bullish_buys = df[(df['z_skew_residual'] > 0) & (df['dir_yes'] == 1)]
    res_bearish_sells = df[(df['z_skew_residual'] < 0) & (df['dir_yes'] == -1)]
    res_signal_following = pd.concat([res_bullish_buys, res_bearish_sells])

    print(f"\nPredictive power (avg markout when FOLLOWING signal):")
    print(f"\nFull z_skew approach:")
    if len(full_signal_following) > 0:
        full_avg = full_signal_following['markout_5s_per_share'].mean()
        print(f"  Bought when z_skew>0: {len(full_bullish_buys):3d} fills, avg markout=${full_bullish_buys['markout_5s_per_share'].mean():.4f}" if len(full_bullish_buys) > 0 else "")
        print(f"  Sold when z_skew<0:   {len(full_bearish_sells):3d} fills, avg markout=${full_bearish_sells['markout_5s_per_share'].mean():.4f}" if len(full_bearish_sells) > 0 else "")
        print(f"  → TOTAL following signal: {len(full_signal_following):3d} fills, avg markout=${full_avg:.4f}")
    else:
        full_avg = 0
        print(f"  No fills following full z_skew signal")

    print(f"\nResidual z_skew approach:")
    if len(res_signal_following) > 0:
        res_avg = res_signal_following['markout_5s_per_share'].mean()
        print(f"  Bought when residual>0: {len(res_bullish_buys):3d} fills, avg markout=${res_bullish_buys['markout_5s_per_share'].mean():.4f}" if len(res_bullish_buys) > 0 else "")
        print(f"  Sold when residual<0:   {len(res_bearish_sells):3d} fills, avg markout=${res_bearish_sells['markout_5s_per_share'].mean():.4f}" if len(res_bearish_sells) > 0 else "")
        print(f"  → TOTAL following signal: {len(res_signal_following):3d} fills, avg markout=${res_avg:.4f}")
    else:
        res_avg = 0
        print(f"  No fills following residual z_skew signal")

    print(f"\nComparison:")
    if res_avg > full_avg + 0.001:
        print(f"  ✅ Residual approach is BETTER by {(res_avg - full_avg)*100:.2f}¢/share")
        print(f"     → Subtracting market_implied improves edge!")
    elif full_avg > res_avg + 0.001:
        print(f"  ❌ Full z_skew is BETTER by {(full_avg - res_avg)*100:.2f}¢/share")
        print(f"     → Residual approach hurts performance, revert to Option 1 or 2")
    else:
        print(f"  ≈  Similar performance (within 0.1¢/share)")

    # 5. Bucketed analysis
    print(f"\nMarkouts by market_implied_move magnitude:")
    df['market_priced_bucket'] = pd.cut(df['market_implied_move'].abs(),
                                         bins=[0, 0.01, 0.02, 0.04, np.inf],
                                         labels=['<1¢', '1-2¢', '2-4¢', '>4¢'])

    for bucket in df['market_priced_bucket'].cat.categories:
        subset = df[df['market_priced_bucket'] == bucket]
        if len(subset) > 0:
            avg_z = subset['z_skew'].abs().mean()
            avg_res = subset['z_skew_residual'].abs().mean()
            avg_markout = subset['markout_5s_per_share'].mean()
            print(f"  {bucket:6s}: {len(subset):3d} fills, |z|={avg_z*100:.2f}¢ → |res|={avg_res*100:.2f}¢, markout=${avg_markout:.4f}")


def sigmoid_zscore_analysis(df):
    """Analyze sigmoid confidence scaling - does filtering by z-score magnitude improve edge?"""
    print("\n" + "=" * 80)
    print("SIGMOID Z-SCORE CONFIDENCE ANALYSIS")
    print("=" * 80)

    required_cols = ['zscore', 'z_skew', 'dir_yes', 'markout_5s_per_share']
    if not all(col in df.columns for col in required_cols):
        print("  ⚠️  Required columns not in data")
        return

    # Filter to only fills where we had a z-score signal
    df_with_z = df[df['zscore'].abs() > 0.001].copy()

    if len(df_with_z) == 0:
        print("  ⚠️  No fills with z-score signal")
        return

    print(f"Analyzing {len(df_with_z)} fills with z-score signal\n")

    # 1. Overall z-score distribution
    avg_zscore = df_with_z['zscore'].abs().mean()
    max_zscore = df_with_z['zscore'].abs().max()

    print(f"Z-score distribution:")
    print(f"  Avg |zscore|: {avg_zscore:.2f}")
    print(f"  Max |zscore|: {max_zscore:.2f}")

    # Sigmoid confidence parameters (should match trading_config.py)
    MIDPOINT = 0.4
    STEEPNESS = 5.0

    # Calculate confidence for each fill
    df_with_z['z_confidence'] = 1.0 / (1.0 + np.exp(-STEEPNESS * (df_with_z['zscore'].abs() - MIDPOINT)))
    avg_confidence = df_with_z['z_confidence'].mean()
    print(f"  Avg confidence: {avg_confidence:.2f} (sigmoid with midpoint={MIDPOINT}, steepness={STEEPNESS})\n")

    # 2. Bucket by |zscore| magnitude to see confidence effect
    print(f"Markouts by z-score magnitude (validates sigmoid filtering):")

    # Define buckets aligned with sigmoid confidence levels
    bins = [0, 0.2, 0.4, 0.6, 1.0, np.inf]
    labels = ['<0.2', '0.2-0.4', '0.4-0.6', '0.6-1.0', '>1.0']
    df_with_z['zscore_bucket'] = pd.cut(df_with_z['zscore'].abs(), bins=bins, labels=labels)

    for bucket in labels:
        subset = df_with_z[df_with_z['zscore_bucket'] == bucket]
        if len(subset) == 0:
            continue

        avg_conf = subset['z_confidence'].mean()
        avg_zscore_val = subset['zscore'].abs().mean()
        avg_z_skew = subset['z_skew'].abs().mean()

        # Calculate markouts when following signal direction
        following = subset[((subset['z_skew'] > 0) & (subset['dir_yes'] == 1)) |
                          ((subset['z_skew'] < 0) & (subset['dir_yes'] == -1))]

        # Calculate markouts when going against signal direction
        against = subset[((subset['z_skew'] > 0) & (subset['dir_yes'] == -1)) |
                        ((subset['z_skew'] < 0) & (subset['dir_yes'] == 1))]

        avg_markout = subset['markout_5s_per_share'].mean()
        avg_markout_following = following['markout_5s_per_share'].mean() if len(following) > 0 else 0
        avg_markout_against = against['markout_5s_per_share'].mean() if len(against) > 0 else 0

        print(f"  |z|={bucket:8s}: {len(subset):3d} fills, conf={avg_conf:.2f}, "
              f"|z_skew|={avg_z_skew*100:.2f}¢")
        print(f"              Following signal: {len(following):3d} fills, "
              f"markout=${avg_markout_following:.4f}")
        if len(against) > 0:
            print(f"              Against signal:   {len(against):3d} fills, "
                  f"markout=${avg_markout_against:.4f}")

    # 3. Compare low vs high confidence fills
    print(f"\nLow vs High Confidence Performance:")

    low_conf = df_with_z[df_with_z['zscore'].abs() < MIDPOINT]  # Low confidence (<50%)
    high_conf = df_with_z[df_with_z['zscore'].abs() >= MIDPOINT]  # High confidence (≥50%)

    # Following signal for each group
    low_following = low_conf[((low_conf['z_skew'] > 0) & (low_conf['dir_yes'] == 1)) |
                             ((low_conf['z_skew'] < 0) & (low_conf['dir_yes'] == -1))]
    high_following = high_conf[((high_conf['z_skew'] > 0) & (high_conf['dir_yes'] == 1)) |
                               ((high_conf['z_skew'] < 0) & (high_conf['dir_yes'] == -1))]

    print(f"\nLow confidence (|z| < {MIDPOINT}):")
    print(f"  Total fills: {len(low_conf)}")
    print(f"  Following signal: {len(low_following)} fills, "
          f"avg markout=${low_following['markout_5s_per_share'].mean():.4f}" if len(low_following) > 0 else "  No fills following signal")

    print(f"\nHigh confidence (|z| ≥ {MIDPOINT}):")
    print(f"  Total fills: {len(high_conf)}")
    print(f"  Following signal: {len(high_following)} fills, "
          f"avg markout=${high_following['markout_5s_per_share'].mean():.4f}" if len(high_following) > 0 else "  No fills following signal")

    # 4. Statistical comparison
    if len(low_following) > 0 and len(high_following) > 0:
        low_avg = low_following['markout_5s_per_share'].mean()
        high_avg = high_following['markout_5s_per_share'].mean()

        print(f"\nSigmoid filtering effectiveness:")
        if high_avg > low_avg + 0.0005:
            print(f"  ✅ High confidence fills are BETTER by {(high_avg - low_avg)*100:.2f}¢/share")
            print(f"     → Sigmoid is correctly filtering noise!")
        elif low_avg > high_avg + 0.0005:
            print(f"  ⚠️  Low confidence fills are BETTER by {(low_avg - high_avg)*100:.2f}¢/share")
            print(f"     → Sigmoid may be over-filtering or parameters need tuning")
        else:
            print(f"  ≈  Similar performance (within 0.05¢/share)")
            print(f"     → Sigmoid has neutral effect, consider adjusting MIDPOINT/STEEPNESS")

    # 5. Directional analysis - favorable vs unfavorable
    print(f"\n" + "=" * 80)
    print(f"DIRECTIONAL ANALYSIS (Favorable vs Unfavorable)")
    print(f"=" * 80)

    # Favorable = following signal direction
    favorable = df_with_z[((df_with_z['z_skew'] > 0) & (df_with_z['dir_yes'] == 1)) |
                          ((df_with_z['z_skew'] < 0) & (df_with_z['dir_yes'] == -1))]

    # Unfavorable = against signal direction
    unfavorable = df_with_z[((df_with_z['z_skew'] > 0) & (df_with_z['dir_yes'] == -1)) |
                            ((df_with_z['z_skew'] < 0) & (df_with_z['dir_yes'] == 1))]

    print(f"\nFavorable direction (following z_skew signal):")
    if len(favorable) > 0:
        # Bucket by magnitude in favorable direction
        bins_mag = [0, 0.01, 0.02, 0.03, np.inf]
        labels_mag = ['<1¢', '1-2¢', '2-3¢', '>3¢']
        favorable['z_skew_mag'] = pd.cut(favorable['z_skew'].abs(), bins=bins_mag, labels=labels_mag)

        for mag_bucket in labels_mag:
            subset = favorable[favorable['z_skew_mag'] == mag_bucket]
            if len(subset) == 0:
                continue
            avg_markout = subset['markout_5s_per_share'].mean()
            avg_zscore = subset['zscore'].abs().mean()
            print(f"  |z_skew|={mag_bucket:6s}: {len(subset):3d} fills, "
                  f"avg |z|={avg_zscore:.2f}, markout=${avg_markout:.4f}")
    else:
        print("  No favorable fills")

    print(f"\nUnfavorable direction (against z_skew signal):")
    if len(unfavorable) > 0:
        # Bucket by magnitude in unfavorable direction
        unfavorable['z_skew_mag'] = pd.cut(unfavorable['z_skew'].abs(), bins=bins_mag, labels=labels_mag)

        for mag_bucket in labels_mag:
            subset = unfavorable[unfavorable['z_skew_mag'] == mag_bucket]
            if len(subset) == 0:
                continue
            avg_markout = subset['markout_5s_per_share'].mean()
            avg_zscore = subset['zscore'].abs().mean()
            print(f"  |z_skew|={mag_bucket:6s}: {len(subset):3d} fills, "
                  f"avg |z|={avg_zscore:.2f}, markout=${avg_markout:.4f}")
    else:
        print("  No unfavorable fills")


def realized_vol_theo_analysis(df):
    """Test if realized vol theo has predictive power vs implied vol theo."""
    print("\n" + "=" * 80)
    print("REALIZED VOL THEO ANALYSIS")
    print("=" * 80)

    if 'realized_vol_theo' not in df.columns:
        print("  ⚠️  realized_vol_theo not in data - run with updated code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Filter out NaN values
    df_vol = df[df['realized_vol_theo'].notna() & df['theo'].notna()].copy()
    if len(df_vol) == 0:
        print("  ⚠️  No realized vol theo data available (cold start period)")
        return

    if 'edge_vs_theo' not in df.columns:
        print("  ⚠️  edge_vs_theo not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    print(f"Fills with vol data: {len(df_vol)} / {len(df)}")

    # Calculate edge vs realized vol theo
    # For buys (dir_yes=+1): positive edge = bought below realized vol theo
    # For sells (dir_yes=-1): positive edge = sold above realized vol theo
    df_vol['edge_vs_realized_theo'] = np.where(
        df_vol['dir_yes'] == 1,
        df_vol['realized_vol_theo'] - df_vol['fill_yes'],
        df_vol['fill_yes'] - df_vol['realized_vol_theo']
    )

    # Compare correlations: which theo predicts markouts better?
    print(f"\nWhich theo predicts markouts better?")

    implied_corr = df_vol['edge_vs_theo'].corr(df_vol['markout_5s_per_share'])
    realized_corr = df_vol['edge_vs_realized_theo'].corr(df_vol['markout_5s_per_share'])

    print(f"  Edge vs Implied Theo → Markout correlation:  {implied_corr:.4f}")
    print(f"  Edge vs Realized Theo → Markout correlation: {realized_corr:.4f}")

    if realized_corr > implied_corr + 0.05:
        print(f"  ✅ REALIZED VOL THEO is MORE predictive!")
        print(f"     → Consider using realized vol for pricing")
    elif implied_corr > realized_corr + 0.05:
        print(f"  ✅ IMPLIED VOL THEO is MORE predictive")
        print(f"     → Current approach is correct")
    else:
        print(f"  ⚠️  Similar predictive power - no clear winner")

    # Compare theos directly
    print(f"\nTheo comparison:")
    theo_diff = (df_vol['realized_vol_theo'] - df_vol['theo']).mean()
    print(f"  Avg (realized_vol_theo - theo): {theo_diff:.4f}")
    print(f"  Realized vol theo tends to be {'HIGHER' if theo_diff > 0 else 'LOWER'} than implied vol theo")

    # Vol edge analysis: when realized > implied, did we make money?
    print(f"\nVol edge analysis:")
    if 'vol_edge_15m' in df_vol.columns:
        df_edge = df_vol[df_vol['vol_edge_15m'].notna()]

        high_realized = df_edge[df_edge['vol_edge_15m'] > 0.05]  # realized > implied by 5%
        low_realized = df_edge[df_edge['vol_edge_15m'] < -0.05]  # implied > realized by 5%

        print(f"  High realized vol (vol_edge > 5%): {len(high_realized)} fills, avg markout=${high_realized['markout_5s_per_share'].mean():.4f}")
        print(f"  Low realized vol (vol_edge < -5%): {len(low_realized)} fills, avg markout=${low_realized['markout_5s_per_share'].mean():.4f}")

        if len(high_realized) >= MIN_SAMPLES_FOR_SIGNIFICANCE and len(low_realized) >= MIN_SAMPLES_FOR_SIGNIFICANCE:
            comparison = compare_groups(high_realized['markout_5s_per_share'], low_realized['markout_5s_per_share'],
                                       "High Real Vol", "Low Real Vol")
            print(f"  {comparison}")

    # Test: when vol_edge is positive and we buy below strike, do we make money?
    print(f"\nDirectional vol edge test:")
    if 'vol_edge_15m' in df_vol.columns:
        # Note: We don't have moneyness directly, but we can infer from theo
        # If theo > 0.5, spot is likely above strike
        # If theo < 0.5, spot is likely below strike

        df_edge = df_vol[df_vol['vol_edge_15m'].notna()]

        # When realized > implied and we bought: do we make money?
        bought_high_vol = df_edge[(df_edge['vol_edge_15m'] > 0.03) & (df_edge['dir_yes'] == 1)]
        sold_high_vol = df_edge[(df_edge['vol_edge_15m'] > 0.03) & (df_edge['dir_yes'] == -1)]
        bought_low_vol = df_edge[(df_edge['vol_edge_15m'] < -0.03) & (df_edge['dir_yes'] == 1)]
        sold_low_vol = df_edge[(df_edge['vol_edge_15m'] < -0.03) & (df_edge['dir_yes'] == -1)]

        print(f"  Bought when realized > implied: {len(bought_high_vol)} fills, avg markout=${bought_high_vol['markout_5s_per_share'].mean():.4f}")
        print(f"  Sold when realized > implied:   {len(sold_high_vol)} fills, avg markout=${sold_high_vol['markout_5s_per_share'].mean():.4f}")
        print(f"  Bought when implied > realized: {len(bought_low_vol)} fills, avg markout=${bought_low_vol['markout_5s_per_share'].mean():.4f}")
        print(f"  Sold when implied > realized:   {len(sold_low_vol)} fills, avg markout=${sold_low_vol['markout_5s_per_share'].mean():.4f}")

    # Performance by edge vs realized vol theo buckets
    print(f"\nMarkout by edge vs realized vol theo:")
    df_vol['realized_edge_bucket'] = pd.cut(
        df_vol['edge_vs_realized_theo'],
        bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
        labels=['Large -ve', 'Small -ve', 'Neutral', 'Small +ve', 'Large +ve']
    )

    for bucket in df_vol['realized_edge_bucket'].cat.categories:
        subset = df_vol[df_vol['realized_edge_bucket'] == bucket]
        if len(subset) > 0:
            print(f"  {bucket:12s}: {len(subset):3d} fills, avg markout=${subset['markout_5s_per_share'].mean():.4f}")

    print(f"\n  Expected: Large +ve edge vs realized theo → positive markouts")
    print(f"  If this pattern holds, realized vol theo has predictive power!")


def momentum_attribution_analysis(df):
    """Test if momentum strategy is responsible for bad fills."""
    print("\n" + "=" * 80)
    print("MOMENTUM ATTRIBUTION: IS MOMENTUM CAUSING ADVERSE SELECTION?")
    print("=" * 80)

    # Split fills by edge_vs_fair
    bad_fills = df[df['edge_vs_fair'] < -0.005]    # Paid up (got picked off)
    good_fills = df[df['edge_vs_fair'] > 0.005]    # Got edge (passive MM)
    neutral = df[abs(df['edge_vs_fair']) <= 0.005] # At fair

    print(f"Fill categories:")
    print(f"  Bad fills (paid up):  {len(bad_fills)} fills")
    print(f"  Good fills (got edge): {len(good_fills)} fills")
    print(f"  Neutral fills:         {len(neutral)} fills")

    # Compare momentum during different fill types
    print(f"\nAverage |momentum| by fill type:")
    if len(bad_fills) > 0:
        bad_mom = bad_fills['momentum'].abs().mean()
        print(f"  Bad fills:  ${bad_mom:.2f}")
    if len(good_fills) > 0:
        good_mom = good_fills['momentum'].abs().mean()
        print(f"  Good fills: ${good_mom:.2f}")
    if len(neutral) > 0:
        neutral_mom = neutral['momentum'].abs().mean()
        print(f"  Neutral:    ${neutral_mom:.2f}")

    if len(bad_fills) > 0 and len(good_fills) > 0:
        if bad_mom > good_mom * 1.5:
            print(f"\n  ❌ BAD FILLS happen during HIGHER momentum ({bad_mom:.2f} vs {good_mom:.2f})")
            print(f"     → Momentum strategy IS causing adverse selection!")
            print(f"     → Recommendation: Disable USE_BINANCE_MOMENTUM")
        elif bad_mom > good_mom:
            print(f"\n  ⚠️  BAD FILLS have slightly higher momentum ({bad_mom:.2f} vs {good_mom:.2f})")
            print(f"     → Momentum might be contributing to adverse selection")
        else:
            print(f"\n  ✅ BAD FILLS have SIMILAR/LOWER momentum ({bad_mom:.2f} vs {good_mom:.2f})")
            print(f"     → Momentum strategy is NOT the main problem")
            print(f"     → Problem is likely: VPN lag, touch joining, or general market dynamics")

    # Check momentum volatility
    if 'momentum_volatility' in df.columns:
        print(f"\nAverage momentum_volatility by fill type:")
        if len(bad_fills) > 0:
            bad_vol = bad_fills['momentum_volatility'].mean()
            print(f"  Bad fills:  ${bad_vol:.2f}")
        if len(good_fills) > 0:
            good_vol = good_fills['momentum_volatility'].mean()
            print(f"  Good fills: ${good_vol:.2f}")
        if len(neutral) > 0:
            neutral_vol = neutral['momentum_volatility'].mean()
            print(f"  Neutral:    ${neutral_vol:.2f}")

        if len(bad_fills) > 0 and len(good_fills) > 0:
            if bad_vol > good_vol * 1.3:
                print(f"\n  ❌ BAD FILLS happen during MORE VOLATILE periods")
                print(f"     → Dynamic spread needs to be MORE aggressive")

    # Show worst fills
    if len(bad_fills) >= 5:
        print(f"\nWorst 5 fills (most adverse selection):")
        worst = bad_fills.nsmallest(5, 'markout_5s_per_share')[['timestamp', 'edge_vs_fair', 'markout_5s_per_share', 'momentum', 'momentum_volatility']]
        print(worst.to_string(index=False))

def signal_interaction_analysis(df):
    """Analyze how z-score skew and book imbalance interact."""
    print("\n" + "=" * 80)
    print("SIGNAL INTERACTION ANALYSIS (Z-Skew vs Book Imbalance)")
    print("=" * 80)

    if 'z_skew' not in df.columns or 'book_imbalance' not in df.columns:
        print("  ⚠️  Missing z_skew or book_imbalance data")
        return

    df_signals = df[(df['z_skew'].notna()) & (df['book_imbalance'].notna())].copy()
    if len(df_signals) == 0:
        print("  ⚠️  No fills with both signals")
        return

    # Classify signals as positive/negative/neutral
    df_signals['z_direction'] = 'neutral'
    df_signals.loc[df_signals['z_skew'] > 0.005, 'z_direction'] = 'positive'  # >0.5¢
    df_signals.loc[df_signals['z_skew'] < -0.005, 'z_direction'] = 'negative'  # <-0.5¢

    df_signals['imb_direction'] = 'neutral'
    df_signals.loc[df_signals['book_imbalance'] > 0.2, 'imb_direction'] = 'positive'
    df_signals.loc[df_signals['book_imbalance'] < -0.2, 'imb_direction'] = 'negative'

    # Agreement scenarios
    both_positive = df_signals[(df_signals['z_direction'] == 'positive') & (df_signals['imb_direction'] == 'positive')]
    both_negative = df_signals[(df_signals['z_direction'] == 'negative') & (df_signals['imb_direction'] == 'negative')]
    z_pos_imb_neg = df_signals[(df_signals['z_direction'] == 'positive') & (df_signals['imb_direction'] == 'negative')]
    z_neg_imb_pos = df_signals[(df_signals['z_direction'] == 'negative') & (df_signals['imb_direction'] == 'positive')]
    both_neutral = df_signals[(df_signals['z_direction'] == 'neutral') & (df_signals['imb_direction'] == 'neutral')]

    print(f"\nSignal Agreement:")
    print(f"  Both bullish (z>0, imb>0):   {len(both_positive):4d} fills, {both_positive['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")
    print(f"  Both bearish (z<0, imb<0):   {len(both_negative):4d} fills, {both_negative['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")

    print(f"\nSignal Conflict:")
    print(f"  Z bullish, Imb bearish:      {len(z_pos_imb_neg):4d} fills, {z_pos_imb_neg['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")
    print(f"  Z bearish, Imb bullish:      {len(z_neg_imb_pos):4d} fills, {z_neg_imb_pos['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")

    print(f"\nBoth Neutral:")
    print(f"  No strong signals:           {len(both_neutral):4d} fills, {both_neutral['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")

    # Statistical tests
    if len(both_positive) > 10 and len(both_neutral) > 10:
        comparison = compare_groups(both_positive['markout_5s_per_share'], both_neutral['markout_5s_per_share'],
                                   "Both Agree", "Both Neutral")
        print(f"\n  Agreement vs Neutral: {comparison}")

        if both_positive['markout_5s_per_share'].mean() > both_neutral['markout_5s_per_share'].mean() + 0.01:
            print(f"  ✅ Signal AGREEMENT has strong edge - when both align, fills are better!")

    # When signals conflict, which wins?
    conflict = pd.concat([z_pos_imb_neg, z_neg_imb_pos])
    if len(conflict) > 20:
        print(f"\nWhen Signals Conflict:")
        print(f"  Conflict fills: {len(conflict)} ({len(conflict)/len(df_signals)*100:.1f}% of all fills)")
        print(f"  Avg markout: {conflict['markout_5s_per_share'].mean()*100:+.2f}¢/share")

        if conflict['markout_5s_per_share'].mean() < both_positive['markout_5s_per_share'].mean() - 0.01:
            print(f"  ⚠️  Conflicting signals have WORSE markouts - wait for alignment!")

    # Signal strength analysis
    print(f"\n{'='*80}")
    print(f"SIGNAL STRENGTH COMBINATIONS")
    print(f"{'='*80}")

    strong_z = df_signals[abs(df_signals['z_skew']) > 0.015]  # >1.5¢
    strong_imb = df_signals[abs(df_signals['book_imbalance']) > 0.4]

    both_strong = df_signals[(abs(df_signals['z_skew']) > 0.015) & (abs(df_signals['book_imbalance']) > 0.4)]
    z_strong_imb_weak = df_signals[(abs(df_signals['z_skew']) > 0.015) & (abs(df_signals['book_imbalance']) <= 0.2)]
    z_weak_imb_strong = df_signals[(abs(df_signals['z_skew']) <= 0.005) & (abs(df_signals['book_imbalance']) > 0.4)]
    both_weak = df_signals[(abs(df_signals['z_skew']) <= 0.005) & (abs(df_signals['book_imbalance']) <= 0.2)]

    print(f"  Both strong:        {len(both_strong):4d} fills, {both_strong['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")
    print(f"  Z strong, Imb weak: {len(z_strong_imb_weak):4d} fills, {z_strong_imb_weak['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")
    print(f"  Z weak, Imb strong: {len(z_weak_imb_strong):4d} fills, {z_weak_imb_strong['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")
    print(f"  Both weak:          {len(both_weak):4d} fills, {both_weak['markout_5s_per_share'].mean()*100:+.2f}¢/share avg")

    if len(both_strong) > 10:
        if both_strong['markout_5s_per_share'].mean() > both_weak['markout_5s_per_share'].mean() + 0.01:
            print(f"\n  ✅ STRONG signals have better edge - wait for conviction!")


def fill_size_distribution_analysis(df):
    """Analyze if larger fills have worse edge (adverse selection on size)."""
    print("\n" + "=" * 80)
    print("FILL SIZE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Size buckets
    tiny = df[df['qty'] <= 5]
    small = df[(df['qty'] > 5) & (df['qty'] <= 10)]
    medium = df[(df['qty'] > 10) & (df['qty'] <= 20)]
    large = df[(df['qty'] > 20) & (df['qty'] <= 40)]
    xlarge = df[df['qty'] > 40]

    print(f"\nFill Size Distribution:")
    print(f"  ≤5 shares:    {len(tiny):4d} fills ({len(tiny)/len(df)*100:4.1f}%), {tiny['qty'].sum():6.0f} shares, {tiny['markout_5s_per_share'].mean()*100:+.2f}¢/share, ${tiny['markout_5s'].sum():+7.2f} PNL")
    print(f"  6-10 shares:  {len(small):4d} fills ({len(small)/len(df)*100:4.1f}%), {small['qty'].sum():6.0f} shares, {small['markout_5s_per_share'].mean()*100:+.2f}¢/share, ${small['markout_5s'].sum():+7.2f} PNL")
    print(f"  11-20 shares: {len(medium):4d} fills ({len(medium)/len(df)*100:4.1f}%), {medium['qty'].sum():6.0f} shares, {medium['markout_5s_per_share'].mean()*100:+.2f}¢/share, ${medium['markout_5s'].sum():+7.2f} PNL")
    print(f"  21-40 shares: {len(large):4d} fills ({len(large)/len(df)*100:4.1f}%), {large['qty'].sum():6.0f} shares, {large['markout_5s_per_share'].mean()*100:+.2f}¢/share, ${large['markout_5s'].sum():+7.2f} PNL")
    print(f"  >40 shares:   {len(xlarge):4d} fills ({len(xlarge)/len(df)*100:4.1f}%), {xlarge['qty'].sum():6.0f} shares, {xlarge['markout_5s_per_share'].mean()*100:+.2f}¢/share, ${xlarge['markout_5s'].sum():+7.2f} PNL")

    if len(tiny) > 0 and len(xlarge) > 0:
        comparison = compare_groups(tiny['markout_5s_per_share'], xlarge['markout_5s_per_share'], "Tiny", "XLarge")
        print(f"\n  Tiny vs XLarge: {comparison}")

        edge_diff = tiny['markout_5s_per_share'].mean() - xlarge['markout_5s_per_share'].mean()
        if edge_diff > 0.005:  # >0.5¢ difference
            print(f"\n  ❌ LARGE FILLS WORSE by {edge_diff*100:.2f}¢/share - adverse selection on size!")
            print(f"     → Reduce BASE_SIZE to improve edge")
        else:
            print(f"\n  ✅ No adverse selection on size - order size is fine")

    # What would PNL be if only small fills?
    small_only = df[df['qty'] <= 10]
    if len(small_only) > 0:
        small_pnl = small_only['markout_5s'].sum()
        total_pnl = df['markout_5s'].sum()
        small_shares = small_only['qty'].sum()
        total_shares = df['qty'].sum()

        print(f"\n  Hypothetical: Only fills ≤10 shares")
        print(f"    PNL: ${small_pnl:+.2f} (actual: ${total_pnl:+.2f})")
        print(f"    Shares: {small_shares:.0f} (actual: {total_shares:.0f})")
        print(f"    Edge: {small_pnl/small_shares*100:+.2f}¢/share (actual: {total_pnl/total_shares*100:+.2f}¢/share)")

        if small_pnl/total_pnl > 0.8 and small_shares/total_shares < 0.6:
            print(f"    ⚠️  80%+ of PNL from <60% of volume - strong signal to reduce size!")


def spread_capture_analysis(df):
    """Analyze spread capture rate and edge distribution."""
    print("\n" + "=" * 80)
    print("SPREAD CAPTURE ANALYSIS")
    print("=" * 80)

    # Need to know spread - check working_params or estimate
    BASE_SPREAD = 0.055  # Update this if different
    half_spread = BASE_SPREAD / 2.0

    total_shares = df['qty'].sum()
    total_pnl = df['markout_5s'].sum()
    avg_edge_per_share = total_pnl / total_shares

    capture_rate = (avg_edge_per_share / half_spread) * 100

    print(f"\nSpread Capture Metrics:")
    print(f"  Assumed spread: {BASE_SPREAD*100:.1f}¢ total, {half_spread*100:.2f}¢ half-spread")
    print(f"  Actual edge: {avg_edge_per_share*100:+.2f}¢/share")
    print(f"  Capture rate: {capture_rate:.1f}% of half-spread")

    if capture_rate < 30:
        print(f"  ❌ LOW capture (<30%) - too much adverse selection or spread too wide")
    elif capture_rate < 45:
        print(f"  ⚠️  MODERATE capture (30-45%) - room for improvement")
    else:
        print(f"  ✅ GOOD capture (>45%) - edge is strong")

    # Edge distribution
    print(f"\nEdge Distribution (per-share):")
    bins = [-np.inf, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, np.inf]
    labels = ['<-2¢', '-2¢--1¢', '-1¢--0.5¢', '-0.5¢-0', '0-0.5¢', '0.5¢-1¢', '1¢-2¢', '>2¢']

    df['edge_bucket'] = pd.cut(df['markout_5s_per_share'], bins=bins, labels=labels)

    for bucket in labels:
        subset = df[df['edge_bucket'] == bucket]
        if len(subset) > 0:
            pct = len(subset) / len(df) * 100
            pnl = subset['markout_5s'].sum()
            print(f"  {bucket:12s}: {len(subset):4d} fills ({pct:4.1f}%), ${pnl:+7.2f} PNL")


def maker_vs_taker_analysis(df):
    """Compare performance of maker (GTC) vs taker (IOC) fills."""
    print("\n" + "=" * 80)
    print("MAKER vs TAKER ANALYSIS")
    print("=" * 80)

    if 'order_type' not in df.columns:
        print("  ⚠️  order_type not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    gtc = df[df['order_type'] == 'GTC']
    ioc = df[df['order_type'] == 'IOC']

    gtc_shares = gtc['qty'].sum() if len(gtc) > 0 else 0
    ioc_shares = ioc['qty'].sum() if len(ioc) > 0 else 0

    print(f"Fill counts:")
    print(f"  Maker (GTC): {len(gtc)} fills, {gtc_shares:.0f} shares ({len(gtc)/len(df)*100:.1f}%)")
    print(f"  Taker (IOC): {len(ioc)} fills, {ioc_shares:.0f} shares ({len(ioc)/len(df)*100:.1f}%)")

    print(f"\nPer-Share Markout Performance:")
    for horizon in [1, 5, 15, 30, 60]:
        per_share_col = f'markout_{horizon}s_per_share'
        if per_share_col in df.columns:
            gtc_avg = gtc[per_share_col].mean() if len(gtc) > 0 else 0
            ioc_avg = ioc[per_share_col].mean() if len(ioc) > 0 else 0
            comparison = compare_groups(gtc[per_share_col], ioc[per_share_col], "Maker", "Taker") if len(gtc) > 0 and len(ioc) > 0 else ""
            print(f"  {horizon}s: Maker={gtc_avg*100:+.2f}¢/share, Taker={ioc_avg*100:+.2f}¢/share")
            if comparison:
                print(f"       {comparison}")

    print(f"\nTotal PNL:")
    for horizon in [1, 5, 15, 30, 60]:
        col = f'markout_{horizon}s'
        if col in df.columns:
            gtc_total = gtc[col].sum() if len(gtc) > 0 else 0
            ioc_total = ioc[col].sum() if len(ioc) > 0 else 0
            print(f"  {horizon}s: Maker=${gtc_total:+.2f}, Taker=${ioc_total:+.2f}, Combined=${gtc_total+ioc_total:+.2f}")

    if len(gtc) > 0 and len(ioc) > 0:
        gtc_hit = (gtc['markout_5s_per_share'] > 0).sum() / len(gtc) * 100
        ioc_hit = (ioc['markout_5s_per_share'] > 0).sum() / len(ioc) * 100

        print(f"\nHit rate (5s markout > 0):")
        print(f"  Maker (GTC): {gtc_hit:.1f}%")
        print(f"  Taker (IOC): {ioc_hit:.1f}%")

        # Edge analysis
        if 'edge_vs_theo' in df.columns:
            print(f"\nEdge vs theo:")
            print(f"  Maker avg edge: ${gtc['edge_vs_theo'].mean():.4f}")
            print(f"  Taker avg edge: ${ioc['edge_vs_theo'].mean():.4f}")

        # Momentum during fills
        if 'momentum' in df.columns:
            print(f"\nMomentum during fills:")
            print(f"  Maker avg |momentum|: ${gtc['momentum'].abs().mean():.2f}")
            print(f"  Taker avg |momentum|: ${ioc['momentum'].abs().mean():.2f}")

        # Diagnosis
        print(f"\nDiagnosis:")
        if gtc_total > 0 and ioc_total > 0:
            print(f"  ✅ Both maker and taker are profitable!")
        elif gtc_total > 0:
            print(f"  ✅ Maker is profitable, ❌ Taker is losing")
            print(f"     → Consider increasing EDGE_TAKE_THRESHOLD or disabling taker")
        elif ioc_total > 0:
            print(f"  ❌ Maker is losing, ✅ Taker is profitable")
            print(f"     → Passive quotes are getting picked off")
            print(f"     → Consider widening spread or faster cancels")
        else:
            print(f"  ❌ Both maker and taker are losing!")

def aggressive_mode_analysis(df):
    """Analyze performance of aggressive mode fills vs normal fills."""
    print("\n" + "=" * 80)
    print("AGGRESSIVE MODE ANALYSIS")
    print("=" * 80)

    if 'aggressive_mode' not in df.columns:
        print("  ⚠️  aggressive_mode not in data - run with updated trading code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Split by aggressive mode
    aggressive = df[df['aggressive_mode'] == True]
    normal = df[df['aggressive_mode'] == False]

    print(f"\n  Fill counts:")
    print(f"    Normal mode:     {len(normal):>6} fills")
    print(f"    Aggressive mode: {len(aggressive):>6} fills ({len(aggressive)/len(df)*100:.1f}%)")

    if len(aggressive) < 5:
        print(f"\n  ⚠️  Too few aggressive fills ({len(aggressive)}) for meaningful analysis")
        print("     Need more data - aggressive mode triggers when:")
        print("       - |z_score| > AGGRESSIVE_Z_THRESHOLD")
        print("       - |z_skew_raw| > AGGRESSIVE_ZSKEW_THRESHOLD")
        print("       - z_score and book_imbalance have same sign (aligned)")
        return

    print(f"\n  Per-share markout comparison:")
    print(f"  {'Horizon':<10} {'Normal':<20} {'Aggressive':<20} {'Difference'}")
    print(f"  {'-'*10} {'-'*20} {'-'*20} {'-'*20}")

    for horizon in [1, 5, 15, 30, 60]:
        per_share_col = f'markout_{horizon}s_per_share'
        if per_share_col not in df.columns:
            continue

        normal_mean = normal[per_share_col].mean() if len(normal) > 0 else 0
        aggressive_mean = aggressive[per_share_col].mean() if len(aggressive) > 0 else 0
        diff = aggressive_mean - normal_mean

        # Significance test comparing the two groups
        comparison = compare_groups(
            aggressive[per_share_col],
            normal[per_share_col],
            name1="Aggressive",
            name2="Normal"
        )

        # Extract stars from comparison
        stars = ""
        if "***" in comparison:
            stars = "***"
        elif "**" in comparison:
            stars = "**"
        elif "*" in comparison:
            stars = "*"

        print(f"  {horizon}s:       {normal_mean:>+.5f}           {aggressive_mean:>+.5f}           {diff:>+.5f} {stars}")

    # Hit rate comparison
    print(f"\n  Hit rate comparison (% of fills with positive markout):")
    for horizon in [5, 15]:
        per_share_col = f'markout_{horizon}s_per_share'
        if per_share_col not in df.columns:
            continue

        normal_hit = (normal[per_share_col] > 0).sum() / len(normal) * 100 if len(normal) > 0 else 0
        aggressive_hit = (aggressive[per_share_col] > 0).sum() / len(aggressive) * 100 if len(aggressive) > 0 else 0

        print(f"    {horizon}s: Normal {normal_hit:.1f}%, Aggressive {aggressive_hit:.1f}%")

    # Total PNL contribution
    print(f"\n  Total PNL contribution:")
    for horizon in [5, 15]:
        col = f'markout_{horizon}s'
        if col not in df.columns:
            continue

        normal_pnl = normal[col].sum() if len(normal) > 0 else 0
        aggressive_pnl = aggressive[col].sum() if len(aggressive) > 0 else 0
        total_pnl = normal_pnl + aggressive_pnl

        normal_pct = normal_pnl / total_pnl * 100 if total_pnl != 0 else 0
        aggressive_pct = aggressive_pnl / total_pnl * 100 if total_pnl != 0 else 0

        print(f"    {horizon}s: Normal ${normal_pnl:+.2f} ({normal_pct:.0f}%), Aggressive ${aggressive_pnl:+.2f} ({aggressive_pct:.0f}%)")

    # Statistical test for 5s markout
    print(f"\n  Statistical comparison (5s markout):")
    if 'markout_5s_per_share' in df.columns:
        comparison = compare_groups(
            aggressive['markout_5s_per_share'],
            normal['markout_5s_per_share'],
            name1="Aggressive",
            name2="Normal"
        )
        print(f"    {comparison}")

    # Verdict
    print(f"\n  Verdict:")
    if len(aggressive) >= MIN_SAMPLES_FOR_SIGNIFICANCE and 'markout_5s_per_share' in df.columns:
        aggressive_mean = aggressive['markout_5s_per_share'].mean()
        normal_mean = normal['markout_5s_per_share'].mean()

        if aggressive_mean > normal_mean + 0.002:
            print(f"    ✅ Aggressive mode outperforms normal by {(aggressive_mean-normal_mean)*100:.2f}¢/share")
            print(f"       → Consider making aggressive mode trigger more often")
        elif aggressive_mean > normal_mean:
            print(f"    ✅ Aggressive mode slightly better (+{(aggressive_mean-normal_mean)*100:.2f}¢/share)")
        elif aggressive_mean > -0.001:
            print(f"    ⚠️  Aggressive mode similar to normal ({(aggressive_mean-normal_mean)*100:.2f}¢/share)")
        else:
            print(f"    ❌ Aggressive mode underperforms by {(normal_mean-aggressive_mean)*100:.2f}¢/share")
            print(f"       → Consider tightening aggressive mode thresholds or disabling")
    else:
        print(f"    ⚠️  Need more aggressive mode fills for reliable verdict")


def summary_and_diagnosis(df):
    """Overall diagnosis and recommendations."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)

    issues = []

    # Check overall profitability (per-share and total)
    avg_markout_per_share = df['markout_5s_per_share'].mean()
    total_pnl = df['markout_5s'].sum()

    if avg_markout_per_share < -0.001:
        issues.append(f"❌ LOSING MONEY: {avg_markout_per_share*100:.2f}¢/share avg, ${total_pnl:.2f} total PNL")
    elif avg_markout_per_share < 0.001:
        issues.append(f"⚠️  BREAKEVEN: {avg_markout_per_share*100:.2f}¢/share avg, ${total_pnl:.2f} total PNL")
    else:
        issues.append(f"✅ PROFITABLE: +{avg_markout_per_share*100:.2f}¢/share avg, ${total_pnl:+.2f} total PNL")

    # Check delta
    if 'delta' in df.columns and df['delta'].mean() < 0.00001:
        issues.append("❌ CRITICAL: Delta near zero - momentum not working!")

    # Check momentum correlation
    if 'momentum' in df.columns:
        mom_corr = df['momentum'].corr(df['markout_5s_per_share'])
        if mom_corr < 0:
            issues.append("❌ Negative momentum correlation - strategy backwards?")
        elif mom_corr < 0.05:
            issues.append("⚠️  Weak momentum correlation - edge unclear")

    # Check theo value
    if 'edge_vs_theo' in df.columns:
        theo_corr = df['edge_vs_theo'].corr(df['markout_5s_per_share'])
        if theo_corr < 0:
            issues.append("❌ Theo has negative correlation - model is wrong")
        elif theo_corr > 0.15:
            issues.append("✅ Theo has value - model is predictive")

    # Check adverse selection
    hit_rate = (df['markout_5s_per_share'] > 0).sum() / len(df) * 100
    if hit_rate < 45:
        issues.append("❌ Low hit rate - getting adverse selected")

    print("\nKey findings:")
    for issue in issues:
        print(f"  {issue}")

    print(f"\nRecommendations:")
    if avg_markout_per_share < 0:
        print("  1. STOP TRADING - You're losing money!")
        print("  2. Check if delta is working (should be 0.0001-0.01)")
        print("  3. Review momentum vs theo correlations")
        if 'momentum' in df.columns and df['momentum'].corr(df['markout_5s_per_share']) < 0.05:
            print("  4. Consider disabling momentum (USE_BINANCE_MOMENTUM=False)")
        if df['delta'].mean() < 0.00001:
            print("  5. CRITICAL: Delta is broken - check why binary_delta is zero")

def main():
    """Run full analysis."""
    df = load_data()
    if df is None:
        return

    if len(df) == 0:
        print("ERROR: No fills found in CSV!")
        return

    # Run all analyses
    overall_performance(df)
    spread_capture_analysis(df)  # NEW: Spread capture rate and edge distribution
    fill_size_distribution_analysis(df)  # NEW: Analyze adverse selection by fill size
    maker_vs_taker_analysis(df)  # NEW: Compare GTC vs IOC fills
    aggressive_mode_analysis(df)  # NEW: Compare aggressive vs normal mode fills
    signal_interaction_analysis(df)  # NEW: Z-skew vs book imbalance interactions
    momentum_analysis(df)
    zscore_predictor_analysis(df)  # NEW: Analyze z-score predictor
    z_skew_analysis(df)  # NEW: Analyze z-score skew (continuous fair value adjustment)
    z_skew_residual_analysis(df)  # NEW: Analyze residual approach (Option 3 validation)
    sigmoid_zscore_analysis(df)  # NEW: Analyze sigmoid confidence scaling effectiveness
    theo_value_test(df)
    model_vs_market_test(df)
    adverse_selection_test(df)
    inventory_analysis(df)
    directional_bias(df)
    volatility_spread_analysis(df)
    book_imbalance_analysis(df)  # NEW: Analyze book imbalance signal
    realized_vol_theo_analysis(df)  # NEW: Compare realized vs implied vol theo
    momentum_attribution_analysis(df)
    summary_and_diagnosis(df)

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE GUIDE")
    print("=" * 80)
    print("""
  *** = p < 0.01 (99% confident result is real, not noise)
  **  = p < 0.05 (95% confident result is real)
  *   = p < 0.10 (90% confident - marginally significant)
  (no stars) = NOT statistically significant - could be random noise!

  95% CI = Confidence interval. True value likely falls within this range.

  Sample size matters:
    n < 30:  Results are unreliable, need more data
    n = 30-100: Moderate confidence
    n > 100: Good statistical power

  If your key metrics are NOT significant (no stars):
    → Collect more data before making decisions
    → The observed effect might just be noise
    """)
    print("=" * 80)

    # Market count summary
    print("\n" + "=" * 80)
    print("TRADING ACTIVITY SUMMARY")
    print("=" * 80)

    # Convert timestamp to datetime if it's a string
    if 'timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp'])

        # Calculate time span
        start_time = df['dt'].min()
        end_time = df['dt'].max()
        duration = end_time - start_time

        print(f"\nTime period:")
        print(f"  Start: {start_time}")
        print(f"  End:   {end_time}")
        print(f"  Duration: {duration}")

        # Count unique 15-minute market periods
        # Round timestamps down to 15-minute intervals
        df['market_period'] = df['dt'].dt.floor('15min')
        unique_markets = df['market_period'].nunique()

        print(f"\nMarket activity:")
        print(f"  Total fills: {len(df)}")
        print(f"  Unique 15-min markets traded: {unique_markets}")
        print(f"  Average fills per market: {len(df)/unique_markets:.1f}")
    else:
        print(f"\nTotal fills: {len(df)}")
        print("  (No timestamp column - cannot calculate unique markets)")

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
