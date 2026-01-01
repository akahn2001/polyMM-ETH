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

    print(f"Total fills: {len(df)}")
    print(f"Buy fills: {(df['dir_yes'] == 1).sum()}")
    print(f"Sell fills: {(df['dir_yes'] == -1).sum()}")

    print(f"\nMarkout Performance (with significance tests):")
    print(f"  Legend: *** p<0.01, ** p<0.05, * p<0.10")
    print()
    for horizon in [1, 5, 15, 30, 60]:
        col = f'markout_{horizon}s'
        if col in df.columns:
            avg = df[col].mean()
            median = df[col].median()
            hit_rate = (df[col] > 0).sum() / len(df) * 100
            total = df[col].sum()
            _, p_val, is_sig, interp = significance_test(df[col], null_value=0)
            print(f"  {horizon}s: {interp}")
            print(f"       median=${median:.4f}, hit={hit_rate:.1f}%, total=${total:.2f}")

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

    mom_corr = df['aligned_momentum'].corr(df['markout_5s'])
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

    print(f"  Rising (>$10):   {len(rising):3d} fills, avg markout=${rising['markout_5s'].mean():.4f}")
    _, _, _, rising_sig = significance_test(rising['markout_5s']) if len(rising) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if rising_sig: print(f"                   {rising_sig}")

    print(f"  Falling (<-$10): {len(falling):3d} fills, avg markout=${falling['markout_5s'].mean():.4f}")
    _, _, _, falling_sig = significance_test(falling['markout_5s']) if len(falling) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if falling_sig: print(f"                   {falling_sig}")

    print(f"  Flat (±$10):     {len(flat):3d} fills, avg markout=${flat['markout_5s'].mean():.4f}")
    _, _, _, flat_sig = significance_test(flat['markout_5s']) if len(flat) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if flat_sig: print(f"                   {flat_sig}")

    # Should momentum fills have better markouts?
    momentum_fills = df[abs(df['momentum']) > 10]
    flat_fills = df[abs(df['momentum']) <= 10]

    print(f"\n  Momentum fills (|mom|>$10): avg markout=${momentum_fills['markout_5s'].mean():.4f}")
    print(f"  Flat fills (|mom|≤$10):     avg markout=${flat_fills['markout_5s'].mean():.4f}")

    # Statistical comparison
    comparison = compare_groups(momentum_fills['markout_5s'], flat_fills['markout_5s'], "Momentum", "Flat")
    print(f"  Comparison: {comparison}")

    if momentum_fills['markout_5s'].mean() > flat_fills['markout_5s'].mean():
        print("  ✅ Momentum fills are more profitable!")
    else:
        print("  ❌ Momentum fills are WORSE - strategy may be broken")

def theo_value_test(df):
    """Test if theo has predictive value."""
    print("\n" + "=" * 80)
    print("THEO VALUE TEST")
    print("=" * 80)

    if 'edge_vs_theo' not in df.columns:
        print("  ⚠️  edge_vs_theo not in data - run with updated markouts code")
        print("     Delete detailed_fills.csv and restart bot to get this data")
        return

    # Edge vs theo correlation with markouts
    edge_corr = df['edge_vs_theo'].corr(df['markout_5s'])
    print(f"Edge vs Theo → Markout correlation: {edge_corr:.3f}")
    if edge_corr > 0.2:
        print("  ✅ Strong positive correlation - theo has value!")
    elif edge_corr > 0.05:
        print("  ⚠️  Weak positive correlation - theo has some value")
    else:
        print("  ❌ No correlation - theo is NOT predictive!")

    # Performance by edge buckets
    print(f"\nMarkout performance by edge vs theo:")
    df['edge_bucket'] = pd.cut(df['edge_vs_theo'], bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                 labels=['Large -ve', 'Small -ve', 'Neutral', 'Small +ve', 'Large +ve'])

    for bucket in df['edge_bucket'].cat.categories:
        subset = df[df['edge_bucket'] == bucket]
        if len(subset) > 0:
            print(f"  {bucket:12s}: {len(subset):3d} fills, avg markout=${subset['markout_5s'].mean():.4f}")

    print(f"\n  Expected: Large +ve edge → positive markouts")
    print(f"  Expected: Large -ve edge → negative markouts")

def model_vs_market_test(df):
    """Test if model disagreement with market is valuable."""
    print("\n" + "=" * 80)
    print("MODEL DISAGREEMENT TEST")
    print("=" * 80)

    # When model thinks market is wrong, and you trade accordingly
    model_thinks_cheap = df[(df['model_vs_market'] > 0.02) & (df['dir_yes'] == 1)]
    model_thinks_rich = df[(df['model_vs_market'] < -0.02) & (df['dir_yes'] == -1)]

    print(f"Bought when model said underpriced (model>market by >2¢):")
    print(f"  {len(model_thinks_cheap)} fills, avg markout=${model_thinks_cheap['markout_5s'].mean():.4f}")

    print(f"\nSold when model said overpriced (model<market by >2¢):")
    print(f"  {len(model_thinks_rich)} fills, avg markout=${model_thinks_rich['markout_5s'].mean():.4f}")

    if len(model_thinks_cheap) > 0 and model_thinks_cheap['markout_5s'].mean() > 0:
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
    print(f"  {len(paid_up)} fills, avg markout=${paid_up['markout_5s'].mean():.4f}")
    if paid_up['markout_5s'].mean() < 0:
        print("  ❌ Paid up fills have negative markouts - getting picked off!")

    print(f"\nFills where you got edge (edge_vs_fair > +0.5¢):")
    print(f"  {len(got_edge)} fills, avg markout=${got_edge['markout_5s'].mean():.4f}")

    print(f"\nFills at fair value (|edge_vs_fair| ≤ 0.5¢):")
    print(f"  {len(neutral)} fills, avg markout=${neutral['markout_5s'].mean():.4f}")

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

    print(f"\nBuilding position: {len(building)} fills, avg markout=${building['markout_5s'].mean():.4f}")
    print(f"Reducing position: {len(reducing)} fills, avg markout=${reducing['markout_5s'].mean():.4f}")

    if building['markout_5s'].mean() < reducing['markout_5s'].mean():
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

    print(f"Buy performance:  {len(buys)} fills, avg markout=${buys['markout_5s'].mean():.4f}")
    _, _, _, buy_sig = significance_test(buys['markout_5s']) if len(buys) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if buy_sig: print(f"                  {buy_sig}")

    print(f"Sell performance: {len(sells)} fills, avg markout=${sells['markout_5s'].mean():.4f}")
    _, _, _, sell_sig = significance_test(sells['markout_5s']) if len(sells) >= MIN_SAMPLES_FOR_SIGNIFICANCE else (None, None, False, "")
    if sell_sig: print(f"                  {sell_sig}")

    # Statistical comparison
    comparison = compare_groups(buys['markout_5s'], sells['markout_5s'], "Buys", "Sells")
    print(f"\nBuy vs Sell comparison: {comparison}")

    if abs(buys['markout_5s'].mean() - sells['markout_5s'].mean()) > 0.005:
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
    print(f"  Low volatility (≤$10):  {len(low_vol):3d} fills, avg markout=${low_vol['markout_5s'].mean():.4f}")
    print(f"  Med volatility ($10-20): {len(med_vol):3d} fills, avg markout=${med_vol['markout_5s'].mean():.4f}")
    print(f"  High volatility (>$20):  {len(high_vol):3d} fills, avg markout=${high_vol['markout_5s'].mean():.4f}")

    # Check if high volatility fills have worse markouts (justifying wider spreads)
    if len(high_vol) > 0 and len(low_vol) > 0:
        if high_vol['markout_5s'].mean() < low_vol['markout_5s'].mean():
            print(f"  ❌ High volatility fills are WORSE - dynamic spread helps!")
            print(f"     Widening spread during volatile times should improve performance")
        else:
            print(f"  ✅ High volatility fills are OK - dynamic spread may not be needed")

    # Check edge_vs_fair in high volatility (are you getting picked off?)
    if len(high_vol) > 0:
        paid_up_in_vol = high_vol[high_vol['edge_vs_fair'] < -0.005]
        print(f"\n  During high volatility:")
        print(f"    Paid up (edge<-0.5¢): {len(paid_up_in_vol)} fills, avg markout=${paid_up_in_vol['markout_5s'].mean():.4f}")
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

    imb_corr = df_imb['aligned_imbalance'].corr(df_imb['markout_5s'])
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

    print(f"  Strong bid imbalance (>0.3):  {len(strong_bid):3d} fills, avg markout=${strong_bid['markout_5s'].mean():.4f}")
    print(f"  Neutral (±0.3):               {len(neutral):3d} fills, avg markout=${neutral['markout_5s'].mean():.4f}")
    print(f"  Strong ask imbalance (<-0.3): {len(strong_ask):3d} fills, avg markout=${strong_ask['markout_5s'].mean():.4f}")

    # Check if trading WITH imbalance is better than against
    # When imbalance > 0 and we buy (dir_yes = 1), we're trading WITH the flow
    # When imbalance < 0 and we sell (dir_yes = -1), we're trading WITH the flow
    with_flow = df_imb[((df_imb['book_imbalance'] > 0.2) & (df_imb['dir_yes'] == 1)) |
                       ((df_imb['book_imbalance'] < -0.2) & (df_imb['dir_yes'] == -1))]
    against_flow = df_imb[((df_imb['book_imbalance'] > 0.2) & (df_imb['dir_yes'] == -1)) |
                          ((df_imb['book_imbalance'] < -0.2) & (df_imb['dir_yes'] == 1))]

    print(f"\nTrading WITH order flow:     {len(with_flow):3d} fills, avg markout=${with_flow['markout_5s'].mean():.4f}")
    print(f"Trading AGAINST order flow:  {len(against_flow):3d} fills, avg markout=${against_flow['markout_5s'].mean():.4f}")

    if len(with_flow) > 0 and len(against_flow) > 0:
        comparison = compare_groups(with_flow['markout_5s'], against_flow['markout_5s'], "With Flow", "Against Flow")
        print(f"  Comparison: {comparison}")

        if with_flow['markout_5s'].mean() > against_flow['markout_5s'].mean():
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

    z_corr = df_z['aligned_zscore'].corr(df_z['markout_5s'])
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

    print(f"  High +z (>0.8):     {len(high_pos_z):3d} fills, avg markout=${high_pos_z['markout_5s'].mean():.4f}")
    print(f"  Med +z (0.3-0.8):   {len(med_pos_z):3d} fills, avg markout=${med_pos_z['markout_5s'].mean():.4f}")
    print(f"  Neutral (±0.3):     {len(neutral_z):3d} fills, avg markout=${neutral_z['markout_5s'].mean():.4f}")
    print(f"  Med -z (-0.8--0.3): {len(med_neg_z):3d} fills, avg markout=${med_neg_z['markout_5s'].mean():.4f}")
    print(f"  High -z (<-0.8):    {len(high_neg_z):3d} fills, avg markout=${high_neg_z['markout_5s'].mean():.4f}")

    # Z-score predictor logic test
    print(f"\nZ-score predictor effectiveness:")
    print(f"  (Testing if fills during extreme z-score have worse markouts)")

    extreme_z = df_z[abs(df_z['zscore']) > 0.8]
    neutral_z_fills = df_z[abs(df_z['zscore']) <= 0.3]

    if len(extreme_z) > 0 and len(neutral_z_fills) > 0:
        print(f"  Extreme z (|z|>0.8): {len(extreme_z)} fills, avg markout=${extreme_z['markout_5s'].mean():.4f}")
        print(f"  Neutral z (|z|≤0.3): {len(neutral_z_fills)} fills, avg markout=${neutral_z_fills['markout_5s'].mean():.4f}")

        comparison = compare_groups(extreme_z['markout_5s'], neutral_z_fills['markout_5s'],
                                   "Extreme Z", "Neutral Z")
        print(f"  {comparison}")

        if extreme_z['markout_5s'].mean() < neutral_z_fills['markout_5s'].mean():
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
    print(f"    Bought when z>0.8 (RTDS will rise):  {len(bought_high_z)} fills, avg markout=${bought_high_z['markout_5s'].mean():.4f}")
    print(f"    Sold when z<-0.8 (RTDS will fall):   {len(sold_low_z)} fills, avg markout=${sold_low_z['markout_5s'].mean():.4f}")

    print(f"\n  WRONG side (should be canceled by z-score logic):")
    print(f"    Sold when z>0.8 (vulnerable ASK):    {len(sold_high_z)} fills, avg markout=${sold_high_z['markout_5s'].mean():.4f}")
    print(f"    Bought when z<-0.8 (vulnerable BID): {len(bought_low_z)} fills, avg markout=${bought_low_z['markout_5s'].mean():.4f}")

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
        comparison = compare_groups(correct_side['markout_5s'], wrong_side['markout_5s'],
                                   "Correct Side", "Wrong Side")
        print(f"\n  Comparison: {comparison}")

        if correct_side['markout_5s'].mean() > wrong_side['markout_5s'].mean():
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

    skew_corr = df_skew['aligned_z_skew'].corr(df_skew['markout_5s'])
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

    print(f"  High +skew (>0.5¢):     {len(high_pos):3d} fills, avg markout=${high_pos['markout_5s'].mean():.4f}")
    print(f"  Med +skew (0.1-0.5¢):   {len(med_pos):3d} fills, avg markout=${med_pos['markout_5s'].mean():.4f}")
    print(f"  Neutral (±0.1¢):        {len(neutral):3d} fills, avg markout=${neutral['markout_5s'].mean():.4f}")
    print(f"  Med -skew (-0.5--0.1¢): {len(med_neg):3d} fills, avg markout=${med_neg['markout_5s'].mean():.4f}")
    print(f"  High -skew (<-0.5¢):    {len(high_neg):3d} fills, avg markout=${high_neg['markout_5s'].mean():.4f}")

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
    print(f"    Bought when z_skew>+0.3¢:  {len(bought_pos_skew)} fills, avg markout=${bought_pos_skew['markout_5s'].mean():.4f}")
    print(f"    Sold when z_skew<-0.3¢:    {len(sold_neg_skew)} fills, avg markout=${sold_neg_skew['markout_5s'].mean():.4f}")

    print(f"\n  UNFAVORABLE direction (trading AGAINST z-skew signal):")
    print(f"    Sold when z_skew>+0.3¢:    {len(sold_pos_skew)} fills, avg markout=${sold_pos_skew['markout_5s'].mean():.4f}")
    print(f"    Bought when z_skew<-0.3¢:  {len(bought_neg_skew)} fills, avg markout=${bought_neg_skew['markout_5s'].mean():.4f}")

    # Compare favorable vs unfavorable
    favorable = pd.concat([bought_pos_skew, sold_neg_skew])
    unfavorable = pd.concat([sold_pos_skew, bought_neg_skew])

    if len(favorable) > 0 and len(unfavorable) > 0:
        comparison = compare_groups(favorable['markout_5s'], unfavorable['markout_5s'],
                                   "Favorable", "Unfavorable")
        print(f"\n  Comparison: {comparison}")

        if favorable['markout_5s'].mean() > unfavorable['markout_5s'].mean():
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
        print(f"  Extreme skew (|skew|>0.5¢): {len(extreme_skew)} fills, avg markout=${extreme_skew['markout_5s'].mean():.4f}")
        print(f"  Neutral skew (|skew|≤0.1¢): {len(neutral_skew)} fills, avg markout=${neutral_skew['markout_5s'].mean():.4f}")

        comparison = compare_groups(extreme_skew['markout_5s'], neutral_skew['markout_5s'],
                                   "Extreme Skew", "Neutral Skew")
        print(f"  {comparison}")

        if extreme_skew['markout_5s'].mean() > neutral_skew['markout_5s'].mean():
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

    implied_corr = df_vol['edge_vs_theo'].corr(df_vol['markout_5s'])
    realized_corr = df_vol['edge_vs_realized_theo'].corr(df_vol['markout_5s'])

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

        print(f"  High realized vol (vol_edge > 5%): {len(high_realized)} fills, avg markout=${high_realized['markout_5s'].mean():.4f}")
        print(f"  Low realized vol (vol_edge < -5%): {len(low_realized)} fills, avg markout=${low_realized['markout_5s'].mean():.4f}")

        if len(high_realized) >= MIN_SAMPLES_FOR_SIGNIFICANCE and len(low_realized) >= MIN_SAMPLES_FOR_SIGNIFICANCE:
            comparison = compare_groups(high_realized['markout_5s'], low_realized['markout_5s'],
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

        print(f"  Bought when realized > implied: {len(bought_high_vol)} fills, avg markout=${bought_high_vol['markout_5s'].mean():.4f}")
        print(f"  Sold when realized > implied:   {len(sold_high_vol)} fills, avg markout=${sold_high_vol['markout_5s'].mean():.4f}")
        print(f"  Bought when implied > realized: {len(bought_low_vol)} fills, avg markout=${bought_low_vol['markout_5s'].mean():.4f}")
        print(f"  Sold when implied > realized:   {len(sold_low_vol)} fills, avg markout=${sold_low_vol['markout_5s'].mean():.4f}")

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
            print(f"  {bucket:12s}: {len(subset):3d} fills, avg markout=${subset['markout_5s'].mean():.4f}")

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
        worst = bad_fills.nsmallest(5, 'markout_5s')[['timestamp', 'edge_vs_fair', 'markout_5s', 'momentum', 'momentum_volatility']]
        print(worst.to_string(index=False))

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

    print(f"Fill counts:")
    print(f"  Maker (GTC): {len(gtc)} fills ({len(gtc)/len(df)*100:.1f}%)")
    print(f"  Taker (IOC): {len(ioc)} fills ({len(ioc)/len(df)*100:.1f}%)")

    print(f"\nMarkout performance (with significance tests):")
    for horizon in [1, 5, 15, 30, 60]:
        col = f'markout_{horizon}s'
        if col in df.columns:
            gtc_avg = gtc[col].mean() if len(gtc) > 0 else 0
            ioc_avg = ioc[col].mean() if len(ioc) > 0 else 0
            comparison = compare_groups(gtc[col], ioc[col], "Maker", "Taker") if len(gtc) > 0 and len(ioc) > 0 else ""
            print(f"  {horizon}s: Maker=${gtc_avg:.4f}, Taker=${ioc_avg:.4f}")
            if comparison:
                print(f"       {comparison}")

    if len(gtc) > 0 and len(ioc) > 0:
        gtc_total = gtc['markout_5s'].sum()
        ioc_total = ioc['markout_5s'].sum()

        print(f"\nTotal P&L (5s markout):")
        print(f"  Maker (GTC): ${gtc_total:.2f}")
        print(f"  Taker (IOC): ${ioc_total:.2f}")

        gtc_hit = (gtc['markout_5s'] > 0).sum() / len(gtc) * 100 if len(gtc) > 0 else 0
        ioc_hit = (ioc['markout_5s'] > 0).sum() / len(ioc) * 100 if len(ioc) > 0 else 0

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

def summary_and_diagnosis(df):
    """Overall diagnosis and recommendations."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)

    issues = []

    # Check overall profitability
    avg_markout = df['markout_5s'].mean()
    if avg_markout < -0.001:
        issues.append("❌ LOSING MONEY: Negative avg markout")
    elif avg_markout < 0.001:
        issues.append("⚠️  BREAKEVEN: Near-zero avg markout")
    else:
        issues.append("✅ PROFITABLE: Positive avg markout")

    # Check delta
    if 'delta' in df.columns and df['delta'].mean() < 0.00001:
        issues.append("❌ CRITICAL: Delta near zero - momentum not working!")

    # Check momentum correlation
    if 'momentum' in df.columns:
        mom_corr = df['momentum'].corr(df['markout_5s'])
        if mom_corr < 0:
            issues.append("❌ Negative momentum correlation - strategy backwards?")
        elif mom_corr < 0.05:
            issues.append("⚠️  Weak momentum correlation - edge unclear")

    # Check theo value
    if 'edge_vs_theo' in df.columns:
        theo_corr = df['edge_vs_theo'].corr(df['markout_5s'])
        if theo_corr < 0:
            issues.append("❌ Theo has negative correlation - model is wrong")
        elif theo_corr > 0.15:
            issues.append("✅ Theo has value - model is predictive")

    # Check adverse selection
    hit_rate = (df['markout_5s'] > 0).sum() / len(df) * 100
    if hit_rate < 45:
        issues.append("❌ Low hit rate - getting adverse selected")

    print("\nKey findings:")
    for issue in issues:
        print(f"  {issue}")

    print(f"\nRecommendations:")
    if avg_markout < 0:
        print("  1. STOP TRADING - You're losing money!")
        print("  2. Check if delta is working (should be 0.0001-0.01)")
        print("  3. Review momentum vs theo correlations")
        if mom_corr < 0.05:
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
    maker_vs_taker_analysis(df)  # NEW: Compare GTC vs IOC fills
    momentum_analysis(df)
    zscore_predictor_analysis(df)  # NEW: Analyze z-score predictor
    z_skew_analysis(df)  # NEW: Analyze z-score skew (continuous fair value adjustment)
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
