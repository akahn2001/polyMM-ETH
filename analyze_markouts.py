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

    # Correlation
    mom_corr = df['binance_momentum'].corr(df['markout_5s'])
    print(f"Momentum → Markout correlation: {mom_corr:.3f}")
    if mom_corr > 0.1:
        print("  ✅ Positive correlation - momentum has predictive power!")
    elif mom_corr > 0:
        print("  ⚠️  Weak positive correlation - momentum helps slightly")
    else:
        print("  ❌ Negative/zero correlation - momentum NOT working!")

    # Performance by momentum regime
    print(f"\nPerformance by Binance momentum:")

    rising = df[df['binance_momentum'] > 10]
    falling = df[df['binance_momentum'] < -10]
    flat = df[abs(df['binance_momentum']) <= 10]

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
    momentum_fills = df[abs(df['binance_momentum']) > 10]
    flat_fills = df[abs(df['binance_momentum']) <= 10]

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
    print(f"\nAverage |binance_momentum| by fill type:")
    if len(bad_fills) > 0:
        bad_mom = bad_fills['binance_momentum'].abs().mean()
        print(f"  Bad fills:  ${bad_mom:.2f}")
    if len(good_fills) > 0:
        good_mom = good_fills['binance_momentum'].abs().mean()
        print(f"  Good fills: ${good_mom:.2f}")
    if len(neutral) > 0:
        neutral_mom = neutral['binance_momentum'].abs().mean()
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
        worst = bad_fills.nsmallest(5, 'markout_5s')[['timestamp', 'edge_vs_fair', 'markout_5s', 'binance_momentum', 'momentum_volatility']]
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
        print(f"\nEdge vs theo:")
        print(f"  Maker avg edge: ${gtc['edge_vs_theo'].mean():.4f}")
        print(f"  Taker avg edge: ${ioc['edge_vs_theo'].mean():.4f}")

        # Momentum during fills
        print(f"\nMomentum during fills:")
        print(f"  Maker avg |momentum|: ${gtc['binance_momentum'].abs().mean():.2f}")
        print(f"  Taker avg |momentum|: ${ioc['binance_momentum'].abs().mean():.2f}")

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
    if df['delta'].mean() < 0.00001:
        issues.append("❌ CRITICAL: Delta near zero - momentum not working!")

    # Check momentum correlation
    mom_corr = df['binance_momentum'].corr(df['markout_5s'])
    if mom_corr < 0:
        issues.append("❌ Negative momentum correlation - strategy backwards?")
    elif mom_corr < 0.05:
        issues.append("⚠️  Weak momentum correlation - edge unclear")

    # Check theo value
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
    theo_value_test(df)
    model_vs_market_test(df)
    adverse_selection_test(df)
    inventory_analysis(df)
    directional_bias(df)
    volatility_spread_analysis(df)
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
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
