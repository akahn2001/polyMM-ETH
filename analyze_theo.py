"""
Compare Blended Theo vs Binance Theo against actual market mid.
Which theo better predicts where the market goes?
"""
import pandas as pd
import numpy as np

def analyze_theo_accuracy(csv_path="theo_comparison.csv", horizons=[1, 5, 15, 30, 60]):
    df = pd.read_csv(csv_path)

    # Drop rows with missing data
    df = df.dropna(subset=['blended_theo', 'binance_theo', 'market_mid'])
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()) / 60:.1f} minutes")
    print()

    results = []

    for horizon in horizons:
        # Calculate future market mid
        df[f'future_mid_{horizon}'] = df['market_mid'].shift(-horizon)

        # Calculate errors: how far was each theo from future market?
        df[f'blended_err_{horizon}'] = df['blended_theo'] - df[f'future_mid_{horizon}']
        df[f'binance_err_{horizon}'] = df['binance_theo'] - df[f'future_mid_{horizon}']

        # Get rows where we have future data
        valid = df.dropna(subset=[f'future_mid_{horizon}'])

        if len(valid) < 10:
            continue

        # Mean absolute error
        blended_mae = valid[f'blended_err_{horizon}'].abs().mean()
        binance_mae = valid[f'binance_err_{horizon}'].abs().mean()

        # Mean squared error
        blended_mse = (valid[f'blended_err_{horizon}'] ** 2).mean()
        binance_mse = (valid[f'binance_err_{horizon}'] ** 2).mean()

        # Bias (is one systematically high or low?)
        blended_bias = valid[f'blended_err_{horizon}'].mean()
        binance_bias = valid[f'binance_err_{horizon}'].mean()

        # Which was closer more often?
        blended_closer = (valid[f'blended_err_{horizon}'].abs() < valid[f'binance_err_{horizon}'].abs()).mean()

        results.append({
            'horizon': horizon,
            'n': len(valid),
            'blended_mae': blended_mae,
            'binance_mae': binance_mae,
            'mae_winner': 'BLENDED' if blended_mae < binance_mae else 'BINANCE',
            'blended_mse': blended_mse,
            'binance_mse': binance_mse,
            'blended_bias': blended_bias,
            'binance_bias': binance_bias,
            'blended_closer_pct': blended_closer * 100
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No valid data for analysis. Check your CSV has enough rows.")
        print(f"Rows after dropping NaN: {len(df)}")
        return None

    print("=" * 70)
    print("THEO ACCURACY COMPARISON: Which predicts market mid better?")
    print("=" * 70)
    print()

    for _, row in results_df.iterrows():
        print(f"--- Horizon: {row['horizon']} ticks ({row['n']} samples) ---")
        print(f"  Mean Absolute Error:")
        print(f"    Blended: {row['blended_mae']:.5f}")
        print(f"    Binance: {row['binance_mae']:.5f}")
        print(f"    Winner:  {row['mae_winner']} (lower is better)")
        print()
        print(f"  Bias (+ = theo too high, - = too low):")
        print(f"    Blended: {row['blended_bias']:+.5f}")
        print(f"    Binance: {row['binance_bias']:+.5f}")
        print()
        print(f"  Blended was closer: {row['blended_closer_pct']:.1f}% of the time")
        print()

    # Overall summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    blended_wins = sum(1 for _, r in results_df.iterrows() if r['mae_winner'] == 'BLENDED')
    binance_wins = len(results_df) - blended_wins
    print(f"Blended wins on MAE: {blended_wins}/{len(results_df)} horizons")
    print(f"Binance wins on MAE: {binance_wins}/{len(results_df)} horizons")

    avg_closer = results_df['blended_closer_pct'].mean()
    print(f"Blended was closer on average: {avg_closer:.1f}% of ticks")

    if avg_closer > 55:
        print("\n=> BLENDED THEO appears more accurate")
    elif avg_closer < 45:
        print("\n=> BINANCE THEO appears more accurate")
    else:
        print("\n=> Both theos are similar in accuracy")

    return results_df


def analyze_lead_lag(csv_path="theo_comparison.csv", max_lag=20):
    """
    Test if theos lead or lag the market.
    Positive correlation at lag N means: theo change predicts market change N ticks later (theo LEADS)
    Negative lag means: market moved first, theo followed (theo LAGS)
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['blended_theo', 'binance_theo', 'market_mid'])
    df = df.reset_index(drop=True)

    # Calculate changes
    df['blended_chg'] = df['blended_theo'].diff()
    df['binance_chg'] = df['binance_theo'].diff()
    df['market_chg'] = df['market_mid'].diff()

    df = df.dropna()

    print("=" * 70)
    print("LEAD/LAG ANALYSIS: Do theos predict market or follow it?")
    print("=" * 70)
    print()
    print("Positive lag = theo change happens BEFORE market (theo leads)")
    print("Negative lag = market changes BEFORE theo (theo lags)")
    print()

    # Test correlations at different lags
    lags = range(-max_lag, max_lag + 1)

    blended_corrs = []
    binance_corrs = []

    for lag in lags:
        if lag == 0:
            blended_corr = df['blended_chg'].corr(df['market_chg'])
            binance_corr = df['binance_chg'].corr(df['market_chg'])
        elif lag > 0:
            # Theo change at t vs market change at t+lag (theo leads)
            blended_corr = df['blended_chg'].iloc[:-lag].corr(df['market_chg'].iloc[lag:].reset_index(drop=True))
            binance_corr = df['binance_chg'].iloc[:-lag].corr(df['market_chg'].iloc[lag:].reset_index(drop=True))
        else:
            # Market change at t vs theo change at t+|lag| (market leads)
            abs_lag = abs(lag)
            blended_corr = df['market_chg'].iloc[:-abs_lag].corr(df['blended_chg'].iloc[abs_lag:].reset_index(drop=True))
            binance_corr = df['market_chg'].iloc[:-abs_lag].corr(df['binance_chg'].iloc[abs_lag:].reset_index(drop=True))

        blended_corrs.append(blended_corr)
        binance_corrs.append(binance_corr)

    # Find peak correlations
    blended_peak_idx = np.argmax(np.abs(blended_corrs))
    binance_peak_idx = np.argmax(np.abs(binance_corrs))

    blended_peak_lag = list(lags)[blended_peak_idx]
    binance_peak_lag = list(lags)[binance_peak_idx]

    print("BLENDED THEO:")
    print(f"  Peak correlation: {blended_corrs[blended_peak_idx]:.4f} at lag {blended_peak_lag}")
    if blended_peak_lag > 0:
        print(f"  => Blended theo LEADS market by ~{blended_peak_lag} ticks")
    elif blended_peak_lag < 0:
        print(f"  => Blended theo LAGS market by ~{abs(blended_peak_lag)} ticks")
    else:
        print(f"  => Blended theo moves WITH market (no lead/lag)")
    print()

    print("BINANCE THEO:")
    print(f"  Peak correlation: {binance_corrs[binance_peak_idx]:.4f} at lag {binance_peak_lag}")
    if binance_peak_lag > 0:
        print(f"  => Binance theo LEADS market by ~{binance_peak_lag} ticks")
    elif binance_peak_lag < 0:
        print(f"  => Binance theo LAGS market by ~{abs(binance_peak_lag)} ticks")
    else:
        print(f"  => Binance theo moves WITH market (no lead/lag)")
    print()

    # Show correlation at key lags
    print("Correlation at key lags:")
    print(f"  {'Lag':>5}  {'Blended':>10}  {'Binance':>10}  {'Interpretation'}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*30}")
    for i, lag in enumerate(lags):
        if lag in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
            interp = ""
            if lag < 0:
                interp = f"market moved {abs(lag)} ticks ago"
            elif lag > 0:
                interp = f"market moves {lag} ticks later"
            else:
                interp = "simultaneous"
            print(f"  {lag:>5}  {blended_corrs[i]:>10.4f}  {binance_corrs[i]:>10.4f}  {interp}")

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("If theo LEADS: your theo predicts where market will go - good for trading!")
    print("If theo LAGS: you're reacting to market moves - bad, you'll get picked off")
    print()

    return pd.DataFrame({'lag': lags, 'blended_corr': blended_corrs, 'binance_corr': binance_corrs})


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PART 1: ACCURACY ANALYSIS")
    print("="*70 + "\n")
    analyze_theo_accuracy()

    print("\n" + "="*70)
    print("PART 2: LEAD/LAG ANALYSIS")
    print("="*70 + "\n")
    analyze_lead_lag()
