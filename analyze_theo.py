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


if __name__ == "__main__":
    analyze_theo_accuracy()
