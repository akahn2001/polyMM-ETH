"""
Analyze Binance -> RTDS lag and build predictive models.

Reads the CSV from record_price_lag.py and builds models to predict
RTDS price at t+horizon given Binance data at time t.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import global_state for usdtusd conversion rate
try:
    import global_state
except ImportError:
    global_state = None

# Configuration
CSV_FILE = "price_lag_data.csv"
HORIZON = 40  # Predict RTDS price N seconds in the future
TRAIN_SPLIT = 0.7  # Use 70% for training, 30% for testing

# Should try 10-25 for HORIZON as well as 40-60

def load_and_prepare_data(csv_file):
    """Load CSV and prepare separate time series for Binance and RTDS."""
    print(f"Loading data from {csv_file}...")

    df = pd.read_csv(csv_file)
    # Handle mixed timestamp formats (some with microseconds, some without)
    df['timestamp'] = pd.to_datetime(df['event_time_iso'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Total records: {len(df)}")
    print(f"  - Binance: {len(df[df['source'] == 'binance'])}")
    print(f"  - Polymarket: {len(df[df['source'] == 'polymarket'])}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}\n")

    # Split into separate dataframes
    binance_df = df[df['source'] == 'binance'][['timestamp', 'price', 'bid', 'ask']].copy()
    binance_df.columns = ['timestamp', 'binance_price', 'binance_bid', 'binance_ask']

    rtds_df = df[df['source'] == 'polymarket'][['timestamp', 'price']].copy()
    rtds_df.columns = ['timestamp', 'rtds_price']

    return df, binance_df, rtds_df


def adjust_binance_for_usdtusd(binance_df):
    """
    Adjust Binance prices from BTCUSDT to BTCUSD using global_state.usdtusd.

    BTCUSD = BTCUSDT * usdtusd

    Returns adjusted binance_df and the usdtusd rate used.
    """
    if global_state is None or not hasattr(global_state, 'usdtusd'):
        print("WARNING: global_state.usdtusd not available. Using 1.0 (no adjustment).\n")
        return binance_df, 1.0

    usdtusd = global_state.usdtusd

    print("Adjusting Binance prices for USDT/USD conversion...")
    print(f"USDT/USD rate: {usdtusd:.6f}")

    if usdtusd > 1.0:
        premium_pct = (usdtusd - 1.0) * 100
        print(f"USDT at {premium_pct:.4f}% premium")
    elif usdtusd < 1.0:
        discount_pct = (1.0 - usdtusd) * 100
        print(f"USDT at {discount_pct:.4f}% discount")
    else:
        print(f"USDT at parity")

    binance_df_adjusted = binance_df.copy()
    binance_df_adjusted['binance_price'] = binance_df_adjusted['binance_price'] * usdtusd

    # Adjust bid/ask if they exist and are not all NaN
    if 'binance_bid' in binance_df_adjusted.columns and not binance_df_adjusted['binance_bid'].isna().all():
        binance_df_adjusted['binance_bid'] = binance_df_adjusted['binance_bid'] * usdtusd
    if 'binance_ask' in binance_df_adjusted.columns and not binance_df_adjusted['binance_ask'].isna().all():
        binance_df_adjusted['binance_ask'] = binance_df_adjusted['binance_ask'] * usdtusd

    print(f"Converted BTCUSDT → BTCUSD by multiplying by {usdtusd:.6f}\n")

    return binance_df_adjusted, usdtusd


def create_training_data(binance_df, rtds_df, horizon_seconds=10):
    """
    Create training dataset by looking back from each RTDS observation.

    For each RTDS price at time T, we look back 'horizon_seconds' to time T-H,
    and use Binance features at T-H to predict RTDS at time T.

    This simulates: "If I see Binance data now, can I predict RTDS in 'horizon' seconds?"
    """
    print(f"Creating training data with horizon = {horizon_seconds} seconds...")

    training_rows = []

    for idx, rtds_row in rtds_df.iterrows():
        rtds_time = rtds_row['timestamp']
        rtds_price = rtds_row['rtds_price']

        # Look back 'horizon' seconds
        lookback_time = rtds_time - timedelta(seconds=horizon_seconds)

        # Find Binance data around the lookback time
        # Get the most recent Binance tick at or before lookback_time
        binance_at_lookback = binance_df[binance_df['timestamp'] <= lookback_time]

        if len(binance_at_lookback) < 10:  # Need at least 10 Binance ticks for features
            continue

        # Get current Binance price (at lookback time)
        current_binance = binance_at_lookback.iloc[-1]
        current_price = current_binance['binance_price']
        current_bid = current_binance['binance_bid']
        current_ask = current_binance['binance_ask']
        current_spread = current_ask - current_bid

        # Get historical Binance prices for momentum features
        recent_10 = binance_at_lookback.iloc[-10:]

        # Calculate features
        # 1. Current Binance price
        feat_binance_price = current_price

        # 2. Binance momentum (5-second, 10-second)
        if len(binance_at_lookback) >= 2:
            price_5s_ago = binance_at_lookback[
                binance_at_lookback['timestamp'] <= lookback_time - timedelta(seconds=5)
            ]
            if len(price_5s_ago) > 0:
                feat_momentum_5s = current_price - price_5s_ago.iloc[-1]['binance_price']
            else:
                feat_momentum_5s = 0.0
        else:
            feat_momentum_5s = 0.0

        if len(binance_at_lookback) >= 2:
            price_10s_ago = binance_at_lookback[
                binance_at_lookback['timestamp'] <= lookback_time - timedelta(seconds=10)
            ]
            if len(price_10s_ago) > 0:
                feat_momentum_10s = current_price - price_10s_ago.iloc[-1]['binance_price']
            else:
                feat_momentum_10s = 0.0
        else:
            feat_momentum_10s = 0.0

        # 3. Binance volatility (std of last 10 ticks)
        feat_volatility = recent_10['binance_price'].std()

        # 4. Binance bid-ask spread
        feat_spread = current_spread

        # 5. Get most recent RTDS price at lookback time
        rtds_at_lookback = rtds_df[rtds_df['timestamp'] <= lookback_time]
        if len(rtds_at_lookback) > 0:
            last_rtds = rtds_at_lookback.iloc[-1]
            feat_last_rtds_price = last_rtds['rtds_price']
            feat_rtds_staleness = (lookback_time - last_rtds['timestamp']).total_seconds()
            feat_binance_rtds_gap = current_price - feat_last_rtds_price
        else:
            # No RTDS data yet
            continue

        # 6. TIME-AWARENESS FEATURES (to handle variable lag)

        # Momentum strength (absolute rate of change)
        feat_momentum_strength = abs(feat_momentum_10s) if feat_momentum_10s != 0 else 0.0

        # Is RTDS update likely soon? (Binary feature)
        # Typically RTDS updates every 10-20 seconds, so flag if >15s
        feat_update_likely_soon = 1.0 if feat_rtds_staleness > 15.0 else 0.0

        # Gap pressure: combination of gap size and staleness
        # Higher value = RTDS is stale AND far from Binance = high pressure to update
        feat_gap_pressure = abs(feat_binance_rtds_gap) * (feat_rtds_staleness / 10.0)

        # Binance velocity (rate of change per second)
        feat_binance_velocity = feat_momentum_10s / 10.0 if feat_momentum_10s != 0 else 0.0

        # How far has Binance moved since last RTDS update?
        # (This captures accumulated movement during staleness)
        feat_binance_move_since_rtds = abs(current_price - feat_last_rtds_price)

        # Volatility normalized by price (percentage volatility)
        feat_volatility_pct = (feat_volatility / current_price) * 100.0 if current_price > 0 else 0.0

        # Target: RTDS price at current time (horizon seconds in the future from lookback)
        target_rtds_price = rtds_price

        training_rows.append({
            # Metadata
            'rtds_timestamp': rtds_time,
            'lookback_timestamp': lookback_time,
            'horizon_seconds': horizon_seconds,

            # Original Features
            'binance_price': feat_binance_price,
            'momentum_5s': feat_momentum_5s,
            'momentum_10s': feat_momentum_10s,
            'volatility': feat_volatility,
            'bid_ask_spread': feat_spread,
            'last_rtds_price': feat_last_rtds_price,
            'rtds_staleness_seconds': feat_rtds_staleness,
            'binance_rtds_gap': feat_binance_rtds_gap,

            # NEW: Time-Awareness Features
            'momentum_strength': feat_momentum_strength,
            'update_likely_soon': feat_update_likely_soon,
            'gap_pressure': feat_gap_pressure,
            'binance_velocity': feat_binance_velocity,
            'binance_move_since_rtds': feat_binance_move_since_rtds,
            'volatility_pct': feat_volatility_pct,

            # Target
            'target_rtds_price': target_rtds_price
        })

    training_df = pd.DataFrame(training_rows)
    print(f"Created {len(training_df)} training samples\n")

    return training_df


def train_and_evaluate(training_df, train_split=0.7):
    """Train models and evaluate performance."""

    # Define features and target
    feature_cols = [
        # Original features
        'binance_price',
        'momentum_5s',
        'momentum_10s',
        'volatility',
        'bid_ask_spread',
        'last_rtds_price',
        'rtds_staleness_seconds',
        'binance_rtds_gap',

        # NEW: Time-awareness features (to handle variable lag)
        'momentum_strength',
        'update_likely_soon',
        'gap_pressure',
        'binance_velocity',
        'binance_move_since_rtds',
        'volatility_pct'
    ]

    target_col = 'target_rtds_price'

    # Remove rows with NaN
    training_df = training_df.dropna()

    if len(training_df) == 0:
        print("ERROR: No valid training data after removing NaN values")
        return

    # Chronological split (don't shuffle time series data!)
    split_idx = int(len(training_df) * train_split)
    train_df = training_df.iloc[:split_idx]
    test_df = training_df.iloc[split_idx:]

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Train period: {train_df['rtds_timestamp'].min()} to {train_df['rtds_timestamp'].max()}")
    print(f"Test period: {test_df['rtds_timestamp'].min()} to {test_df['rtds_timestamp'].max()}\n")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Baseline 1: just use current Binance price
    print("="*60)
    print("BASELINE 1: Use current Binance price as prediction")
    print("="*60)
    baseline_binance_pred = test_df['binance_price']
    baseline_binance_mae = mean_absolute_error(y_test, baseline_binance_pred)
    baseline_binance_rmse = np.sqrt(mean_squared_error(y_test, baseline_binance_pred))
    print(f"MAE: ${baseline_binance_mae:.2f}")
    print(f"RMSE: ${baseline_binance_rmse:.2f}\n")

    # Baseline 2: just use current RTDS price (assume no change)
    print("="*60)
    print("BASELINE 2: Use current RTDS price as prediction (no-change forecast)")
    print("="*60)
    baseline_rtds_pred = test_df['last_rtds_price']
    baseline_rtds_mae = mean_absolute_error(y_test, baseline_rtds_pred)
    baseline_rtds_rmse = np.sqrt(mean_squared_error(y_test, baseline_rtds_pred))
    print(f"MAE: ${baseline_rtds_mae:.2f}")
    print(f"RMSE: ${baseline_rtds_rmse:.2f}\n")

    # Model 1: Linear Regression
    print("="*60)
    print("MODEL 1: Linear Regression")
    print("="*60)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    print(f"MAE: ${lr_mae:.2f}")
    print(f"RMSE: ${lr_rmse:.2f}")
    print(f"R²: {lr_r2:.4f}")
    print(f"Improvement over Binance baseline: {(baseline_binance_mae - lr_mae) / baseline_binance_mae * 100:.1f}%")
    print(f"Improvement over RTDS baseline: {(baseline_rtds_mae - lr_mae) / baseline_rtds_mae * 100:.1f}%")

    # Show coefficients
    print("\nFeature Coefficients:")
    for feat, coef in zip(feature_cols, lr_model.coef_):
        print(f"  {feat:25s}: {coef:10.4f}")
    print(f"  {'Intercept':25s}: {lr_model.intercept_:10.4f}\n")

    # Model 2: Random Forest
    print("="*60)
    print("MODEL 2: Random Forest")
    print("="*60)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    print(f"MAE: ${rf_mae:.2f}")
    print(f"RMSE: ${rf_rmse:.2f}")
    print(f"R²: {rf_r2:.4f}")
    print(f"Improvement over Binance baseline: {(baseline_binance_mae - rf_mae) / baseline_binance_mae * 100:.1f}%")
    print(f"Improvement over RTDS baseline: {(baseline_rtds_mae - rf_mae) / baseline_rtds_mae * 100:.1f}%")

    # Show feature importance
    print("\nFeature Importance:")
    importances = sorted(zip(feature_cols, rf_model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        print(f"  {feat:25s}: {imp:.4f}")
    print()

    # Analyze errors
    print("="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    test_df_copy = test_df.copy()
    test_df_copy['baseline_binance_pred'] = baseline_binance_pred.values
    test_df_copy['baseline_rtds_pred'] = baseline_rtds_pred.values
    test_df_copy['lr_pred'] = lr_pred
    test_df_copy['rf_pred'] = rf_pred
    test_df_copy['baseline_binance_error'] = baseline_binance_pred.values - y_test.values
    test_df_copy['baseline_rtds_error'] = baseline_rtds_pred.values - y_test.values
    test_df_copy['lr_error'] = lr_pred - y_test.values
    test_df_copy['rf_error'] = rf_pred - y_test.values

    print(f"Mean error (bias):")
    print(f"  Baseline (Binance): ${test_df_copy['baseline_binance_error'].mean():.2f}")
    print(f"  Baseline (RTDS): ${test_df_copy['baseline_rtds_error'].mean():.2f}")
    print(f"  Linear Regression: ${test_df_copy['lr_error'].mean():.2f}")
    print(f"  Random Forest: ${test_df_copy['rf_error'].mean():.2f}\n")

    print(f"Error percentiles:")
    for pct in [50, 75, 90, 95, 99]:
        baseline_binance_err = np.percentile(np.abs(test_df_copy['baseline_binance_error']), pct)
        baseline_rtds_err = np.percentile(np.abs(test_df_copy['baseline_rtds_error']), pct)
        lr_err = np.percentile(np.abs(test_df_copy['lr_error']), pct)
        rf_err = np.percentile(np.abs(test_df_copy['rf_error']), pct)
        print(f"  {pct}th percentile:")
        print(f"    Baseline (Binance): ${baseline_binance_err:.2f}")
        print(f"    Baseline (RTDS): ${baseline_rtds_err:.2f}")
        print(f"    Linear Regression: ${lr_err:.2f}")
        print(f"    Random Forest: ${rf_err:.2f}")

    return {
        'test_df': test_df_copy,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'feature_cols': feature_cols
    }


def plot_predictions(test_df, save_path='predictions_plot.png'):
    """Plot actual vs predicted values."""
    print(f"\nGenerating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Time series of actual vs predicted
    ax = axes[0]
    sample_size = min(200, len(test_df))  # Plot first 200 points
    sample_df = test_df.iloc[:sample_size]

    ax.plot(sample_df['rtds_timestamp'], sample_df['target_rtds_price'],
            'o-', label='Actual RTDS', alpha=0.7, markersize=4, linewidth=2)
    ax.plot(sample_df['rtds_timestamp'], sample_df['baseline_binance_pred'],
            's-', label='Baseline (Binance)', alpha=0.5, markersize=3)
    ax.plot(sample_df['rtds_timestamp'], sample_df['baseline_rtds_pred'],
            'x-', label='Baseline (RTDS no-change)', alpha=0.5, markersize=3)
    ax.plot(sample_df['rtds_timestamp'], sample_df['lr_pred'],
            '^-', label='Linear Reg', alpha=0.5, markersize=3)
    ax.plot(sample_df['rtds_timestamp'], sample_df['rf_pred'],
            'D-', label='Random Forest', alpha=0.5, markersize=3)

    ax.set_xlabel('Time')
    ax.set_ylabel('BTC Price ($)')
    ax.set_title(f'Predictions vs Actual (First {sample_size} test samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter plot - actual vs predicted
    ax = axes[1]
    ax.scatter(test_df['target_rtds_price'], test_df['baseline_binance_pred'],
              alpha=0.3, s=10, label='Baseline (Binance)')
    ax.scatter(test_df['target_rtds_price'], test_df['baseline_rtds_pred'],
              alpha=0.3, s=10, label='Baseline (RTDS)')
    ax.scatter(test_df['target_rtds_price'], test_df['lr_pred'],
              alpha=0.3, s=10, label='Linear Reg')
    ax.scatter(test_df['target_rtds_price'], test_df['rf_pred'],
              alpha=0.3, s=10, label='Random Forest')

    # Perfect prediction line
    min_val = test_df['target_rtds_price'].min()
    max_val = test_df['target_rtds_price'].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction', linewidth=2)

    ax.set_xlabel('Actual RTDS Price ($)')
    ax.set_ylabel('Predicted RTDS Price ($)')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error distribution
    ax = axes[2]
    ax.hist(test_df['baseline_binance_error'], bins=50, alpha=0.4, label='Baseline (Binance)', density=True)
    ax.hist(test_df['baseline_rtds_error'], bins=50, alpha=0.4, label='Baseline (RTDS)', density=True)
    ax.hist(test_df['lr_error'], bins=50, alpha=0.4, label='Linear Reg', density=True)
    ax.hist(test_df['rf_error'], bins=50, alpha=0.4, label='Random Forest', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Prediction Error ($)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

    # Also show it
    # plt.show()  # Uncomment if you want to display interactively


def main():
    """Main analysis pipeline."""
    print("\n" + "="*60)
    print("BINANCE -> RTDS PREDICTIVE MODEL")
    print("="*60 + "\n")

    # Load data
    df, binance_df, rtds_df = load_and_prepare_data(CSV_FILE)

    # Adjust Binance prices using global_state.usdtusd
    binance_df, usdtusd = adjust_binance_for_usdtusd(binance_df)

    # Create training data
    training_df = create_training_data(binance_df, rtds_df, horizon_seconds=HORIZON)

    if len(training_df) == 0:
        print("ERROR: No training data created. Check your CSV file.")
        return

    # Train and evaluate
    results = train_and_evaluate(training_df, train_split=TRAIN_SPLIT)

    if results is None:
        return

    # Plot results
    plot_predictions(results['test_df'])

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nModel predicts RTDS price {HORIZON} seconds in the future")
    print("based on current Binance data.")


if __name__ == "__main__":
    main()
