"""
Record synchronized price data from Binance and Polymarket to analyze RTDS lag.

This script streams prices from:
1. Binance WebSocket (real-time BTC/USDT)
2. Polymarket Chainlink RTDS (BTC/USD)

Data is saved to CSV for later model fitting to predict RTDS from Binance prices.
"""

import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
import sys
import threading

# Import existing stream functions
from binance_price_stream import stream_binance_btcusdt_mid
from price_stream import stream_btc_usd

# Data collection storage
price_data = []
data_lock = threading.Lock()  # Use threading.Lock for synchronous callback
csv_file = None
csv_writer = None

# Configuration
OUTPUT_FILE = "price_lag_data.csv"
FLUSH_INTERVAL = 10  # Flush to disk every N records


def setup_csv_file():
    """Initialize CSV file with headers."""
    global csv_file, csv_writer

    filepath = Path(OUTPUT_FILE)
    file_exists = filepath.exists()

    csv_file = open(OUTPUT_FILE, 'a', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        # Write headers
        csv_writer.writerow([
            'event_timestamp',      # When we received the event
            'event_time_iso',       # Human-readable timestamp
            'source',               # 'binance' or 'polymarket'
            'price',                # The price value
            'source_timestamp',     # Original timestamp from source (if available)
            'bid',                  # Bid price (Binance only)
            'ask'                   # Ask price (Binance only)
        ])
        print(f"Created new CSV file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing CSV file: {OUTPUT_FILE}")


def record_binance_price(mid, bid, ask, ts):
    """Callback for Binance price updates (synchronous)."""
    with data_lock:
        event_time = time.time()
        event_iso = datetime.fromtimestamp(event_time).isoformat()

        row = [
            event_time,
            event_iso,
            'binance',
            mid,
            ts,
            bid,
            ask
        ]

        csv_writer.writerow(row)
        price_data.append({
            'event_time': event_time,
            'source': 'binance',
            'price': mid,
            'bid': bid,
            'ask': ask,
            'source_ts': ts
        })

        # Flush periodically
        if len(price_data) % FLUSH_INTERVAL == 0:
            csv_file.flush()
            #print(f"[{event_iso}] Binance: ${mid:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f}) | Total records: {len(price_data)}")


async def monitor_polymarket_stream():
    """Monitor Polymarket stream and record price updates."""
    # Import global_state to monitor price changes
    try:
        import global_state
    except ImportError:
        print("ERROR: Cannot import global_state. Make sure it exists.")
        return

    last_price = None
    last_timestamp = None

    print("Starting Polymarket price monitor...")

    while True:
        try:
            # Check if price has updated
            current_price = global_state.mid_price
            current_timestamp = global_state.timestamp

            if current_price != last_price or current_timestamp != last_timestamp:
                with data_lock:
                    event_time = time.time()
                    event_iso = datetime.fromtimestamp(event_time).isoformat()

                    row = [
                        event_time,
                        event_iso,
                        'polymarket',
                        current_price,
                        current_timestamp,
                        '',  # No bid for Polymarket
                        ''   # No ask for Polymarket
                    ]

                    csv_writer.writerow(row)
                    price_data.append({
                        'event_time': event_time,
                        'source': 'polymarket',
                        'price': current_price,
                        'source_ts': current_timestamp
                    })

                    # Flush periodically
                    if len(price_data) % FLUSH_INTERVAL == 0:
                        csv_file.flush()
                        #print(f"[{event_iso}] Polymarket: ${current_price:.2f} (RTDS) | Total records: {len(price_data)}")

                    last_price = current_price
                    last_timestamp = current_timestamp

            await asyncio.sleep(0.1)  # Check 10 times per second

        except Exception as e:
            print(f"Error monitoring Polymarket stream: {e}")
            await asyncio.sleep(1)


async def stats_reporter():
    """Periodically report statistics."""
    while True:
        await asyncio.sleep(60)  # Report every minute

        with data_lock:
            if len(price_data) > 0:
                binance_count = sum(1 for d in price_data if d['source'] == 'binance')
                polymarket_count = sum(1 for d in price_data if d['source'] == 'polymarket')

                print(f"\n{'='*60}")
                print(f"STATS: Total records: {len(price_data)}")
                print(f"  - Binance updates: {binance_count}")
                print(f"  - Polymarket updates: {polymarket_count}")
                print(f"  - File: {OUTPUT_FILE}")
                print(f"{'='*60}\n")


async def main():
    """Main entry point - run both streams and record data."""
    setup_csv_file()

    print("\n" + "="*60)
    print("Price Lag Data Recorder")
    print("="*60)
    print(f"Recording to: {OUTPUT_FILE}")
    print("Press Ctrl+C to stop\n")

    try:
        # Run both streams concurrently
        await asyncio.gather(
            # Binance stream with callback
            stream_binance_btcusdt_mid(on_mid=record_binance_price, verbose=True),

            # Polymarket stream (updates global_state)
            stream_btc_usd(),

            # Monitor Polymarket updates
            monitor_polymarket_stream(),

            # Stats reporter
            stats_reporter()
        )
    except KeyboardInterrupt:
        print("\n\nStopping data collection...")
    finally:
        csv_file.flush()
        csv_file.close()
        print(f"\nData saved to {OUTPUT_FILE}")
        print(f"Total records collected: {len(price_data)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
