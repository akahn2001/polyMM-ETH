#!/usr/bin/env python3
"""
Fetch all Polymarket markets and save to CSV.
Run this before starting the bot to get the latest markets.

Usage:
    python fetch_markets.py
"""

import pandas as pd
from datetime import datetime
from polymarket_client import PolymarketClient, get_all_markets

# Standard filename that main.py will look for
OUTPUT_FILE = "all_markets.csv"


def main():
    print("[FETCH] Initializing Polymarket client...")
    client = PolymarketClient()

    print("[FETCH] Fetching all markets (this may take a minute)...")
    df = get_all_markets(client.client)

    print(f"[FETCH] Retrieved {len(df)} total markets")

    # Save to standard filename
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[FETCH] Saved to {OUTPUT_FILE}")

    # Also save a dated backup
    date_str = datetime.now().strftime("%m_%d")
    backup_file = f"all_markets_df_{date_str}.csv"
    df.to_csv(backup_file, index=False)
    print(f"[FETCH] Backup saved to {backup_file}")

    # Show some stats about BTC markets
    btc_markets = df[
        df['question'].str.contains('Bitcoin Up or Down', na=False) &
        df['question'].str.contains(r'\d+:\d+[AP]M-\d+:\d+[AP]M', na=False, regex=True)
    ]
    print(f"[FETCH] Found {len(btc_markets)} BTC Up/Down markets with time patterns")

    # Show upcoming markets (accepting orders)
    if 'accepting_orders' in df.columns:
        accepting = btc_markets[btc_markets['accepting_orders'].astype(str).str.upper() == 'TRUE']
        print(f"[FETCH] {len(accepting)} are currently accepting orders")


if __name__ == "__main__":
    main()
