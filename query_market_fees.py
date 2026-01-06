"""
Test script to query Polymarket fee rates
Run this to check if your markets have taker fees enabled
"""

import requests
import global_state
from main import load_markets

def get_fee_rate(token_id: str) -> dict:
    """
    Query Polymarket fee rate for a specific token

    Args:
        token_id: The market's token identifier

    Returns:
        dict with 'fee_rate_bps' key (1000 = 10% base fee, 0 = fee-free)
    """
    url = f"https://clob.polymarket.com/fee-rate?token_id={token_id}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fee rate: {e}")
        return {"fee_rate_bps": None, "error": str(e)}


def test_current_markets():
    """Test fee rates for all markets in btc_15min_markets.csv"""

    # Load markets
    csv_path = "btc_15min_markets.csv"
    all_markets = load_markets(csv_path)

    if not all_markets:
        print(f"No markets found in {csv_path}")
        return

    print(f"Checking fee rates for {len(all_markets)} markets...\n")

    fee_enabled_count = 0
    fee_free_count = 0
    errors = 0

    for market in all_markets[:5]:  # Test first 5 markets
        condition_id = market['condition_id']
        yes_token = market['tokens']['YES']
        no_token = market['tokens']['NO']

        print(f"Market: {market['question'][:60]}...")
        print(f"  Condition ID: {condition_id}")

        # Check YES token fee
        yes_fee = get_fee_rate(yes_token)
        if yes_fee.get('error'):
            print(f"  YES token: ERROR - {yes_fee['error']}")
            errors += 1
        else:
            fee_bps = yes_fee.get('fee_rate_bps')
            if fee_bps == 1000:
                print(f"  YES token: ⚠️  FEES ENABLED (base rate: 10%)")
                fee_enabled_count += 1
            elif fee_bps == 0:
                print(f"  YES token: ✅ Fee-free")
                fee_free_count += 1
            else:
                print(f"  YES token: Unknown rate {fee_bps} bps")

        # Check NO token fee
        no_fee = get_fee_rate(no_token)
        if no_fee.get('error'):
            print(f"  NO token: ERROR - {no_fee['error']}")
            errors += 1
        else:
            fee_bps = no_fee.get('fee_rate_bps')
            if fee_bps == 1000:
                print(f"  NO token: ⚠️  FEES ENABLED (base rate: 10%)")
            elif fee_bps == 0:
                print(f"  NO token: ✅ Fee-free")
            else:
                print(f"  NO token: Unknown rate {fee_bps} bps")

        print()

    print("=" * 70)
    print(f"Summary: {fee_enabled_count} with fees, {fee_free_count} fee-free, {errors} errors")
    print()

    if fee_enabled_count > 0:
        print("⚠️  WARNING: Some markets have taker fees enabled!")
        print("   With MIN_TICKS_BUILD = -1, you may be paying fees.")
        print("   Recommend: Set MIN_TICKS_BUILD = 0 to stay as maker")
    else:
        print("✅ All tested markets are currently fee-free")


def test_single_token(token_id: str):
    """Test a specific token ID"""
    print(f"Querying fee rate for token: {token_id}")
    result = get_fee_rate(token_id)

    if result.get('error'):
        print(f"ERROR: {result['error']}")
    else:
        fee_bps = result.get('fee_rate_bps')
        if fee_bps == 1000:
            print(f"⚠️  FEES ENABLED: Base rate is 10% (1000 bps)")
            print("   Actual fee depends on volume tier (curve-based)")
        elif fee_bps == 0:
            print(f"✅ FEE-FREE: No taker fees for this market")
        else:
            print(f"Fee rate: {fee_bps} bps")

        print(f"\nFull response: {result}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test specific token ID from command line
        token_id = sys.argv[1]
        test_single_token(token_id)
    else:
        # Test all current markets
        test_current_markets()
