"""
Simple standalone fee rate checker
Doesn't require any of your bot code - just tests a token ID
"""

import requests

def check_fee_rate(token_id: str):
    """Check if a market has taker fees enabled"""
    url = f"https://clob.polymarket.com/fee-rate?token_id={token_id}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        fee_bps = data.get('fee_rate_bps')

        print(f"Token ID: {token_id}")
        print(f"Fee Rate: {fee_bps} bps")

        if fee_bps == 1000:
            print("Status: ⚠️  FEES ENABLED (10% base rate before volume tiers)")
            print("        Taker orders will incur fees!")
        elif fee_bps == 0:
            print("Status: ✅ FEE-FREE")
        else:
            print(f"Status: Unknown ({fee_bps} bps)")

        return data

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Example: Test with a specific token ID
    # Replace with your actual token ID from btc_15min_markets.csv
    test_token = "71321045679252212594626385532706912750332728571942532289631379312455583992563"

    print("Checking Polymarket Fee Rates")
    print("=" * 70)
    check_fee_rate(test_token)
    print()
    print("To check your own token:")
    print("  python check_fee_rates.py")
    print("  Then edit the test_token variable in the script")
