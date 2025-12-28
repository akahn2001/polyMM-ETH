"""
Claim winnings from resolved Polymarket positions.

This script:
1. Queries your positions via the data-api
2. Filters for redeemable positions (resolved markets where you hold winning tokens)
3. Displays what can be claimed
4. Attempts redemption via the relayer API

Usage:
    python claim_winnings.py           # Show redeemable positions
    python claim_winnings.py --claim   # Actually claim them
"""

import os
import requests
import json
import time
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

load_dotenv()

# API endpoints
DATA_API = "https://data-api.polymarket.com"
RELAYER_API = "https://relayer.polymarket.com"  # May need adjustment

# Contract addresses on Polygon
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
NEG_RISK_CTF_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# Null bytes32 for parentCollectionId
PARENT_COLLECTION_ID = "0x0000000000000000000000000000000000000000000000000000000000000000"


def get_proxy_wallet(funder_address: str) -> str:
    """Get the proxy wallet address for a funder address."""
    url = f"https://gamma-api.polymarket.com/users/{funder_address.lower()}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("proxyWallet") or data.get("proxy_wallet")
    return None


def get_positions(user_address: str, redeemable_only: bool = False) -> list:
    """
    Get positions for a user address.

    Args:
        user_address: The proxy wallet address (not funder address)
        redeemable_only: If True, only return positions that can be redeemed

    Returns:
        List of position dictionaries
    """
    params = {
        "user": user_address,
        "sizeThreshold": 0.01,
        "limit": 500,
    }

    if redeemable_only:
        params["redeemable"] = "true"

    resp = requests.get(f"{DATA_API}/positions", params=params)

    if resp.status_code != 200:
        print(f"Error fetching positions: {resp.status_code} - {resp.text}")
        return []

    return resp.json()


def display_positions(positions: list, title: str = "Positions"):
    """Pretty print positions."""
    if not positions:
        print(f"\n{title}: None found")
        return

    print(f"\n{'='*60}")
    print(f"{title} ({len(positions)} total)")
    print('='*60)

    total_redeemable_value = 0

    for pos in positions:
        title = pos.get("title", "Unknown")[:50]
        outcome = pos.get("outcome", "?")
        size = float(pos.get("size", 0))
        cur_price = float(pos.get("curPrice", 0))
        current_value = float(pos.get("currentValue", 0))
        redeemable = pos.get("redeemable", False)
        condition_id = pos.get("conditionId", "")
        neg_risk = pos.get("negRisk", False)

        status = "✓ REDEEMABLE" if redeemable else ""

        print(f"\n{title}")
        print(f"  Outcome: {outcome} | Size: {size:.2f} | Value: ${current_value:.2f} {status}")
        print(f"  Condition: {condition_id[:20]}... | NegRisk: {neg_risk}")

        if redeemable:
            total_redeemable_value += current_value

    if total_redeemable_value > 0:
        print(f"\n{'='*60}")
        print(f"TOTAL REDEEMABLE: ${total_redeemable_value:.2f}")
        print('='*60)


def create_redeem_signature(private_key: str, condition_id: str, timestamp: int) -> str:
    """Create a signature for the redeem request."""
    # This is a simplified version - actual implementation may differ
    message = f"redeem:{condition_id}:{timestamp}"
    account = Account.from_key(private_key)
    message_hash = encode_defunct(text=message)
    signed = account.sign_message(message_hash)
    return signed.signature.hex()


def redeem_position_via_relayer(
    private_key: str,
    funder_address: str,
    condition_id: str,
    index_sets: list = [1, 2],
    neg_risk: bool = False
) -> dict:
    """
    Attempt to redeem a position via Polymarket's relayer.

    Note: This may require additional authentication setup.
    """
    account = Account.from_key(private_key)

    # Determine which contract to use
    if neg_risk:
        ctf_address = NEG_RISK_CTF_ADDRESS
    else:
        ctf_address = CTF_ADDRESS

    # Encode the redeemPositions call
    web3 = Web3()

    # ABI for redeemPositions
    redeem_abi = [{
        "name": "redeemPositions",
        "type": "function",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"}
        ]
    }]

    contract = web3.eth.contract(address=ctf_address, abi=redeem_abi)

    # Ensure condition_id is bytes32
    if not condition_id.startswith("0x"):
        condition_id = "0x" + condition_id

    call_data = contract.encode_abi(
        fn_name="redeemPositions",
        args=[
            USDC_ADDRESS,
            bytes.fromhex(PARENT_COLLECTION_ID[2:]),
            bytes.fromhex(condition_id[2:]),
            index_sets
        ]
    )

    print(f"  Encoded call data: {call_data[:50]}...")

    # Try to submit via relayer
    # Note: This endpoint structure is speculative - may need adjustment
    timestamp = int(time.time())

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "target": ctf_address,
        "data": call_data,
        "owner": funder_address,
        "timestamp": timestamp,
    }

    # Sign the payload
    message = json.dumps(payload, sort_keys=True)
    signature = create_redeem_signature(private_key, condition_id, timestamp)

    headers["X-Signature"] = signature
    headers["X-Address"] = funder_address

    print(f"  Submitting to relayer...")

    try:
        resp = requests.post(
            f"{RELAYER_API}/execute",
            headers=headers,
            json=payload,
            timeout=30
        )

        if resp.status_code == 200:
            return {"success": True, "response": resp.json()}
        else:
            return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claim Polymarket winnings")
    parser.add_argument("--claim", action="store_true", help="Actually claim the positions")
    parser.add_argument("--all", action="store_true", help="Show all positions, not just redeemable")
    args = parser.parse_args()

    # Load credentials
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    funder_address = os.environ.get("POLY_FUNDER_ADDRESS")

    if not private_key or not funder_address:
        print("ERROR: Set POLY_PRIVATE_KEY and POLY_FUNDER_ADDRESS in .env")
        return

    print(f"Funder address: {funder_address}")

    # Get proxy wallet
    proxy_wallet = get_proxy_wallet(funder_address)
    if not proxy_wallet:
        print("ERROR: Could not find proxy wallet for funder address")
        print("Using funder address directly...")
        proxy_wallet = funder_address
    else:
        print(f"Proxy wallet: {proxy_wallet}")

    # Get positions
    if args.all:
        positions = get_positions(proxy_wallet, redeemable_only=False)
        display_positions(positions, "All Positions")

    # Get redeemable positions
    redeemable = get_positions(proxy_wallet, redeemable_only=True)
    display_positions(redeemable, "Redeemable Positions")

    if not redeemable:
        print("\nNo positions to redeem!")
        return

    if not args.claim:
        print("\n" + "="*60)
        print("To claim these positions, run:")
        print("  python claim_winnings.py --claim")
        print("="*60)
        return

    # Attempt redemption
    print("\n" + "="*60)
    print("ATTEMPTING REDEMPTION")
    print("="*60)

    for pos in redeemable:
        title = pos.get("title", "Unknown")[:40]
        condition_id = pos.get("conditionId")
        neg_risk = pos.get("negRisk", False)
        size = float(pos.get("size", 0))

        print(f"\nRedeeming: {title}")
        print(f"  Condition: {condition_id}")
        print(f"  Size: {size:.2f}")

        result = redeem_position_via_relayer(
            private_key=private_key,
            funder_address=funder_address,
            condition_id=condition_id,
            neg_risk=neg_risk
        )

        if result["success"]:
            print(f"  ✓ SUCCESS: {result['response']}")
        else:
            print(f"  ✗ FAILED: {result['error']}")
            print("  Note: Redemption via relayer may require additional setup.")
            print("  Try claiming manually at https://polymarket.com/portfolio")


if __name__ == "__main__":
    main()
