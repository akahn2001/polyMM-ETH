from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType, PartialCreateOrderOptions, OrderType, OpenOrderParams
#from py_clob_client.order_builder.constants import AssetType
import os
import asyncio
import requests
import pandas as pd
from dotenv import load_dotenv

import global_state

# Load environment variables from .env file
load_dotenv()

pd.set_option("display.max_columns", None)

#https://quantpylib.hangukquant.com/wrappers/polymarket/

class PolymarketClient():

    def __init__(self):
        # Load credentials from environment variables
        host = os.environ.get("POLY_HOST", "https://clob.polymarket.com")
        chain_id = int(os.environ.get("POLY_CHAIN_ID", "137"))
        private_key = os.environ.get("POLY_PRIVATE_KEY")
        funder_address = os.environ.get("POLY_FUNDER_ADDRESS")

        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY not set! Create a .env file with your credentials.")
        if not funder_address:
            raise ValueError("POLY_FUNDER_ADDRESS not set! Create a .env file with your credentials.")

        temp_client = ClobClient(host, key=private_key, chain_id=chain_id)
        api_creds = temp_client.create_or_derive_api_creds()

        # See 'Signature Types' note below
        signature_type = 1

        # Initialize trading client
        self.client = ClobClient(
            host,
            key=private_key,
            chain_id=chain_id,
            creds=api_creds,
            signature_type=signature_type,
            funder=funder_address)

        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds=self.creds)


    def get_usdc_balance(self):
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        ba = self.client.get_balance_allowance(
                params)  # L2 method: returns balance + allowance :contentReference[oaicite:1]{index=1}

        print("Raw response:", ba)
        print("USDC balance:", float(ba["balance"])/(10**6))  # typically already in USDC units

    def get_order_book(self, market):
        """
        Get the current order book for a specific market.

        Args:
            market (str): Market ID to query

        Returns:
            tuple: (bids_df, asks_df) - DataFrames containing bid and ask orders
        """
        orderBook = self.client.get_order_book(market)
        return pd.DataFrame(orderBook.bids).astype(float), pd.DataFrame(orderBook.asks).astype(float)

    def get_market_details(self, condition_id: str) -> dict:
        """
        Fetch market details for a given condition_id using Polymarket CLOB API.
        Returns the market dictionary.
        """
        resp = self.client.get_market(condition_id=condition_id)
        print(resp)
        print(condition_id)
        return resp.get("market", resp) if isinstance(resp, dict) else resp

    def create_order(self, token_id, action, price, size, time_in_force="GTC", neg_risk=False):
        """
        Create and submit an order with configurable time-in-force.

        Args:
            token_id (str): YES token ID to trade
            action (str): "BUY" or "SELL"
            price (float): price [0-1]
            size (float): order size
            time_in_force (str): "GTC" or "IOC"
            neg_risk (bool): leave default False

        Returns:
            str: order_id
        """

        # Map string input to Polymarket enum
        tif_map = {
            "GTC": OrderType.GTC,
            "IOC": OrderType.FAK,
        }

        if time_in_force not in tif_map:
            raise ValueError(f"Invalid time_in_force '{time_in_force}'. Expected one of {list(tif_map.keys())}")

        # Construct + sign order
        order_args = OrderArgs(
            price=price,
            size=size,
            side=action.upper(),
            token_id=token_id
        )
        signed_order = self.client.create_order(order_args)

        # Submit to order book
        resp = self.client.post_order(signed_order, tif_map[time_in_force])
        print(f"[TRADE] {action} {size} @ {price:.4f} ({time_in_force}) => {resp['orderID']}")

        return resp["orderID"]

    def get_order(self, order_id: str) -> dict:
        """
        Fetch a single order's details by order_id from Polymarket CLOB API.
        """
        resp = self.client.get_order(order_id=order_id)
        return resp

    def cancel_order(self, order_id: str) -> dict:
        """
        Cancel a single order by ID using the CLOB client.
        Returns the raw response from the API.
        """
        resp = self.client.cancel(order_id=order_id)
        print(f"[CANCEL] {order_id} -> {resp}")
        return resp

    def cancel_orders(self, order_ids: list) -> dict:
        """
        Cancel a given order on Polymarket CLOB by order_id.
        Returns the cancel response.
        """
        resp = self.client.cancel_orders(order_ids=order_ids)
        return resp

    def get_open_orders(self, market: str | None = None):
        """
        Return your open orders. If market is provided, filter to that market/condition id.
        """
        if market is None:
            params = OpenOrderParams()  # all open orders
        else:
            params = OpenOrderParams(market=market)  # filtered
        return self.client.get_orders(params)

    def get_open_orders_all(self):
        """
        Fetch ALL open orders for the account (handles pagination if supported).
        If your py_clob_client doesn't paginate, this still returns the first page.
        """
        out = []
        cursor = None

        while True:
            # Try with cursor if supported; fall back to no cursor
            try:
                params = OpenOrderParams(next_cursor=cursor) if cursor else OpenOrderParams()
                resp = self.client.get_orders(params)
            except TypeError:
                # OpenOrderParams might not accept next_cursor in your version
                resp = self.client.get_orders(OpenOrderParams())

            # resp shape varies: sometimes list, sometimes dict with data/next_cursor
            if isinstance(resp, dict):
                data = resp.get("data") or resp.get("orders") or []
                out.extend(data)
                cursor = resp.get("next_cursor")
                if not cursor:
                    break
            elif isinstance(resp, list):
                out.extend(resp)
                break
            else:
                break

        return out



def get_all_markets(client):
    cursor = ""
    all_markets = []

    while True:
        try:
            markets = client.get_markets(next_cursor=cursor)
            markets_df = pd.DataFrame(markets['data'])

            cursor = markets['next_cursor']

            all_markets.append(markets_df)

            if cursor is None:
                break
        except:
            break

    all_df = pd.concat(all_markets)
    all_df = all_df.reset_index(drop=True)
    return all_df



pclient = PolymarketClient()
#print(pclient.get_open_orders_all())

# TODO: TEST THE OPEN ORDERS CANCEL LOGIC
#pclient.get_usdc_balance()

#order_id = pclient.create_order('83914993881923778482137022469906689704669211086336322092168498921376755497769',"BUY", .42, 5)
#print(order_id)
#print(pclient.get_order(order_id)) # get status == MATCHED but doesnt say size?
#pclient.cancel_orders([order_id])

df = get_all_markets(pclient.client)
df.to_csv("all_markets_df_12_25.csv")
exit()
#filtered_df = df[df['tags'].apply(lambda x: x is not None and "Crypto" in x and "Up or Down" in x)]
#filtered_df = filtered_df[filtered_df["accepting_orders"] == "TRUE"]
#print(filtered_df)
#filtered_df.to_csv("all_markets_df_12_8.csv")


#market_id = "22811028941052873561408639690086570122285924306281326211564552857555508352277"
#bids, offers = pclient.get_order_book(market_id)
#print(bids)
#print(offers)