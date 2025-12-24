import asyncio  # Asynchronous I/O
import json  # JSON handling
import websockets  # WebSocket client
import traceback  # Exception handling
from process_data import process_data, process_book_data, process_user_data, process_price_change
import time

#from poly_data.data_processing import process_data, process_user_data
#import poly_data.global_state as global_state

#chunk = ["106393185783537449644078805690800189614172565600484574723938260022241088055271"]

# TODO: All markets gives you the up token and the down token, so can parse that out- all markets also includes the market id- so can have market id, and then yes token and no token

async def connect_market_websocket(chunk):
    """
    Connect to Polymarket's market WebSocket API and process market updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket API
    2. Subscribes to updates for a specified list of market tokens
    3. Processes incoming order book and price updates

    Args:
        chunk (list): List of token IDs to subscribe to

    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare and send subscription message
        message = {"assets_ids": chunk}
        await websocket.send(json.dumps(message))

        print("\n")
        print(f"Sent market subscription message: {message}")

        try:
            # Process incoming market data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                if type(json_data) != list:
                    json_data = [json_data]
                #print(json_data)
                # Process order book updates and trigger trading as needed
                process_data(json_data)
        except websockets.ConnectionClosed:
            print("Connection closed in market websocket")
            print(traceback.format_exc())
        except Exception as e:
            print(f"Exception in market websocket: {e}")
            print(traceback.format_exc())
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)

#asyncio.run(connect_market_websocket(chunk))

#TODO: NEED TO DIFFERENTIATE BETWEEN YES ASSET AND NO ASSET, HAVE DIFFERENT ASSET IDs, NEED TO PROCESS ORDER BOOK SOMEHOW


async def connect_user_websocket(key, secret, passphrase):
    """
    Connect to Polymarket's user WebSocket API and process order/trade updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket user API
    2. Authenticates using API credentials
    3. Processes incoming order and trade updates for the user

    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare authentication message with API credentials
        message = {
            "type": "user",
            "auth": {
                "apiKey": key,
                "secret": secret,
                "passphrase": passphrase
            }
        }

        # Send authentication message
        await websocket.send(json.dumps(message))

        print("\n")
        print(f"Sent user subscription message")

        try:
            # Process incoming user data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                # Process trade and order updates
                #print(json_data)
                process_user_data(json_data) # TODO: bring back websocket so can process user data
        except websockets.ConnectionClosed:
            print("Connection closed in user websocket")
            print(traceback.format_exc())
        except Exception as e:
            print(f"Exception in user websocket: {e}")
            print(traceback.format_exc())
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)