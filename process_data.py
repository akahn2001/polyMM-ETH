import json
from sortedcontainers import SortedDict
import global_state as global_state
from util import update_fair_vol_for_market, update_fair_value_for_market, update_binance_fair_value_for_market
from trading import update_position_yes_space, perform_trade
#import poly_data.CONSTANTS as CONSTANTS

#from trading import perform_trade
import time
import asyncio
#from poly_data.data_utils import set_position, set_order, update_positions


def process_book_data(asset, json_data):
    # Fires when you connect to the market or when a trade affects the order book
    global_state.all_data[asset] = {
        'asset_id': json_data['asset_id'],  # token_id for the Yes token
        'bids': SortedDict(),
        'asks': SortedDict()
    }

    global_state.all_data[asset]['bids'].update(
        {float(entry['price']): float(entry['size']) for entry in json_data['bids']})
    global_state.all_data[asset]['asks'].update(
        {float(entry['price']): float(entry['size']) for entry in json_data['asks']})


def process_price_change(asset, side, price_level, new_size, asset_id):
    # Fires when there is a change to the order book, ie order is modified or cancelled
    # TODO: THIS FUNCTION SCREWS THINGS UP SOMEHOW, what does this even do in the API?
    #print(asset) # asset in this case is the market id
    #print("PROCESSING PRICE CHG")
    if asset_id != global_state.condition_to_token_id[asset]: # so in all_data global dict asset should map to the yes token?
        return  # skip updates for the No token to prevent duplicated updates
    if side == 'bids':
        book = global_state.all_data[asset]['bids']
    else:
        book = global_state.all_data[asset]['asks']

    if new_size == 0:
        if price_level in book:
            del book[price_level]
    else:
        book[price_level] = new_size


def process_data(json_datas, trade=True):
    #print(json_datas)
    for json_data in json_datas:
        event_type = json_data['event_type']
        asset = json_data['market']
        try:
            asset_id = json_data["asset_id"]
        except:
            asset_id = json_data["price_changes"][0]["asset_id"]

        if event_type == 'book':
            process_book_data(asset, json_data)

            update_fair_vol_for_market(asset)
            update_fair_value_for_market(asset)

            # Update Binance theo with latest vol and Binance spot price
            if hasattr(global_state, 'binance_mid_price') and global_state.binance_mid_price is not None:
                usdtusd = getattr(global_state, 'usdtusd', 1.0)
                binance_spot_usd = global_state.binance_mid_price * usdtusd
                update_binance_fair_value_for_market(asset, binance_spot_usd)

            asyncio.create_task(perform_trade(asset))

        elif event_type == 'price_change':
            for data in json_data['price_changes']:
                asset_id = data["asset_id"]
                side = 'bids' if data['side'] == 'BUY' else 'asks'
                price_level = float(data['price'])
                new_size = float(data['size'])
                process_price_change(asset, side, price_level, new_size, asset_id)
                if trade:
                    #asyncio.create_task(perform_trade(asset))
                    continue
        # pretty_print(f'Received book update for {asset}:', global_state.all_data[asset])
            update_fair_vol_for_market(asset)
            update_fair_value_for_market(asset)

            # Update Binance theo with latest vol and Binance spot price
            if hasattr(global_state, 'binance_mid_price') and global_state.binance_mid_price is not None:
                usdtusd = getattr(global_state, 'usdtusd', 1.0)
                binance_spot_usd = global_state.binance_mid_price * usdtusd
                update_binance_fair_value_for_market(asset, binance_spot_usd)

            asyncio.create_task(perform_trade(asset))

def process_user_data(rows):
    """
    Handle user-level websocket events from Polymarket.

    Behavior:
      - Always log incoming events.
      - Track working orders from ORDER events.
      - Update positions:
          * from ORDER events via size_matched deltas (your resting orders getting filled)
          * from TRADE events for your own trades, but only for status in {"MATCHED", "FILLED"}
    """

    # --- Normalize `rows` into a list of dicts ---
    if isinstance(rows, dict):
        rows_list = [rows]
    elif isinstance(rows, list):
        rows_list = rows
    else:
        print("[USER DEBUG] process_user_data got non-dict/non-list:", type(rows), rows)
        return

    normalized = []
    for item in rows_list:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            print("[USER DEBUG] Skipping non-dict row:", item)

    if not normalized:
        return

    # Ensure global containers exist
    if not hasattr(global_state, "working_orders_by_market"):
        global_state.working_orders_by_market = {}
    if not hasattr(global_state, "positions_by_market"):
        global_state.positions_by_market = {}
    if not hasattr(global_state, "filled_size_by_order"):
        global_state.filled_size_by_order = {}

    my_owner = getattr(global_state, "USER_OWNER_ID", None)

    for row in normalized:
        # Raw log so we always see something
        #print("[USER RAW EVENT]", row)

        event_type = row.get("event_type")
        market     = row.get("market")
        token      = row.get("asset_id")              # outcome token_id (YES or NO)
        side       = (row.get("side") or "").lower()  # "buy"/"sell"
        owner      = row.get("owner")

        # Optional: only process tokens in our YES/NO mapping
        if (token not in global_state.REVERSE_TOKENS and
                token not in global_state.REVERSE_TOKENS.values()):
            #print("[USER DEBUG] Skipping event with unknown token:", token)
            continue

        # ------------- ORDER EVENTS (book-keeping + maker fills) -------------
        if event_type == "order":
            order_id   = row.get("id", "")
            outcome    = row.get("outcome", "")
            order_type = row.get("order_type", "")  # GTC, FOK, etc.
            status     = row.get("status", "")

            try:
                original_sz = float(row.get("original_size", 0) or 0.0)
            except (ValueError, TypeError):
                original_sz = 0.0

            try:
                matched_sz = float(row.get("size_matched", 0) or 0.0)
            except (ValueError, TypeError):
                matched_sz = 0.0

            try:
                price = float(row.get("price", 0) or 0.0)
            except (ValueError, TypeError):
                price = 0.0

            remaining = original_sz - matched_sz

            #print(
             #   f"[USER ORDER] market={market} id={order_id} owner={owner} "
              #  f"order_type={order_type} status={status} "
               # f"side={side} outcome={outcome} "
                #f"orig={original_sz} matched={matched_sz} rem={remaining}"
            #)

            # Only track working orders for *your* orders
            if my_owner is not None and owner is not None and owner != my_owner:
                #print("[USER ORDER] Not our order (owner mismatch), not tracking as working")
                # Also: we don't treat this as our fill
                continue

            orders_for_market = global_state.working_orders_by_market.setdefault(market, {})

            terminal_statuses = {"CANCELLED", "KILLED", "EXPIRED", "FILLED", "FAILED"}

            if status in terminal_statuses or remaining <= 0:
                if order_id in orders_for_market:
                    #print(f"[USER ORDER] Removing order {order_id} from working_orders_by_market")
                    orders_for_market.pop(order_id, None)
            else:
                orders_for_market[order_id] = {
                    "order_id": order_id,
                    "status": status,
                    "order_type": order_type,
                    "side": side,
                    "token": token,
                    "price": price,
                    "original_size": original_sz,
                    "matched_size": matched_sz,
                    "remaining": remaining,
                    "outcome": outcome,
                    "owner": owner,
                }

            # --- NEW: incremental maker fills from size_matched deltas ---
            last_matched = global_state.filled_size_by_order.get(order_id, 0.0)
            delta_fill = matched_sz - last_matched

            if delta_fill > 0:
                global_state.filled_size_by_order[order_id] = matched_sz

                #print(
                 #   f"[USER FILL-ORDER] market={market} order={order_id} "
                  #  f"side={side} token={token} delta_fill={delta_fill} price={price}"
                #)

                # Treat this as a trade in the outcome token (maker fill = GTC)
                update_position_yes_space(
                    market_id=market,
                    outcome_token=token,
                    side=side,
                    size=delta_fill,
                    price=price,
                    order_type="GTC",
                )

                pos = global_state.positions_by_market.get(market)
                if pos is not None:
                    print(
                        f"[POS] market={market} net_yes={pos.net_yes} "
                        f"vwap_yes={pos.vwap_yes:.4f}"
                    )

                try:
                    asyncio.create_task(perform_trade(market))
                except RuntimeError:
                    print("[USER FILL-ORDER] No running event loop for perform_trade")

        # ------------- TRADE EVENTS (taker fills) -------------
        elif event_type == "trade":
            trade_id = row.get("id", "")
            status   = row.get("status", "")
            outcome  = row.get("outcome", "")

            try:
                trade_size = float(
                    row.get("size")
                    or row.get("size_matched")
                    or row.get("size_filled")
                    or 0.0
                )
                trade_price = float(row.get("price", 0) or 0.0)
            except (ValueError, TypeError):
                print("[USER TRADE] Invalid size/price:", row)
                continue

            #print(
             #   f"[USER TRADE] market={market} status={status} owner={owner} "
              #  f"side={side} token={token} outcome={outcome} "
               # f"size={trade_size} price={trade_price}"
            #)

            if trade_size <= 0:
                print("[USER TRADE] Zero-size trade, skipping position update")
                continue

            # Only update positions for OUR trades (when we are the taker)
            if my_owner is not None and owner is not None and owner != my_owner:
                print("[USER TRADE] Trade not ours (owner mismatch), NOT updating position")
                continue

            # Skip non-fill statuses to avoid double counting
            if status not in ("MATCHED", "FILLED"):
                print(f"[USER TRADE] Ignoring status={status} for position update")
                continue

            # ✅ Update YES-space position for taker-side fills (IOC)
            update_position_yes_space(
                market_id=market,
                outcome_token=token,
                side=side,
                size=trade_size,
                price=trade_price,
                order_type="IOC",
            )

            pos = global_state.positions_by_market.get(market)
            if pos is not None:
                print(
                    f"[POS] market={market} net_yes={pos.net_yes} "
                    f"vwap_yes={pos.vwap_yes:.4f}"
                )
            else:
                print(f"[POS] No position object yet for market={market}")

            try:
                asyncio.create_task(perform_trade(market))
            except RuntimeError:
                print("[USER TRADE] No running event loop for perform_trade")