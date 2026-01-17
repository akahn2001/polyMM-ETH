import gc                      # Garbage collection
import time                    # Time functions
import asyncio                 # Asynchronous I/O
import traceback               # Exception handling
import threading               # Thread management
import psutil                  # Process monitoring

# Use uvloop for faster async (2-4x speedup on event loop operations)
try:
    import uvloop
    uvloop.install()
    print("[INIT] uvloop installed - using faster event loop")
except ImportError:
    print("[INIT] uvloop not available - using default asyncio")
import pandas as pd
import ast
import csv

import global_state
from global_state import all_data, client, token_to_condition_id, condition_to_token_id, REVERSE_TOKENS, net_position
from polymarket_client import PolymarketClient
from websocket_handlers import connect_market_websocket, connect_user_websocket
from price_stream import stream_btc_usd
from binance_price_stream import get_usdt_usd_rate, get_binance_btcusdt_mid, stream_binance_btcusdt_mid
from coinbase_price_stream import stream_coinbase_btcusd_mid
from util import get_best_bid_offer
from rust_math import bs_binary_call, bs_binary_call_implied_vol_closed
from datetime import datetime
from zoneinfo import ZoneInfo
from kalman_filter import VolKalman1D
from price_blend_kalman import PriceBlendKalman
from trading import sweep_zombie_orders_from_working, reconcile_loop, reconcile_loop_all
from markouts import markout_loop, markout_dump_loop
from market_scheduler import load_btc_15min_markets, run_scheduler

async def health_monitor():
    """Log health metrics every 30 seconds to diagnose performance decay."""
    proc = psutil.Process()
    start_time = time.time()

    while True:
        await asyncio.sleep(30)

        uptime_min = (time.time() - start_time) / 60
        mem_mb = proc.memory_info().rss / 1024 / 1024
        tasks = len(asyncio.all_tasks())

        # Count data structure sizes
        markouts_count = len(global_state.markouts) if hasattr(global_state, 'markouts') else 0
        vol_filters_count = len(global_state.vol_filters) if hasattr(global_state, 'vol_filters') else 0
        processed_trades = len(global_state.processed_trade_ids) if hasattr(global_state, 'processed_trade_ids') else 0
        filled_orders = len(global_state.filled_size_by_order) if hasattr(global_state, 'filled_size_by_order') else 0

        print(f"[HEALTH] uptime={uptime_min:.1f}min mem={mem_mb:.0f}MB tasks={tasks} "
              f"markouts={markouts_count} vol_filters={vol_filters_count} "
              f"processed_trades={processed_trades} filled_orders={filled_orders}")

def update_once(client):
    # TODO: too much looping here is wasting time
    all_markets = pd.read_csv("all_markets.csv")
    for row in all_markets.iterrows():
        row=row[1]
        tokens = row["tokens"]
        yes_token = ast.literal_eval(tokens)[0]
        no_token = ast.literal_eval(tokens)[1]
        global_state.REVERSE_TOKENS[yes_token["token_id"]] = no_token["token_id"]
        # TODO: Add no tokens here
        if yes_token["outcome"] == "Down":
            raise Exception("wrong token! ", yes_token)
        condition_to_token_id[row["condition_id"]] = yes_token["token_id"]
        token_to_condition_id[yes_token["token_id"]] = row["condition_id"]


def update_periodically(client):
    last_usdtusd_update = 0

    # Open CSV file for logging
    csv_file = open("theo_comparison.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "rtds_spot", "coinbase_spot", "coinbase_bias", "coinbase_adjusted", "binance_spot_usd", "blended_spot", "theo", "fair_vol", "market_bid", "market_offer", "market_mid", "realized_vol_5m", "realized_vol_15m", "z_score", "z_skew"])

    while True:
        #print(global_state.working_orders_by_market)
        #print("POS: ", global_state.positions_by_market)
        # Print only active market position instead of entire dict
        active_market = global_state.active_market_id
        if active_market:
            pos = global_state.positions_by_market.get(active_market)
            fair = global_state.fair_value.get(active_market)
            pos_str = f"net_yes={pos.net_yes:.1f}" if pos else "no position"
            print(f"Active: {active_market[:8]}... | Pos: {pos_str} | Fair: {fair:.4f}" if fair else f"Active: {active_market[:8]}... | Pos: {pos_str}")

        # Update USDT/USD exchange rate every 60 seconds
        current_time = time.time()
        if current_time - last_usdtusd_update >= 60:
            try:
                usdtusd_rate = get_usdt_usd_rate(verbose=True)
                global_state.usdtusd = usdtusd_rate
                last_usdtusd_update = current_time
                print(f"[UPDATE] Updated global_state.usdtusd = {usdtusd_rate:.6f}")
            except Exception as e:
                print(f"[UPDATE] Error updating USDT/USD rate: {e}")

        # Print theos and spot prices for the first market, and log to CSV
        if global_state.all_tokens:
            market_id = token_to_condition_id.get(global_state.all_tokens[0])
            if market_id:
                main_theo = global_state.fair_value.get(market_id)  # Uses Coinbase if USE_COINBASE_PRICE=True, else blend
                rtds_spot = global_state.mid_price
                fair_vol = global_state.fair_vol.get(market_id)

                # Get all price data (log everything regardless of mode)
                coinbase_spot = global_state.coinbase_mid_price
                coinbase_bias = global_state.coinbase_bias_correction
                coinbase_adjusted = coinbase_spot + coinbase_bias if coinbase_spot is not None else None
                binance_spot_usd = global_state.binance_mid_price * global_state.usdtusd if global_state.binance_mid_price else None
                blended_spot = global_state.blended_price

                # For display, use the active price source
                price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
                if price_source == "COINBASE":
                    exchange_spot = coinbase_spot
                elif price_source == "RTDS":
                    exchange_spot = rtds_spot
                else:  # BLEND
                    exchange_spot = binance_spot_usd

                # Get Polymarket order book bid/offer
                market_bid, market_offer = None, None
                market_mid = None
                try:
                    market_bid, market_offer = get_best_bid_offer(global_state.all_data[market_id])
                    market_mid = (market_bid + market_offer) / 2.0
                except:
                    pass

                # Get spread multiplier for display
                spread_mult = 1.0
                if hasattr(global_state, 'spread_mult_by_market'):
                    spread_mult = global_state.spread_mult_by_market.get(market_id, 1.0)

                # Get signal adjustments for display (in cents)
                z_skew_cents = 0.0
                z_skew_residual_cents = 0.0
                if hasattr(global_state, 'z_skew_by_market'):
                    z_skew_cents = global_state.z_skew_by_market.get(market_id, 0.0) * 100
                if hasattr(global_state, 'z_skew_residual_by_market'):
                    z_skew_residual_cents = global_state.z_skew_residual_by_market.get(market_id, 0.0) * 100

                imbalance_adj_cents = 0.0
                if hasattr(global_state, 'imbalance_adj_by_market'):
                    imbalance_adj_cents = global_state.imbalance_adj_by_market.get(market_id, 0.0) * 100

                total_signal_cents = z_skew_residual_cents + imbalance_adj_cents

                # Only print if we have valid prices
                if price_source == "COINBASE":
                    if exchange_spot is not None and main_theo is not None and fair_vol is not None:
                        # Show Coinbase, bias correction, and adjusted price
                        print(f"RTDS: {rtds_spot:.2f}  CB: {exchange_spot:.2f}  BIAS: {coinbase_bias:+.2f}  CB_ADJ: {coinbase_adjusted:.2f}  |  THEO: {main_theo:.4f}  VOL: {fair_vol:.3f}  SPREAD: {spread_mult:.2f}x  |  SKEWS: Z={z_skew_cents:+.2f}¢(res={z_skew_residual_cents:+.2f}¢) IMB={imbalance_adj_cents:+.2f}¢ TOT={total_signal_cents:+.2f}¢")
                    elif exchange_spot is None and rtds_spot is not None:
                        # Waiting for Coinbase price
                        print(f"[WAITING] RTDS connected ({rtds_spot:.2f}), waiting for Coinbase price...")
                    elif rtds_spot is not None and exchange_spot is not None and (main_theo is None or fair_vol is None):
                        # Waiting for vol/theo calibration
                        print(f"[WAITING] RTDS: {rtds_spot:.2f}  CB: {exchange_spot:.2f}  |  Calibrating vol/theo...")
                elif price_source == "RTDS":
                    if rtds_spot is not None and main_theo is not None and fair_vol is not None:
                        # Show pure RTDS with Coinbase predictor data
                        z_score = getattr(global_state, 'coinbase_rtds_zscore', 0.0)
                        cb_price = coinbase_spot if coinbase_spot is not None else 0.0
                        print(f"RTDS: {rtds_spot:.2f}  CB: {cb_price:.2f}  Z: {z_score:+.2f}  |  THEO: {main_theo:.4f}  VOL: {fair_vol:.3f}  SPREAD: {spread_mult:.2f}x  |  SKEWS: Z={z_skew_cents:+.2f}¢(res={z_skew_residual_cents:+.2f}¢) IMB={imbalance_adj_cents:+.2f}¢ TOT={total_signal_cents:+.2f}¢")
                    elif rtds_spot is None:
                        # Waiting for RTDS
                        print(f"[WAITING] Waiting for RTDS connection...")
                    elif rtds_spot is not None and (main_theo is None or fair_vol is None):
                        # Waiting for vol/theo calibration
                        z_score = getattr(global_state, 'coinbase_rtds_zscore', 0.0)
                        cb_price = coinbase_spot if coinbase_spot is not None else 0.0
                        print(f"[WAITING] RTDS: {rtds_spot:.2f}  CB: {cb_price:.2f}  Z: {z_score:+.2f}  |  Calibrating vol/theo...")
                else:  # BLEND
                    if exchange_spot is not None and blended_spot is not None and main_theo is not None and fair_vol is not None:
                        print(f"RTDS: {rtds_spot:.2f}  BINANCE: {exchange_spot:.2f}  BLEND: {blended_spot:.2f}  |  THEO: {main_theo:.4f}  VOL: {fair_vol:.3f}  SPREAD: {spread_mult:.2f}x  |  SKEWS: Z={z_skew_cents:+.2f}¢(res={z_skew_residual_cents:+.2f}¢) IMB={imbalance_adj_cents:+.2f}¢ TOT={total_signal_cents:+.2f}¢")
                    elif exchange_spot is None and rtds_spot is not None:
                        # Waiting for Binance price
                        print(f"[WAITING] RTDS connected ({rtds_spot:.2f}), waiting for Binance price...")
                    elif rtds_spot is not None and exchange_spot is not None and blended_spot is not None and (main_theo is None or fair_vol is None):
                        # Waiting for vol/theo calibration
                        print(f"[WAITING] RTDS: {rtds_spot:.2f}  BINANCE: {exchange_spot:.2f}  BLEND: {blended_spot:.2f}  |  Calibrating vol/theo...")

                # Get realized vol from global state
                realized_vol_5m = global_state.realized_vol_5m
                realized_vol_15m = global_state.realized_vol_15m

                # Get z-score and z-skew
                z_score = getattr(global_state, 'coinbase_rtds_zscore', 0.0)
                z_skew_by_market = getattr(global_state, 'z_skew_by_market', {})
                z_skew = z_skew_by_market.get(market_id, 0.0)

                # Write to CSV - always log all data regardless of mode
                # Columns: timestamp, rtds_spot, coinbase_spot, coinbase_bias, coinbase_adjusted, binance_spot_usd, blended_spot, theo, fair_vol, market_bid, market_offer, market_mid, realized_vol_5m, realized_vol_15m, z_score, z_skew
                csv_writer.writerow([
                    current_time,
                    rtds_spot,
                    coinbase_spot,
                    coinbase_bias,
                    coinbase_adjusted,
                    binance_spot_usd,
                    blended_spot,
                    main_theo,
                    fair_vol,
                    market_bid,
                    market_offer,
                    market_mid,
                    realized_vol_5m,
                    realized_vol_15m,
                    z_score,
                    z_skew
                ])
                csv_file.flush()  # Ensure data is written immediately

        time.sleep(0.5)

    # TODO: Need to make markets- step 1 is figure out how to use websocket to track filled orders- should prevent the risk issue happening here
    # TODO: Add qty filled variable from takerAmount in order response
    # TODO: Need to define these programatically
    call_strike = 91332.87
    threshold = .030 # TODO: MAKE THRESHOLD DYNAMIC AS A FUNCTION OF FAIR VOL AND dYES CONTRACT/dSPOT
    max_position = 20
    size_traded = 10
    # TODO: Need to make quantity we trade a function of size on the bid/offer!!
    exp = datetime(2025, 12, 8, 20, 15, tzinfo=ZoneInfo("America/New_York"))

    best_bids, best_offers, option_mids, theos, bid_vols, offer_vols, mid_vols, fair_vols, ul_price, mins_exp = [], [], [], [], [], [], [], [], [], []

    kf = VolKalman1D(
        x0=.35,
        P0 = .08**2,
        process_var = .0003 **2, # how much vol can move per tick (tunable)
        meas_var = .10 ** 2, # R: how noisy each vol observation is (tunable)
        spread_sensitivity= 10
    )
    print("STARTING")
    for i in range(0, 1000000): # use 4000 for testing
        #print(global_state.timestamp, global_state.mid_price)
        for tk in global_state.all_tokens:
            condition_id = token_to_condition_id[tk] # gets the market id
        try: # Needs try except because order book may not be loaded yet
            #print(all_data[condition_id])
            #print(all_data[condition_id])
            print("try")
            best_bid, best_offer = get_best_bid_offer(all_data[condition_id])
            option_mid = (best_bid + best_offer)/2
            now_et = datetime.now(ZoneInfo("America/New_York"))
            T_seconds = ((exp-now_et).total_seconds())
            T = ((exp-now_et).total_seconds())/(60*60*24*365)
            bid_vol = bs_binary_call_implied_vol_closed(best_bid, global_state.mid_price, call_strike, T, 0, 0) # missing parameters are expiry time and strike
            offer_vol = bs_binary_call_implied_vol_closed(best_offer, global_state.mid_price, call_strike, T, 0, 0)
            mid_vol = bs_binary_call_implied_vol_closed(option_mid, global_state.mid_price, call_strike, T, 0, 0)
            kf.process_tick(bid_vol, offer_vol)
            fair_vol = kf.x
            theo = bs_binary_call(global_state.mid_price, call_strike, T, 0, fair_vol, 0)
            print("BID: ", best_bid, " OFFR: ", best_offer, " THEO: ", theo, " BID VOL: ", round(bid_vol, 4), " OFFR VOL: ", round(offer_vol, 4), " fair vol: ", fair_vol, " UL: ", global_state.mid_price, " MINS TO EXP: ", T_seconds/60)
            #print(global_state.net_position)

            """
            # Trading logic v1
            bid_edge = best_bid - theo
            offer_edge = theo - best_offer

            if bid_edge > threshold:
                # Want to hit the bid, ie BUY NO
                if global_state.net_position > -max_position:
                    order_id = client.create_order(REVERSE_TOKENS[global_state.all_tokens[0]], "BUY", 1-best_bid, size_traded) # TODO: Need to make these tokens flexible eventually
                    resp = client.get_order(order_id) # get fill quantity here, then cancel order immediately
                    if (resp["status"] == "MATCHED") or (resp["status"] == "matched"): # TODO: THIS DOESNT SEEM TO ACCOUNT FOR PARTIAL FILLS, IE NEED RESPONSE TO TELL ME HOW MANY UNITS WE GOT FILLED ON
                        # TODO: SHOULD ALSO BE WILLING TO REDUCE RISK AT MIDS OR BETTER!!
                        global_state.net_position -= size_traded
                    client.cancel_orders([order_id])
                    client.cancel_orders([order_id])


            elif offer_edge > threshold:
                # Want to lift the offer
                if global_state.net_position < max_position:
                    order_id = client.create_order(global_state.all_tokens[0], "BUY", best_offer, size_traded)
                    resp = client.get_order(order_id)
                    if (resp["status"] == "MATCHED") or (resp["status"] == "matched"):
                        global_state.net_position += size_traded
                    client.cancel_orders([order_id])
                    client.cancel_orders([order_id])

            if global_state.net_position > 0:
                # IF we are long, we are willing to buy NO at any price better than value
                if best_bid > theo+.0075:
                    order_id = client.create_order(REVERSE_TOKENS[global_state.all_tokens[0]], "BUY", 1 - best_bid,
                                                   size_traded)
                    resp = client.get_order(order_id)  # get fill quantity here, then cancel order immediately
                    if (resp["status"] == "MATCHED") or (resp["status"]=="matched"):
                        global_state.net_position -= size_traded
                    client.cancel_orders([order_id])
                    client.cancel_orders([order_id])
            elif global_state.net_position < 0:
                if best_offer < theo-.0075:
                    order_id = client.create_order(global_state.all_tokens[0], "BUY", best_offer, size_traded)
                    resp = client.get_order(order_id)
                    if (resp["status"] == "MATCHED") or (resp["status"]=="matched"):
                        global_state.net_position += size_traded
                    client.cancel_orders([order_id])
                    client.cancel_orders([order_id])
                # If we are short, we are willing to buy YES at any price better than value
            """

            # TODO: Need to impute a fair value for mid vol using a kalman filter, then calc mid of option each timestep

            best_bids.append(best_bid)
            best_offers.append(best_offer)
            option_mids.append(option_mid)
            theos.append(theo)
            bid_vols.append(bid_vol)
            offer_vols.append(offer_vol)
            mid_vols.append(mid_vol)
            fair_vols.append(fair_vol)
            ul_price.append(global_state.mid_price)
            mins_exp.append(T_seconds/60)
        except:
            pass
            i+=1
            traceback.print_exc()
        time.sleep(.1)
        i += 1

    result = pd.DataFrame({"best_bid": best_bids,
                           "best_offer": best_offers,
                           "mid": option_mids,
                           "theo": theos,
                           "bid_vol": bid_vols,
                           "offer_vol": offer_vols,
                           "mid_vol": mid_vols,
                           "fair_vol": fair_vols,
                           "ul_price": ul_price,
                           "mins_exp": mins_exp})
    result.to_csv("result8.csv")

        # now we have the price... just need the best bid and offer now

async def zombie_sweeper_loop(interval_sec: float = 60.0):
    """
    Periodically sweep zombie orders using local working_orders_by_market.
    """
    while True:
        try:
            await sweep_zombie_orders_from_working()
        except Exception as e:
            print(f"[SWEEP_LOOP] Error in sweep_zombie_orders_from_working: {type(e).__name__}: {e}")
        await asyncio.sleep(interval_sec)


# TODO: Link token to market id (ie condiiton id)- making a new inverse dictionary from YES TOKEN to CONDITION ID so can then pull market order book in the all data dict
async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """
    print("starting")

    # Initialize position check lock (prevents race condition in concurrent order placement)
    global_state.position_check_lock = asyncio.Lock()

    # Initialize client
    client = PolymarketClient()
    global_state.client = client

    # Load all BTC 15-min markets from CSV
    # Run `python fetch_markets.py` to generate this file
    csv_path = "all_markets.csv"
    all_markets = load_btc_15min_markets(csv_path)
    print(f"[MAIN] Loaded {len(all_markets)} BTC 15-min markets from {csv_path}")

    # Extract all token IDs for websocket subscription (subscribe to all upcoming markets)
    all_token_ids = []
    for m in all_markets:
        all_token_ids.append(m['yes_token'])
        # Also register token mappings upfront
        global_state.condition_to_token_id[m['condition_id']] = m['yes_token']
        global_state.token_to_condition_id[m['yes_token']] = m['condition_id']
        global_state.token_to_condition_id[m['no_token']] = m['condition_id']
        global_state.REVERSE_TOKENS[m['yes_token']] = m['no_token']
        global_state.REVERSE_TOKENS[m['no_token']] = m['yes_token']

    # Initialize with first market's token (scheduler will update this)
    if all_markets:
        global_state.all_tokens = [all_markets[0]['yes_token']]
        # btc_markets should contain condition_ids (not token_ids) for fair value updates
        global_state.btc_markets.add(all_markets[0]['condition_id'])
    else:
        print("[MAIN] WARNING: No BTC 15-min markets found!")
        global_state.all_tokens = []

    # Store all token IDs for websocket subscription
    global_state.all_subscription_tokens = all_token_ids
    print(f"[MAIN] Will subscribe to {len(all_token_ids)} market tokens")

    # Initialize price blend Kalman filter with current market price
    print("[INIT] Querying Binance and Kraken APIs for initial price...")
    btcusdt = get_binance_btcusdt_mid(verbose=True)
    usdtusd = get_usdt_usd_rate(verbose=True)

    if btcusdt is not None and usdtusd is not None:
        initial_btcusd = btcusdt * usdtusd
        print(f"[INIT] Initial BTCUSD price (from Binance): {initial_btcusd:.2f}")
    else:
        initial_btcusd = 88000.0  # Fallback
        print(f"[INIT] Failed to query APIs, using fallback price: {initial_btcusd:.2f}")

    global_state.usdtusd = usdtusd if usdtusd is not None else 0.999425
    global_state.binance_mid_price = btcusdt  # Initialize with API query result

    # Correct initial estimate for known Binance bias
    # If Binance is $7 lower, and we got price from Binance, add $7 to get true estimate
    initial_btcusd_corrected = initial_btcusd - (-8.0)  # Subtract the bias
    print(f"[INIT] Bias-corrected initial price: {initial_btcusd_corrected:.2f}")

    global_state.price_blend_filter = PriceBlendKalman(
        x0=initial_btcusd_corrected,
        P0=100.0**2,
        process_var_per_sec=10.0**2,
        rtds_meas_var=2.0**2,
        binance_meas_var=14.0**2,  # Lower noise = higher weight on Binance = faster response
        bias_learning_rate=0.06,  # Faster bias adaptation (was 0.01)
        pause_bias_learning_during_movement=True
    )
    global_state.blended_price = initial_btcusd_corrected
    print(f"[INIT] Price blend Kalman filter initialized with x0={initial_btcusd_corrected:.2f}")

    #print("After initial updates: ", global_state.orders, global_state.positions)


    #client.get_market_details(token_to_condition_id[global_state.all_tokens[0]])

    #print("\n")
    #print(
     #   f'There are {len(global_state.df)} market, {len(global_state.positions)} positions and {len(global_state.orders)} orders. Starting positions: {global_state.positions}')

    # Start background update thread
    update_thread = threading.Thread(target=update_periodically, args=(global_state.client,), daemon=True)
    update_thread.start()

    # Main loop - maintain websocket connections
    print("STARTING")
    global_state.bot_start_ts = time.time()
    while True:
        try:
            # Connect to market and user websockets simultaneously
           #await asyncio.gather(
            #    connect_market_websocket(global_state.all_tokens),
             #   connect_user_websocket()
            #)
            # Select which exchange stream to use based on configuration
            price_source = getattr(global_state, 'PRICE_SOURCE', 'RTDS')
            if price_source == "COINBASE":
                # Coinbase mode: RTDS + Coinbase trigger trades
                exchange_stream = stream_coinbase_btcusd_mid(verbose=False)
            elif price_source == "RTDS":
                # RTDS mode: RTDS triggers trades, Coinbase used for z-score predictor
                exchange_stream = stream_coinbase_btcusd_mid(verbose=False)
            else:  # BLEND
                # Blend mode: RTDS + Binance trigger trades
                exchange_stream = stream_binance_btcusdt_mid(verbose=False)

            await asyncio.gather(
                connect_market_websocket(global_state.all_subscription_tokens),  # Subscribe to all markets
                stream_btc_usd(),
                exchange_stream,  # Only one exchange stream needed based on mode
                connect_user_websocket(client.creds.api_key, client.creds.api_secret, client.creds.api_passphrase),
                reconcile_loop_all(),
                markout_loop(),
                markout_dump_loop(),
                run_scheduler(csv_path, stop_before_end_seconds=60, preloaded_markets=all_markets),  # Auto-transition between markets
                health_monitor()  # Log health metrics every 30 seconds
            )
            print("Reconnecting to the websocket")
        except:
            print("Error in main loop")
            print(traceback.format_exc())

        await asyncio.sleep(1)
        gc.collect()  # Clean up memory


if __name__ == "__main__":
    print("ok")
    asyncio.run(main())
#print("hello")
#asyncio.run(main())