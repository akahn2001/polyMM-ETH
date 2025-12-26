import math
import global_state
from scipy.stats import norm
from kalman_filter import VolKalman1D
from datetime import datetime
from zoneinfo import ZoneInfo

# TODO: Do this to derive option strike
"""
Example: derive price_to_beat via RTDS + market slug

Assumptions:

You already know the market’s slug, e.g. btc-updown-15m-1765134000

The trailing number is the end timestamp (seconds) of the 15m window → start = end − 900

You run a background RTDS listener to record Chainlink BTC/USD prices

Then call get_price_to_beat(start_ts) to get the price as of the start of the interval
"""

# Miscellaneous helper functions

def get_best_bid_offer(d):
    # Takes sortedDict dictionary in all data and returns best bid/offer market
    if not d:
        best_bid = 0
    else:
        best_bid = max(d["bids"].keys())
    if not d:
        best_ask = 1
    else:
        offers = [1-x for x in d["asks"].keys()]
        best_ask = min(d["asks"].keys())
    return best_bid, best_ask

def update_fair_vol_for_market(market_id):
    book = global_state.all_data.get(market_id)
    if not book:
        return

    bids = book["bids"]
    asks = book["asks"]
    if not bids or not asks:
        return

    best_bid_px = bids.peekitem(-1)[0]  # highest bid price
    best_ask_px = asks.peekitem(0)[0]   # lowest ask price

    # Get current BTC spot (use blended price for consistency)
    S = global_state.blended_price
    if S is None:
        return  # no spot yet

    now_et = datetime.now(ZoneInfo("America/New_York"))
    T_seconds = ((global_state.exp - now_et).total_seconds())
    T = ((global_state.exp - now_et).total_seconds()) / (60 * 60 * 24 * 365)
    K = global_state.strike

    try:
        bid_vol = bs_binary_call_implied_vol_closed(best_bid_px, S, K, T, 0)
        ask_vol = bs_binary_call_implied_vol_closed(best_ask_px, S, K, T, 0)  # Fixed: was using bid_px
    except Exception:
        # ATM calibration often fails - use default vol to keep system running
        bid_vol = 0.35
        ask_vol = 0.35

    # Initialize filter if needed
    if market_id not in global_state.vol_filters:
        mid_vol0 = 0.5 * (bid_vol + ask_vol) # TODO: check if this is better off set to 35
        global_state.vol_filters[market_id] = VolKalman1D(
            x0=mid_vol0,
            P0=0.08**2,
            process_var=0.003**2,
            meas_var=0.10**2,
            spread_sensitivity=10.0
        )

    kf = global_state.vol_filters[market_id]
    fair_vol = kf.process_tick(bid_vol, ask_vol)
    global_state.fair_vol[market_id] = fair_vol
    #print("FAIR VOL: ", fair_vol)

def update_fair_value_for_market(market_id: str):
    """
    Recompute the fair value (theo) of a Polymarket binary for this market_id,
    using blended BTC spot and current fair vol from the Kalman filter.
    """
    # 1) Inputs - use blended price for consistency with vol calibration
    S = global_state.blended_price
    sigma = global_state.fair_vol.get(market_id)

    if S is None or sigma is None:
        return  # nothing to do yet

    # You need to define these mappings somewhere:
    K = global_state.strike       # strike for this market
    now_et = datetime.now(ZoneInfo("America/New_York"))
    T_seconds = ((global_state.exp - now_et).total_seconds())
    T = ((global_state.exp - now_et).total_seconds()) / (60 * 60 * 24 * 365)
    r = 0                        # or per-market if needed
    q = 0.0
    payoff = 1.0

    # 2) Fair price under BS (or Merton, etc.)
    fair_price = bs_binary_call(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        payoff=payoff,
    )

    call_delta = bs_binary_call_delta(S=S, K=K, T=T, r=r, sigma=sigma, q=q, payoff=payoff)

    # 3) Optional: convert to fair probability (risk-neutral)
    # C = e^{-rT} * payoff * N(d2) => p_fair = C / (payoff * e^{-rT})
    #discount = math.exp(-r * T)
    #fair_prob = fair_price / (payoff * discount) if discount > 0 else None

    # 4) Store
    if not hasattr(global_state, "fair_value"):
        global_state.fair_value = {}

    if not hasattr(global_state, "binary_delta"):
        global_state.binary_delta = {}

    global_state.fair_value[market_id] = fair_price
    global_state.binary_delta[market_id] = call_delta
    #print(global_state.fair_value)

def update_binance_fair_value_for_market(market_id: str, binance_spot_usd: float):
    """
    Recompute Binance-based fair value (theo) for this market_id,
    using Binance BTC spot (adjusted for USDT/USD) and current fair vol from the Kalman filter.

    This allows us to calculate a separate theo based on Binance prices
    to potentially get edge from the lag between Binance and RTDS.

    Parameters
    ----------
    market_id : str
        Market identifier
    binance_spot_usd : float
        Binance BTCUSDT price adjusted to BTCUSD (multiplied by usdtusd rate)
    """
    # 1) Inputs
    S = binance_spot_usd
    sigma = global_state.fair_vol.get(market_id)

    if S is None or sigma is None:
        return  # nothing to do yet

    # Get market parameters from global state
    K = global_state.strike
    now_et = datetime.now(ZoneInfo("America/New_York"))
    T_seconds = ((global_state.exp - now_et).total_seconds())
    T = ((global_state.exp - now_et).total_seconds()) / (60 * 60 * 24 * 365)
    r = 0
    q = 0.0
    payoff = 1.0

    # 2) Fair price under BS
    fair_price = bs_binary_call(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        payoff=payoff,
    )

    call_delta = bs_binary_call_delta(S=S, K=K, T=T, r=r, sigma=sigma, q=q, payoff=payoff)

    # 3) Store in separate Binance fair value dictionary
    if not hasattr(global_state, "binance_fair_value"):
        global_state.binance_fair_value = {}

    if not hasattr(global_state, "binance_binary_delta"):
        global_state.binance_binary_delta = {}

    global_state.binance_fair_value[market_id] = fair_price
    global_state.binance_binary_delta[market_id] = call_delta

import math

def bs_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    payoff: float = 1.0,
) -> float:
    """
    Price a Black–Scholes cash-or-nothing binary call.

    Payoff at expiry:
        payoff * 1_{S_T > K}

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (cont. comp.).
    sigma : float
        Volatility (Black–Scholes implied vol).
    q : float, default 0.0
        Dividend/convenience yield (cont. comp.).
    payoff : float, default 1.0
        Cash amount paid if in-the-money at expiry.

    Returns
    -------
    float
        Present value of the binary call.
    """

    # Immediate expiry case
    if T <= 0:
        return math.exp(-r * T) * (payoff if S > K else 0.0)

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Standard normal CDF
    def phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Black–Scholes d2
    d2 = (
        math.log(S / K)
        + (r - q - 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    # Cash-or-nothing call: e^{-rT} * payoff * N(d2)
    return math.exp(-r * T) * payoff * phi(d2)

def bs_binary_call_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    payoff: float = 1.0,
) -> float:
    """
    Delta of a Black–Scholes cash-or-nothing binary call.

    Returns dPrice / dS.
    """

    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    # Standard normal PDF
    def pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    d2 = (
        math.log(S / K)
        + (r - q - 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    return (
        math.exp(-r * T)
        * payoff
        * pdf(d2)
        / (S * sigma * math.sqrt(T))
    )

def bs_binary_call_implied_vol_closed(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    payoff: float = 1.0,
) -> float:
    """
    Closed-form implied volatility for a Black–Scholes cash-or-nothing
    binary call, using analytic inversion (no root-finding loop).

    Price formula:
        C = payoff * exp(-rT) * N(d2)

    Parameters
    ----------
    price : float
        Observed market price of the binary call.
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity (years).
    r : float
        Risk-free rate (cont. comp.).
    q : float, default 0.0
        Dividend / convenience yield (cont. comp.).
    payoff : float, default 1.0
        Cash payoff if in-the-money at expiry.

    Returns
    -------
    float
        Implied volatility (sigma).

    Raises
    ------
    ValueError
        If price is out of arbitrage bounds or discriminant is invalid.
    """

    if T <= 0:
        raise ValueError("T must be positive for implied vol.")

    # Arbitrage bounds for a cash-or-nothing call:
    # 0 < price < payoff * exp(-rT)
    upper = payoff * math.exp(-r * T)
    if not (0.0 < price < upper):
        raise ValueError(
            f"Price {price} out of bounds for binary call (0, {upper})."
        )

    # Step 1: get d2 from price
    p = price / (payoff * math.exp(-r * T))  # this should be in (0,1)
    d2 = norm.ppf(p)

    # Step 2: solve quadratic for x = sigma * sqrt(T)
    A = math.log(S / K) + (r - q) * T
    disc = d2 * d2 + 2.0 * A  # discriminant: d2^2 + 2A

    if disc <= 0:
        raise ValueError("No real solution for implied vol (discriminant <= 0).")

    sqrt_disc = math.sqrt(disc)

    # Quadratic: 0.5 x^2 + d2 x - A = 0
    # Solutions: x = -d2 ± sqrt(d2^2 + 2A)
    x1 = -d2 + sqrt_disc
    x2 = -d2 - sqrt_disc

    # We need a positive x = sigma * sqrt(T)
    candidates = [x for x in (x1, x2) if x > 0]
    if not candidates:
        raise ValueError("No positive volatility solution.")

    x = min(candidates)  # usually the '+' root, but be safe
    sigma = x / math.sqrt(T)
    return sigma