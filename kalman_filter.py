import math
from scipy.stats import norm


class VolKalman1D:
    """
    1D Kalman filter for tracking a 'fair' implied vol from bid/ask vols.

    State:
        x_t = true fair vol

    Model:
        x_t = x_{t-1} + w_t        (process noise ~ N(0, Q))
        y_t = x_t + v_t            (measurement noise ~ N(0, R_t))

    R_t can be constant or a function of bid/ask vol spread.
    """

    def __init__(
        self,
        x0: float = 0.35,             # initial vol estimate
        P0: float = 0.05**2,          # initial variance of estimate
        process_var: float = 0.01**2, # Q: process noise variance
        meas_var: float = 0.01**2,    # baseline R
        use_spread_weighting: bool = True,
        spread_sensitivity: float = 1.0,
    ):
        self.x = x0          # current fair vol estimate
        self.P = P0          # state variance
        self.Q = process_var
        self.meas_var = meas_var
        self.use_spread_weighting = use_spread_weighting
        self.spread_sensitivity = spread_sensitivity

    def _meas_var_from_spread(self, spread_vol: float) -> float:
        """
        Map bid/ask vol spread to measurement variance R_t.

        If use_spread_weighting is False, just returns self.meas_var.

        Otherwise:
            R_t = meas_var * (1 + spread_sensitivity * spread_vol^2)
        """
        if not self.use_spread_weighting:
            return self.meas_var

        spread_vol = max(spread_vol, 0.0)
        return self.meas_var * (1.0 + self.spread_sensitivity * spread_vol**2)

    def process_tick(self, bid_vol: float, ask_vol: float) -> float:
        """
        Ingest a new bid/ask vol quote (even if crossed), update the fair vol
        estimate, and return the updated fair vol.

        Parameters
        ----------
        bid_vol : float
            Bid implied vol.
        ask_vol : float
            Ask implied vol.

        Returns
        -------
        float
            Updated fair vol estimate.
        """
        # 1) Minimal sanity: vols must be positive
        if bid_vol <= 0 or ask_vol <= 0:
            return self.x

        # Allow crossed markets: mid and spread still make sense
        mid_vol = 0.5 * (bid_vol + ask_vol)
        spread_vol = abs(ask_vol - bid_vol)

        # 2) Measurement variance (possibly spread-dependent)
        R_t = self._meas_var_from_spread(spread_vol)

        # 3) Kalman predict step
        x_pred = self.x
        P_pred = self.P + self.Q

        # 4) Kalman update step
        K = P_pred / (P_pred + R_t)
        self.x = x_pred + K * (mid_vol - x_pred)
        self.P = (1.0 - K) * P_pred

        return self.x


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