import math
from typing import Literal

def binary_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Price a Black-Scholes cash-or-nothing (digital) option.

    Payoff at expiry:
      Call: 1 if S_T > K, else 0
      Put:  1 if S_T < K, else 0

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to maturity in years
    r : risk-free rate (continuously compounded)
    sigma : implied volatility (Black-Scholes vol)
    q : continuous dividend yield / convenience yield (default: 0)
    option_type : "call" or "put"

    Returns
    -------
    price : present value of the binary option
    """
    if T <= 0:
        # Immediate expiry: option is either 0 or discounted payoff if strictly ITM
        if option_type == "call":
            return math.exp(-r * T) * (1.0 if S > K else 0.0)
        elif option_type == "put":
            return math.exp(-r * T) * (1.0 if S < K else 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # d2 in Black-Scholes
    d2 = (
        math.log(S / K)
        + (r - q - 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    # standard normal CDF
    def phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if option_type == "call":
        # cash-or-nothing call: e^{-rT} * N(d2)
        return math.exp(-r * T) * phi(d2)
    elif option_type == "put":
        # cash-or-nothing put: e^{-rT} * N(-d2)
        return math.exp(-r * T) * phi(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

import math
from typing import Literal

def merton_binary_option_price_series(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    option_type: Literal["call", "put"] = "call",
    payoff: float = 1.0,
    n_max: int = 50,
    tol: float = 1e-12,
) -> float:
    """
    Cash-or-nothing binary option price under Merton Jump-Diffusion
    using the Poisson mixture of Black–Scholes digitals.

    Model:
      dS_t / S_t = (r - q - λ k) dt + σ dW_t + (J - 1) dN_t
      J = exp(Y), Y ~ N(mu_J, sigma_J^2)
      k = E[J - 1] = exp(mu_J + 0.5 * sigma_J^2) - 1

    Parameters
    ----------
    S0 : float
        Spot price at time 0.
    K : float
        Strike.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Dividend / convenience yield (cont. comp.).
    sigma : float
        Diffusion (Brownian) volatility.
    lam : float
        Jump intensity λ (Poisson rate).
    mu_J : float
        Mean of log jump size Y.
    sigma_J : float
        Vol of log jump size Y.
    option_type : {"call", "put"}
        Binary call or put.
    payoff : float
        Cash payoff if in the money.
    n_max : int
        Max number of jumps to include in the Poisson sum.
    tol : float
        Early-stop tolerance on Poisson tail contribution.

    Returns
    -------
    price : float
        Present value of the binary option.
    """
    if T <= 0:
        if option_type == "call":
            return math.exp(-r * T) * (payoff if S0 > K else 0.0)
        elif option_type == "put":
            return math.exp(-r * T) * (payoff if S0 < K else 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if sigma_J < 0:
        raise ValueError("sigma_J must be non-negative")

    def phi(x: float) -> float:
        # Standard normal CDF
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Jump-drift adjustment
    k = math.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # Poisson base weight p_0
    lamT = lam * T
    poisson_p = math.exp(-lamT)

    res = 0.0
    logS0 = math.log(S0)
    logK = math.log(K)

    for n in range(n_max + 1):
        # Conditional log S_T ~ N(m_n, v_n)
        v_n = sigma**2 * T + n * sigma_J**2
        sqrt_v_n = math.sqrt(v_n) if v_n > 0 else 1e-16  # guard

        m_n = (
            logS0
            + (r - q - lam * k - 0.5 * sigma**2) * T
            + n * mu_J
        )

        d2_n = (m_n - logK) / sqrt_v_n

        if option_type == "call":
            prob_ITM_n = phi(d2_n)
        elif option_type == "put":
            prob_ITM_n = phi(-d2_n)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        contrib = poisson_p * prob_ITM_n
        res += contrib

        # Update Poisson weight for next n: p_{n+1} = p_n * lamT / (n+1)
        if n < n_max:
            poisson_p = poisson_p * lamT / (n + 1)

        # Optional early break if Poisson tail is negligible
        if contrib < tol and n > lamT + 5 * math.sqrt(lamT + 1e-12):
            break

    price = math.exp(-r * T) * payoff * res
    return price



S = 89813.99     # spot
K = 89901.57     # strike
T = 1.03333/525600     # 525600 minutes per year
r = 0   # 3% risk-free
q = 0.0     # no dividend
sigma = 0.27  # 25% implied vol

call_bin = binary_option_price(S, K, T, r, sigma, q, "call")
put_bin  = binary_option_price(S, K, T, r, sigma, q, "put")

print(call_bin, put_bin)


from datetime import datetime
import pytz
from zoneinfo import ZoneInfo

et = pytz.timezone("America/New_York")
et_time = datetime.now(et)
print(et_time)

dt_et = datetime(2025, 12, 6, 11, 45, tzinfo=ZoneInfo("America/New_York"))


#Estimated parameters (σ, λ, μ_J, σ_J): [ 5.27353544e-01  5.00000000e+00 -5.14390451e-04  4.46197484e-03]


# Merton jump model pricer

# --- Market / model parameters ---
S = 89863.99     # spot
K = 89901.57     # strike
T = 1.03333/525600     # 525600 minutes per year
r = 0   # 3% risk-free
q = 0.0     # no dividend
sigma = 0.27  # 25% implied vol

# --- Jump parameters (Merton) ---
lam     = 5   # 0.5 jumps per year
mu_J    = 5.14390451e-04  # mean log jump (downward)
sigma_J = 4.46197484e-03   # log jump vol

# --- Merton digital call ---
merton_call = merton_binary_option_price_series(
    S, K, T, r, q, sigma,
    lam, mu_J, sigma_J,
    option_type="call",
    payoff=1.0,
)

print("Merton digital call:", merton_call)

# --- Compare to Black–Scholes digital (λ = 0) ---
bs_call = merton_binary_option_price_series(
    S, K, T, r, q, sigma,
    lam=0.0,       # no jumps
    mu_J=0.0,
    sigma_J=0.0,
    option_type="call",
    payoff=1.0,
)

print("Black–Scholes digital call (via λ=0):", bs_call)

# maybe we can just use basic black scholes

# polymarket API stream in market details !!
