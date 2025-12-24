import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

prices = pd.read_csv("btc_usdt_minute.csv")["Close"][-1000:]

print(prices)

def merton_log_likelihood(params, returns, dt):
    sigma, lam, mu_J, sigma_J = params
    if sigma <= 0 or lam < 0 or sigma_J < 0:
        return np.inf

    # Truncate Poisson sum
    n_max = 10
    logL = 0.0

    for r in returns:
        terms = []
        for n in range(n_max):
            p_n = np.exp(-lam*dt) * (lam*dt)**n / np.math.factorial(n)
            mean_n = (n*mu_J)  # risk-neutral drift dropped for historical fit
            var_n = sigma**2*dt + n*sigma_J**2
            terms.append(p_n * norm.pdf(r, mean_n, np.sqrt(var_n)))
        density = np.sum(terms)
        logL += np.log(density + 1e-12)

    return -logL  # negative LL for minimizer

def fit_merton(returns, dt=1/525600): #24/7 trading, time series by minute
    # heuristic initial guesses
    sigma0 = np.std(returns)
    lam0 = 0.1
    mu_J0 = np.mean(returns[returns < -2*sigma0]) if np.any(returns < -2*sigma0) else -0.02
    sigma_J0 = np.std(returns[returns < -2*sigma0]) if np.any(returns < -2*sigma0) else 0.1

    x0 = np.array([sigma0, lam0, mu_J0, sigma_J0])

    bounds = [(1e-6,1.0),(0,5.0),(-1.0,1.0),(1e-6,1.0)]
    res = minimize(merton_log_likelihood, x0, args=(returns,dt),
                   method="L-BFGS-B", bounds=bounds)
    return res.x

returns = np.diff(np.log(prices))
params = fit_merton(returns)
print("Estimated parameters (σ, λ, μ_J, σ_J):", params)
