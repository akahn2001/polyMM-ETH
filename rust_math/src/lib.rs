use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f64::consts::{PI, SQRT_2};

// ============================================================================
// Math helpers
// ============================================================================

/// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Standard normal PDF: phi(x) = exp(-0.5 * x^2) / sqrt(2*pi)
#[inline]
fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Error function (erf) - using approximation accurate to ~1e-7
/// Based on Abramowitz and Stegun formula 7.1.26
#[inline]
fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Inverse standard normal CDF (ppf/quantile function)
/// Uses Acklam's algorithm, accurate to ~1e-9
#[inline]
fn ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        // Lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5]) /
        ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0]*r + A[1])*r + A[2])*r + A[3])*r + A[4])*r + A[5])*q /
        (((((B[0]*r + B[1])*r + B[2])*r + B[3])*r + B[4])*r + 1.0)
    } else {
        // Upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5]) /
        ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    };

    x
}

// ============================================================================
// Black-Scholes Binary Option Functions
// ============================================================================

/// Price a Black-Scholes cash-or-nothing binary call.
///
/// Payoff at expiry: payoff * 1_{S_T > K}
///
/// # Arguments
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to maturity in years
/// * `r` - Risk-free rate (continuous compounding)
/// * `sigma` - Volatility (implied vol)
/// * `q` - Dividend/convenience yield (continuous compounding)
/// * `payoff` - Cash amount paid if in-the-money at expiry
///
/// # Returns
/// Present value of the binary call
#[pyfunction]
#[pyo3(signature = (s, k, t, r, sigma, q=0.0, payoff=1.0))]
fn bs_binary_call(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, payoff: f64) -> PyResult<f64> {
    // Immediate expiry case
    if t <= 0.0 {
        let value = (-r * t).exp() * if s > k { payoff } else { 0.0 };
        return Ok(value);
    }

    if sigma <= 0.0 {
        return Err(PyValueError::new_err("sigma must be positive"));
    }

    // Black-Scholes d2
    let d2 = ((s / k).ln() + (r - q - 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

    // Cash-or-nothing call: e^{-rT} * payoff * N(d2)
    Ok((-r * t).exp() * payoff * phi(d2))
}

/// Delta of a Black-Scholes cash-or-nothing binary call.
/// Returns dPrice / dS.
#[pyfunction]
#[pyo3(signature = (s, k, t, r, sigma, q=0.0, payoff=1.0))]
fn bs_binary_call_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, payoff: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 || s <= 0.0 {
        return 0.0;
    }

    let d2 = ((s / k).ln() + (r - q - 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

    (-r * t).exp() * payoff * pdf(d2) / (s * sigma * t.sqrt())
}

/// Closed-form implied volatility for a Black-Scholes cash-or-nothing binary call.
/// Uses analytic inversion (no root-finding loop).
///
/// Price formula: C = payoff * exp(-rT) * N(d2)
///
/// # Arguments
/// * `price` - Observed market price of the binary call
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to maturity in years
/// * `r` - Risk-free rate
/// * `q` - Dividend yield
/// * `payoff` - Cash payoff if ITM
///
/// # Returns
/// Implied volatility (sigma)
///
/// # Errors
/// Returns error if price is out of arbitrage bounds or no valid solution exists.
#[pyfunction]
#[pyo3(signature = (price, s, k, t, r, q=0.0, payoff=1.0))]
fn bs_binary_call_implied_vol_closed(price: f64, s: f64, k: f64, t: f64, r: f64, q: f64, payoff: f64) -> PyResult<f64> {
    if t <= 0.0 {
        return Err(PyValueError::new_err("T must be positive for implied vol."));
    }

    // Arbitrage bounds for a cash-or-nothing call: 0 < price < payoff * exp(-rT)
    let upper = payoff * (-r * t).exp();
    if price <= 0.0 || price >= upper {
        return Err(PyValueError::new_err(format!(
            "Price {} out of bounds for binary call (0, {}).",
            price, upper
        )));
    }

    // Step 1: get d2 from price
    let p = price / (payoff * (-r * t).exp());
    let d2 = ppf(p);

    // Step 2: solve quadratic for x = sigma * sqrt(T)
    let a = (s / k).ln() + (r - q) * t;
    let disc = d2 * d2 + 2.0 * a;  // discriminant: d2^2 + 2A

    if disc <= 0.0 {
        return Err(PyValueError::new_err("No real solution for implied vol (discriminant <= 0)."));
    }

    let sqrt_disc = disc.sqrt();

    // Quadratic: 0.5 x^2 + d2 x - A = 0
    // Solutions: x = -d2 ± sqrt(d2^2 + 2A)
    let x1 = -d2 + sqrt_disc;
    let x2 = -d2 - sqrt_disc;

    // We need a positive x = sigma * sqrt(T)
    let x = if x1 > 0.0 && x2 > 0.0 {
        x1.min(x2)
    } else if x1 > 0.0 {
        x1
    } else if x2 > 0.0 {
        x2
    } else {
        return Err(PyValueError::new_err("No positive volatility solution."));
    };

    Ok(x / t.sqrt())
}

// ============================================================================
// Python Module
// ============================================================================

#[pymodule]
fn rust_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bs_binary_call, m)?)?;
    m.add_function(wrap_pyfunction!(bs_binary_call_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bs_binary_call_implied_vol_closed, m)?)?;
    Ok(())
}
