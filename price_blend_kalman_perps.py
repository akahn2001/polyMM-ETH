"""
Kalman filter for blending RTDS with Binance and Kraken perpetual prices.

Strategy:
- RTDS provides the accurate price level (high trust)
- Binance perp and Kraken perp provide fast updates (lower trust due to funding bias)
- Result: price level tracks RTDS closely, but responds quickly to perp movements
"""

import time


class PriceBlendKalmanPerps:
    """
    1D Kalman filter for blending RTDS + Binance perp + Kraken perp.

    State: x = true BTC/USD price
    Process model: x_t = x_{t-1} + w_t, where w_t ~ N(0, Q*dt)
    Measurement models:
        RTDS: z = x + bias_rtds + v, v ~ N(0, R_rtds)
        Binance perp: z = x + bias_binance_perp + v, v ~ N(0, R_binance_perp)
        Kraken perp: z = x + bias_kraken_perp + v, v ~ N(0, R_kraken_perp)
    """

    def __init__(
        self,
        x0: float = 88000.0,
        P0: float = 500.0**2,  # High initial uncertainty for fast calibration
        process_var_per_sec: float = 10.0**2,
        rtds_meas_var: float = 2.0**2,  # High trust in RTDS for level (same as original blend)
        binance_perp_meas_var: float = 3.5**2,  # ~5x less trust than RTDS (funding bias, but fast)
        kraken_perp_meas_var: float = 4.0**2,  # ~7.5x less trust (funding bias + slower)
        bias_learning_rate: float = 0.15,  # Much faster bias learning - perps can have significant funding bias
    ):
        """
        Parameters
        ----------
        x0 : float
            Initial price estimate
        P0 : float
            Initial uncertainty variance
        process_var_per_sec : float
            Process noise variance per second (how much price can change)
        rtds_meas_var : float
            RTDS measurement noise variance (LOW = high trust for level)
        binance_perp_meas_var : float
            Binance perp measurement noise variance (HIGHER = lower trust due to funding, but faster latency than Kraken)
        kraken_perp_meas_var : float
            Kraken perp measurement noise variance (HIGHER = lower trust due to funding + slower latency)
        bias_learning_rate : float
            How quickly to adapt bias estimates (0 = no learning, 1 = instant)
        """
        self.x = x0  # State estimate (blended price)
        self.P = P0  # State covariance
        self.Q = process_var_per_sec  # Process noise per second

        # Measurement noise for each source
        self.R_rtds = rtds_meas_var
        self.R_binance_perp = binance_perp_meas_var
        self.R_kraken_perp = kraken_perp_meas_var

        # Track last update time for predict step
        self.last_update_time = time.time()

        # Bias tracking with exponential moving average
        self.rtds_bias = 0.0
        self.binance_perp_bias = 0.0  # Will learn funding-driven bias
        self.kraken_perp_bias = 0.0  # Will learn funding-driven bias
        self.bias_alpha = bias_learning_rate  # EMA smoothing factor

        # Track observations for bias estimation
        self.rtds_observation_count = 0
        self.binance_perp_observation_count = 0
        self.kraken_perp_observation_count = 0

    def predict(self, dt: float):
        """
        Prediction step: propagate state forward in time.

        Parameters
        ----------
        dt : float
            Time elapsed since last update (seconds)
        """
        # State doesn't change (random walk model)
        # x_t = x_{t-1}

        # Uncertainty increases with time
        self.P += self.Q * dt

    def update_rtds(self, z_rtds: float):
        """
        Update filter with RTDS price observation.

        Parameters
        ----------
        z_rtds : float
            RTDS BTC/USD price observation
        """
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Predict step
        self.predict(dt)

        # Update step (Kalman update)
        # Innovation: y = z - H*x where H=1
        y = z_rtds - self.rtds_bias - self.x

        # Innovation covariance: S = H*P*H' + R
        S = self.P + self.R_rtds

        # Kalman gain: K = P*H' / S
        K = self.P / S

        # State update: x = x + K*y
        self.x += K * y

        # Covariance update: P = (1-K*H)*P = (1-K)*P
        self.P *= (1 - K)

        # Update bias estimate (learn systematic offset)
        # Start learning bias early for faster calibration
        if self.rtds_observation_count > 1:
            residual = z_rtds - self.x
            self.rtds_bias += self.bias_alpha * (residual - self.rtds_bias)

        self.rtds_observation_count += 1
        self.last_update_time = current_time

    def update_binance_perp(self, z_binance_perp: float):
        """
        Update filter with Binance perp price observation.

        Parameters
        ----------
        z_binance_perp : float
            Binance perp BTC/USD price observation
        """
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Predict step
        self.predict(dt)

        # Update step (Kalman update)
        y = z_binance_perp - self.binance_perp_bias - self.x
        S = self.P + self.R_binance_perp
        K = self.P / S
        self.x += K * y
        self.P *= (1 - K)

        # Update bias estimate (learn funding-driven offset)
        # Start learning early for faster calibration
        if self.binance_perp_observation_count > 2:
            residual = z_binance_perp - self.x
            self.binance_perp_bias += self.bias_alpha * (residual - self.binance_perp_bias)

        self.binance_perp_observation_count += 1
        self.last_update_time = current_time

    def update_kraken_perp(self, z_kraken_perp: float):
        """
        Update filter with Kraken perp price observation.

        Parameters
        ----------
        z_kraken_perp : float
            Kraken perp BTC/USD price observation
        """
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Predict step
        self.predict(dt)

        # Update step (Kalman update)
        y = z_kraken_perp - self.kraken_perp_bias - self.x
        S = self.P + self.R_kraken_perp
        K = self.P / S
        self.x += K * y
        self.P *= (1 - K)

        # Update bias estimate (learn funding-driven offset)
        # Start learning early for faster calibration
        if self.kraken_perp_observation_count > 2:
            residual = z_kraken_perp - self.x
            self.kraken_perp_bias += self.bias_alpha * (residual - self.kraken_perp_bias)

        self.kraken_perp_observation_count += 1
        self.last_update_time = current_time

    def get_state(self) -> dict:
        """
        Get current filter state for debugging/monitoring.

        Returns
        -------
        dict
            Dictionary containing state estimate, uncertainty, and biases
        """
        return {
            "price": self.x,
            "uncertainty": self.P**0.5,
            "rtds_bias": self.rtds_bias,
            "binance_perp_bias": self.binance_perp_bias,
            "kraken_perp_bias": self.kraken_perp_bias,
            "rtds_obs_count": self.rtds_observation_count,
            "binance_perp_obs_count": self.binance_perp_observation_count,
            "kraken_perp_obs_count": self.kraken_perp_observation_count,
        }
