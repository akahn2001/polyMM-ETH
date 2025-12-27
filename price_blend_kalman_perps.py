"""
Kalman filter for blending RTDS with Binance perpetual price.

Strategy:
- RTDS provides the accurate price level (high trust)
- Binance perp provides fast updates (lower trust due to funding bias)
- Result: price level tracks RTDS closely, but responds quickly to perp movements
"""

import time


class PriceBlendKalmanPerps:
    """
    1D Kalman filter for blending RTDS + Binance perp.

    State: x = true BTC/USD price
    Process model: x_t = x_{t-1} + w_t, where w_t ~ N(0, Q*dt)
    Measurement models:
        RTDS: z = x + bias_rtds + v, v ~ N(0, R_rtds)
        Binance perp: z = x + bias_binance_perp + v, v ~ N(0, R_binance_perp)
    """

    def __init__(
        self,
        x0: float = 88000.0,
        P0: float = 100.0**2,  # Same as spot/RTDS blend
        process_var_per_sec: float = 10.0**2,
        rtds_meas_var: float = 2.0**2,  # Same as spot/RTDS blend
        binance_perp_meas_var: float = 3.5**2,  # Same as Binance spot (3x trust in RTDS)
        bias_learning_rate: float = 0.01,  # Same as spot/RTDS blend
        initial_binance_perp_bias: float = 0.0,  # Initialize with current funding-driven premium
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
            Binance perp measurement noise variance (same as Binance spot)
        bias_learning_rate : float
            How quickly to adapt bias estimates (0 = no learning, 1 = instant)
        initial_binance_perp_bias : float
            Initial bias for Binance perp (funding-driven premium)
        """
        self.x = x0  # State estimate (blended price)
        self.P = P0  # State covariance
        self.Q = process_var_per_sec  # Process noise per second

        # Measurement noise for each source
        self.R_rtds = rtds_meas_var
        self.R_binance_perp = binance_perp_meas_var

        # Track last update time for predict step
        self.last_update_time = time.time()

        # Bias tracking with exponential moving average
        self.rtds_bias = 0.0
        self.binance_perp_bias = initial_binance_perp_bias  # Initialize with current funding premium
        self.bias_alpha = bias_learning_rate  # EMA smoothing factor

        # Track observations for bias estimation
        self.rtds_observation_count = 0
        self.binance_perp_observation_count = 0

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
        # Only update bias after we have some confidence in state
        if self.rtds_observation_count > 5:
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
        # Only update bias after we have some confidence in state
        if self.binance_perp_observation_count > 5:
            residual = z_binance_perp - self.x
            self.binance_perp_bias += self.bias_alpha * (residual - self.binance_perp_bias)

        self.binance_perp_observation_count += 1
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
            "rtds_obs_count": self.rtds_observation_count,
            "binance_perp_obs_count": self.binance_perp_observation_count,
        }
