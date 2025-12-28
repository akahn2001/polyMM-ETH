"""
Kalman filter for blending Binance and RTDS BTC price streams.

State space model:
- State: true BTC/USD price
- Observations: RTDS price (slow, accurate) and Binance price (fast, noisier)
- Handles different update rates and systematic biases between sources
"""

import time


class PriceBlendKalman:
    """
    1D Kalman filter for blending multiple price sources.

    State: x = true BTC/USD price
    Process model: x_t = x_{t-1} + w_t, where w_t ~ N(0, Q*dt)
    Measurement models:
        RTDS: z = x + bias_rtds + v, v ~ N(0, R_rtds)
        Binance: z = x + bias_binance + v, v ~ N(0, R_binance)
    """

    def __init__(
        self,
        x0: float = 88000.0,
        P0: float = 100.0**2,
        process_var_per_sec: float = 10.0**2,
        rtds_meas_var: float = 2.0**2,
        binance_meas_var: float = 14.0**2,  # 3x trust in RTDS vs Binance (was 6x with 5.0**2)
        bias_learning_rate: float = 0.06, # was .01
        pause_bias_learning_during_movement: bool = True,  # Only learn bias during stable periods
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
            RTDS measurement noise variance
        binance_meas_var : float
            Binance measurement noise variance
        bias_learning_rate : float
            How quickly to adapt bias estimates (0 = no learning, 1 = instant)
        pause_bias_learning_during_movement : bool
            If True, only update bias during stable periods (>10s since last Binance change)
            Prevents bias learning from confusing lead/lag dynamics with systematic bias
        """
        self.x = x0  # State estimate (blended price)
        self.P = P0  # State covariance
        self.Q = process_var_per_sec  # Process noise per second

        # Measurement noise for each source
        self.R_rtds = rtds_meas_var
        self.R_binance = binance_meas_var

        # Track last update time for predict step
        self.last_update_time = time.time()
        self.init_time = time.time()  # Track initialization time for warmup period

        # Bias tracking with exponential moving average
        self.rtds_bias = 0.0
        self.binance_bias = -9.0  # Initialize: Binance typically $7 lower than RTDS
        self.bias_alpha = bias_learning_rate  # EMA smoothing factor (steady-state)
        self.bias_alpha_warmup = 0.5  # Very fast learning during first 10 seconds (was 0.25)
        self.warmup_duration = 10.0  # seconds
        self.warmup_complete = False  # Track if we've logged the transition
        self.pause_bias_learning_during_movement = pause_bias_learning_during_movement

        # Track observations for bias estimation
        self.rtds_observation_count = 0
        self.binance_observation_count = 0

        # Track Binance price changes for adaptive trust
        self.last_binance_value = None
        self.last_binance_change_time = time.time()

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

        # Update step with adaptive measurement variance
        # Innovation: y = z - H*x where H=1
        y = z_rtds - self.rtds_bias - self.x

        # Adaptive variance: trust RTDS LESS when Binance is recently changing
        time_since_binance_change = current_time - self.last_binance_change_time
        if time_since_binance_change < 4.0:
            # Quadratic decay: stay aggressive longer, then transition quickly
            # This allows blend to follow Binance during active movement
            normalized_time = time_since_binance_change / 4.0
            decay = normalized_time ** 2  # Quadratic decay
            trust_multiplier = 10.0 - 9.0 * decay  # 10x -> 1x
            R_adaptive = self.R_rtds * trust_multiplier
        else:
            R_adaptive = self.R_rtds  # Normal trust after 4s of Binance stability

        # Innovation covariance: S = H*P*H' + R
        S = self.P + R_adaptive

        # Kalman gain: K = P*H' / S
        K = self.P / S

        # State update: x = x + K*y
        self.x += K * y

        # Covariance update: P = (1 - K*H)*P = (1 - K)*P
        self.P *= (1.0 - K)

        # Update bias estimate (EMA of innovations)
        # After state converges, persistent innovation indicates bias
        self.rtds_observation_count += 1
        if self.rtds_observation_count > 10:  # Wait for convergence
            # Check if we should pause bias learning during active price movement
            time_since_change = current_time - self.last_binance_change_time
            should_learn_bias = not self.pause_bias_learning_during_movement or time_since_change > 4.0

            if should_learn_bias:
                # Adaptive learning rate: fast for first 10 seconds, then steady-state
                elapsed = current_time - self.init_time
                alpha = self.bias_alpha_warmup if elapsed < self.warmup_duration else self.bias_alpha

                # Debug: Log transition from warmup to steady-state (only once)
                if elapsed >= self.warmup_duration and self.rtds_observation_count == 11:
                    print(f"[KALMAN] RTDS bias learning switched from warmup ({self.bias_alpha_warmup}) to steady-state ({self.bias_alpha}) at {elapsed:.1f}s")

                # Only update bias after filter has settled and during stable periods
                residual = z_rtds - self.x
                self.rtds_bias = (1 - alpha) * self.rtds_bias + alpha * residual

        self.last_update_time = current_time

    def update_binance(self, z_binance: float):
        """
        Update filter with Binance price observation.

        Parameters
        ----------
        z_binance : float
            Binance BTC/USD price observation (already adjusted for USDT/USD)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Predict step
        self.predict(dt)

        # Track if Binance price changed (detect movement)
        if self.last_binance_value is not None and abs(z_binance - self.last_binance_value) > 0.01:
            # Price changed - update last change time
            self.last_binance_change_time = current_time
        self.last_binance_value = z_binance

        # Update step with adaptive measurement variance
        # Innovation (before Kalman update)
        y = z_binance - self.binance_bias - self.x

        # Adaptive variance: trust Binance MORE when it's recently changed
        time_since_change = current_time - self.last_binance_change_time
        if time_since_change < 4.0:
            # Quadratic decay: stay aggressive longer, then transition quickly
            normalized_time = time_since_change / 4.0
            decay = normalized_time ** 2  # Quadratic decay
            trust_multiplier = 0.1 + 0.9 * decay  # 0.1x -> 1x
            R_adaptive = self.R_binance * trust_multiplier
        else:
            R_adaptive = self.R_binance  # Normal trust after 4s of no changes

        # Kalman update with adaptive variance
        S = self.P + R_adaptive
        K = self.P / S
        self.x += K * y
        self.P *= (1.0 - K)

        # Update bias estimate (EMA of innovations)
        self.binance_observation_count += 1
        if self.binance_observation_count > 50:  # Wait longer for Binance (more observations)
            # Check if we should pause bias learning during active price movement
            time_since_change = current_time - self.last_binance_change_time
            should_learn_bias = not self.pause_bias_learning_during_movement or time_since_change > 4.0

            if should_learn_bias:
                # Adaptive learning rate: fast for first 10 seconds, then steady-state
                elapsed = current_time - self.init_time
                alpha = self.bias_alpha_warmup if elapsed < self.warmup_duration else self.bias_alpha

                # After filter settles, track systematic bias (only during stable periods)
                residual = z_binance - self.x
                self.binance_bias = (1 - alpha) * self.binance_bias + alpha * residual

        self.last_update_time = current_time

    def get_blended_price(self) -> float:
        """
        Get current blended price estimate.

        Returns
        -------
        float
            Best estimate of true BTC/USD price
        """
        return self.x

    def get_uncertainty(self) -> float:
        """
        Get current price uncertainty (standard deviation).

        Returns
        -------
        float
            Standard deviation of price estimate
        """
        return self.P ** 0.5
