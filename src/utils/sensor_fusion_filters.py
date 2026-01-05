"""
Sensor Fusion Filters
Implements Kalman Filter (KF) and Extended Kalman Filter (EKF) for IMU noise reduction
"""

import numpy as np
from typing import Dict, Tuple, Optional


class KalmanFilter:
    """
    Linear Kalman Filter for sensor data denoising
    Removes noise and drift from accelerometer and gyroscope data

    Algorithm:
    1. Prediction phase: x_pred = A * x + u
    2. Update phase: x_updated = x_pred + K * (z - H * x_pred)
    where K is the Kalman gain
    """

    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        Initialize Kalman Filter

        Args:
            process_variance: Q - Process noise covariance (how much we trust the model)
            measurement_variance: R - Measurement noise covariance (how much we trust the sensor)
        """
        self.Q = process_variance  # Process noise
        self.R = measurement_variance  # Measurement noise

        # State estimate and covariance
        self.x = 0.0  # State estimate
        self.P = 1.0  # Estimate error covariance
        self.initialized = False

    def update(self, z: float) -> float:
        """
        Single step Kalman filter update

        Args:
            z: Measurement (sensor reading)

        Returns:
            Filtered sensor reading
        """
        if not self.initialized:
            self.x = z
            self.initialized = True
            return z

        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update step - calculate Kalman gain
        K = P_pred / (P_pred + self.R)

        # Update state estimate
        self.x = x_pred + K * (z - x_pred)

        # Update error covariance
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):
        """Reset filter state"""
        self.x = 0.0
        self.P = 1.0
        self.initialized = False


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for IMU fusion
    Handles nonlinear sensor models by linearizing around current estimate

    Algorithm:
    1. Prediction: x_pred = f(x, u) - Nonlinear state propagation
    2. Linearization: F = ∂f/∂x at x_pred
    3. Update: K = P_pred * H^T / (H * P_pred * H^T + R)
    """

    def __init__(
        self, process_variance: np.ndarray = None, measurement_variance: np.ndarray = None, dt: float = 0.0048
    ):  # 208Hz sampling (1/208 = 0.0048s)
        """
        Initialize Extended Kalman Filter

        Args:
            process_variance: Q - Process noise covariance matrix (3x3 for 3D)
            measurement_variance: R - Measurement noise covariance (3x3)
            dt: Time step between measurements (seconds)
        """
        self.dt = dt

        # Default 3D covariances
        if process_variance is None:
            self.Q = np.eye(3) * 0.001  # Small process noise
        else:
            self.Q = process_variance

        if measurement_variance is None:
            self.R = np.eye(3) * 0.1  # Measurement noise
        else:
            self.R = measurement_variance

        # State: [x, y, z] accelerations or angular velocities
        self.x = np.zeros(3)

        # Error covariance
        self.P = np.eye(3)

        # Velocity for bias tracking
        self.velocity = np.zeros(3)
        self.bias = np.zeros(3)
        self.initialized = False

    def _state_transition_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of state transition function
        For constant acceleration model: F = I (identity)

        Args:
            x: Current state

        Returns:
            Jacobian matrix (3x3)
        """
        return np.eye(3)

    def _measurement_matrix(self) -> np.ndarray:
        """
        Measurement matrix H (maps state to measurement)
        For direct measurement: H = I

        Returns:
            Measurement matrix (3x3)
        """
        return np.eye(3)

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Single step Extended Kalman Filter update

        Args:
            z: Measurement vector [x, y, z]

        Returns:
            Filtered measurement vector
        """
        z = np.array(z).flatten()

        if not self.initialized:
            self.x = z
            self.velocity = np.zeros(3)
            self.bias = np.zeros(3)
            self.initialized = True
            return z

        # Prediction step
        # x_pred = f(x) for constant velocity model
        x_pred = self.x

        # Compute Jacobian (state transition matrix)
        F = self._state_transition_matrix(x_pred)

        # Predict error covariance
        P_pred = F @ self.P @ F.T + self.Q

        # Update step
        H = self._measurement_matrix()

        # Innovation (measurement residual)
        y = z - H @ x_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = x_pred + K @ y

        # Update error covariance
        self.P = (np.eye(3) - K @ H) @ P_pred

        # Track velocity and bias for diagnostics
        self.velocity = self.velocity + self.x * self.dt

        return self.x

    def reset(self):
        """Reset filter state"""
        self.x = np.zeros(3)
        self.velocity = np.zeros(3)
        self.bias = np.zeros(3)
        self.P = np.eye(3)
        self.initialized = False


class ComplementaryFilter:
    """
    Complementary Filter for sensor fusion
    Fuses low-frequency (accurate) and high-frequency (noisy) data

    Algorithm:
    output = α * high_freq + (1-α) * low_freq
    """

    def __init__(self, alpha: float = 0.95):
        """
        Initialize Complementary Filter

        Args:
            alpha: Blending factor (0.95 = 95% high-freq, 5% low-freq)
        """
        self.alpha = alpha
        self.prev_output = None

    def update(self, high_freq: np.ndarray, low_freq: np.ndarray) -> np.ndarray:
        """
        Fuse high-frequency and low-frequency data

        Args:
            high_freq: Noisy but responsive data (e.g., accelerometer)
            low_freq: Smooth but delayed data (e.g., low-pass filtered)

        Returns:
            Fused output
        """
        high_freq = np.array(high_freq).flatten()
        low_freq = np.array(low_freq).flatten()

        output = self.alpha * high_freq + (1 - self.alpha) * low_freq
        self.prev_output = output

        return output

    def reset(self):
        """Reset filter state"""
        self.prev_output = None


class NoiseFilter:
    """
    Collection of signal processing filters for noise removal
    """

    @staticmethod
    def moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Moving average filter

        Args:
            data: Input signal
            window_size: Number of samples to average

        Returns:
            Filtered signal
        """
        if len(data) < window_size:
            return data

        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(data, kernel, mode="same")
        return filtered

    @staticmethod
    def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Exponential smoothing filter

        Args:
            data: Input signal
            alpha: Smoothing factor (0-1)

        Returns:
            Filtered signal
        """
        filtered = np.zeros_like(data)
        filtered[0] = data[0]

        for i in range(1, len(data)):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]

        return filtered

    @staticmethod
    def remove_bias(data: np.ndarray) -> np.ndarray:
        """
        Remove DC bias from signal

        Args:
            data: Input signal

        Returns:
            Zero-mean signal
        """
        return data - np.mean(data)


class SensorFusionManager:
    """
    Manages multiple filters for multi-sensor fusion
    Provides unified interface for different filter types
    """

    def __init__(self, filter_type: str = "kf"):
        """
        Initialize Sensor Fusion Manager

        Args:
            filter_type: 'kf' (Kalman), 'ekf' (Extended Kalman), 'cf' (Complementary)
        """
        self.filter_type = filter_type
        self.filters = {}
        self._init_filters()

    def _init_filters(self):
        """Initialize 3D filters for each axis pair"""
        if self.filter_type == "kf":
            # Create KF for each axis
            self.filters = {"accel": [KalmanFilter() for _ in range(3)], "gyro": [KalmanFilter() for _ in range(3)]}
        elif self.filter_type == "ekf":
            self.filters = {"accel": ExtendedKalmanFilter(), "gyro": ExtendedKalmanFilter()}
        elif self.filter_type == "cf":
            self.filters = {"accel": ComplementaryFilter(), "gyro": ComplementaryFilter()}

    def filter_accel(self, accel: Dict[str, float]) -> Dict[str, float]:
        """
        Filter accelerometer data

        Args:
            accel: {'x': val, 'y': val, 'z': val}

        Returns:
            Filtered accelerometer data
        """
        if self.filter_type == "kf":
            return {
                "x": self.filters["accel"][0].update(accel["x"]),
                "y": self.filters["accel"][1].update(accel["y"]),
                "z": self.filters["accel"][2].update(accel["z"]),
            }
        elif self.filter_type == "ekf":
            filtered = self.filters["accel"].update([accel["x"], accel["y"], accel["z"]])
            return {"x": filtered[0], "y": filtered[1], "z": filtered[2]}
        elif self.filter_type == "cf":
            # For complementary, we'd need low-freq reference
            return accel

    def filter_gyro(self, gyro: Dict[str, float]) -> Dict[str, float]:
        """
        Filter gyroscope data

        Args:
            gyro: {'x': val, 'y': val, 'z': val}

        Returns:
            Filtered gyroscope data
        """
        if self.filter_type == "kf":
            return {
                "x": self.filters["gyro"][0].update(gyro["x"]),
                "y": self.filters["gyro"][1].update(gyro["y"]),
                "z": self.filters["gyro"][2].update(gyro["z"]),
            }
        elif self.filter_type == "ekf":
            filtered = self.filters["gyro"].update([gyro["x"], gyro["y"], gyro["z"]])
            return {"x": filtered[0], "y": filtered[1], "z": filtered[2]}
        elif self.filter_type == "cf":
            return gyro

    def reset(self):
        """Reset all filters"""
        for sensor_filters in self.filters.values():
            if isinstance(sensor_filters, list):
                for f in sensor_filters:
                    f.reset()
            else:
                sensor_filters.reset()

    def set_filter_type(self, filter_type: str):
        """Switch to different filter type"""
        self.filter_type = filter_type
        self._init_filters()
