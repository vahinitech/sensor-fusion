"""
Sensor Data Preprocessor for Character Recognition
Handles padding, normalization, and channel arrangement
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import deque


class SensorDataPreprocessor:
    """Preprocess sensor data for character recognition model"""

    SENSOR_CHANNELS = {
        "front_accel": 3,
        "gyro": 3,
        "rear_accel": 3,
        "magnet": 3,
        "force": 3,  # FIXED: Force has X, Y, Z values (not just 1)
    }

    CHANNEL_ORDER = ["front_accel", "gyro", "rear_accel", "magnet", "force"]
    TOTAL_CHANNELS = sum(SENSOR_CHANNELS.values())  # 16 (was 13 before force fix)

    def __init__(self, max_timesteps=512, sampling_rate=208):
        """
        Initialize preprocessor

        Args:
            max_timesteps: Maximum sequence length (at 208Hz = ~2.5 seconds)
            sampling_rate: Sensor sampling rate in Hz (updated to 208Hz)
        """
        self.max_timesteps = max_timesteps
        self.sampling_rate = sampling_rate
        self.robust_scaler = None
        self.std_scaler = None

    def pad_sequence(self, data, max_len=None):
        """
        Pad sequence to fixed length

        Args:
            data: Input sequence (T, C)
            max_len: Target length

        Returns:
            Padded sequence (max_len, C)
        """
        if max_len is None:
            max_len = self.max_timesteps

        data = np.asarray(data, dtype=np.float32)

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        current_len = data.shape[0]
        n_channels = data.shape[1]

        if current_len >= max_len:
            # Truncate if too long
            return data[:max_len, :]
        else:
            # Pad with zeros if too short
            padded = np.zeros((max_len, n_channels), dtype=np.float32)
            padded[:current_len, :] = data
            return padded

    def normalize_data(self, X_train, X_test=None, fit=True):
        """
        Normalize data using RobustScaler + StandardScaler

        Args:
            X_train: Training data (N, T, C)
            X_test: Test data (N, T, C)
            fit: Whether to fit scalers on training data

        Returns:
            Normalized train/test data
        """
        N_train, T, C = X_train.shape

        # Reshape for scaling
        X_train_flat = X_train.reshape(N_train, T * C)

        if fit:
            # Fit on training data
            self.robust_scaler = RobustScaler()
            X_train_robust = self.robust_scaler.fit_transform(X_train_flat)

            self.std_scaler = StandardScaler()
            X_train_scaled = self.std_scaler.fit_transform(X_train_robust)
        else:
            # Use fitted scalers
            X_train_robust = self.robust_scaler.transform(X_train_flat)
            X_train_scaled = self.std_scaler.transform(X_train_robust)

        X_train_scaled = X_train_scaled.reshape(N_train, T, C)

        # Scale test data if provided
        if X_test is not None:
            N_test = X_test.shape[0]
            X_test_flat = X_test.reshape(N_test, T * C)
            X_test_robust = self.robust_scaler.transform(X_test_flat)
            X_test_scaled = self.std_scaler.transform(X_test_robust)
            X_test_scaled = X_test_scaled.reshape(N_test, T, C)

            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def arrange_channels(self, top_accel, gyro, rear_accel, magnet, force):
        """
        Arrange sensor data into proper channel order

        Args:
            top_accel: Front accelerometer (T, 3)
            gyro: Gyroscope (T, 3)
            rear_accel: Rear accelerometer (T, 3)
            magnet: Magnetometer (T, 3)
            force: Force sensor (T, 3) - X, Y, Z from firmware

        Returns:
            Arranged data (T, 16)
        """
        T = top_accel.shape[0]
        arranged = np.zeros((T, 16), dtype=np.float32)

        arranged[:, 0:3] = top_accel
        arranged[:, 3:6] = gyro
        arranged[:, 6:9] = rear_accel
        arranged[:, 9:12] = magnet
        arranged[:, 12:16] = force

        return arranged

    def process_raw_buffers(self, sensor_buffers):
        """
        Process raw sensor buffers into model input

        Args:
            sensor_buffers: Dictionary with sensor data deques/lists

        Returns:
            Processed array (T, 16)
        """
        # Extract data
        top_accel = np.column_stack(
            [
                sensor_buffers.get("top_accel_x", []),
                sensor_buffers.get("top_accel_y", []),
                sensor_buffers.get("top_accel_z", []),
            ]
        )

        gyro = np.column_stack(
            [
                sensor_buffers.get("top_gyro_x", []),
                sensor_buffers.get("top_gyro_y", []),
                sensor_buffers.get("top_gyro_z", []),
            ]
        )

        rear_accel = np.column_stack(
            [
                sensor_buffers.get("rear_accel_x", []),
                sensor_buffers.get("rear_accel_y", []),
                sensor_buffers.get("rear_accel_z", []),
            ]
        )

        magnet = np.column_stack(
            [
                sensor_buffers.get("mag_x", []),
                sensor_buffers.get("mag_y", []),
                sensor_buffers.get("mag_z", []),
            ]
        )

        # FIXED: Force now has 3 channels (X, Y, Z) from firmware
        force = np.column_stack(
            [
                sensor_buffers.get("force_x", []),
                sensor_buffers.get("force_y", []),
                sensor_buffers.get("force_z", []),
            ]
        )

        # Arrange channels
        data = self.arrange_channels(top_accel, gyro, rear_accel, magnet, force)

        # Pad to fixed length
        data = self.pad_sequence(data)

        # Normalize
        if self.std_scaler is not None:
            data_flat = data.reshape(1, -1)
            data_robust = self.robust_scaler.transform(data_flat)
            data_scaled = self.std_scaler.transform(data_robust)
            data = data_scaled.reshape(self.max_timesteps, 13)

        return data

    def batch_process(self, data_list, normalize=True):
        """
        Process batch of data

        Args:
            data_list: List of raw data arrays/tuples
            normalize: Whether to normalize

        Returns:
            Batch array (N, max_timesteps, 13)
        """
        processed = []

        for data in data_list:
            if isinstance(data, tuple):
                # Assume tuple format: (accel, gyro, rear, mag, force)
                arranged = self.arrange_channels(*data)
            else:
                arranged = np.asarray(data, dtype=np.float32)

            padded = self.pad_sequence(arranged)
            processed.append(padded)

        batch = np.array(processed, dtype=np.float32)

        if normalize and self.std_scaler is not None:
            N, T, C = batch.shape
            batch_flat = batch.reshape(N, T * C)
            batch_robust = self.robust_scaler.transform(batch_flat)
            batch_scaled = self.std_scaler.transform(batch_robust)
            batch = batch_scaled.reshape(N, T, C)

        return batch
