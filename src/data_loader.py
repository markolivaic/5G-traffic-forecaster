"""Data generation and preprocessing module for LSTM training.

This module handles synthetic 5G RAN traffic generation and transforms
time-series data into sequences suitable for LSTM neural network training.
The data generation simulates realistic network patterns including daily
seasonality, stochastic noise, trends, and anomalous events to ensure
model robustness in production scenarios.
"""

import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import SCALER_PATH


class DataLoader:
    """Handles synthetic data generation and sequence preparation for LSTM.

    The DataLoader class generates realistic 5G network traffic patterns
    and transforms them into supervised learning sequences. It maintains
    a MinMaxScaler instance for normalizing data to [0, 1] range, which
    is critical for LSTM convergence and numerical stability.
    """

    def __init__(self):
        """Initialize DataLoader with a MinMaxScaler instance."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def generate_ran_traffic(self, days):
        """Generate synthetic network throughput data with realistic patterns.

        Simulates 5G RAN (Radio Access Network) traffic with three key
        components: daily seasonality (sinusoidal pattern), stochastic
        network noise, and a linear growth trend. Additionally injects
        anomalies to represent real-world events such as mass gatherings
        or network outages. This approach ensures the model learns to
        handle both regular patterns and edge cases.

        Args:
            days (int): Number of days of data to generate.

        Returns:
            pd.DataFrame: DataFrame with 'timestamp' and 'throughput_mbps'
                columns containing the generated traffic data.
        """
        hours = days * 24
        time = np.arange(hours)

        # Daily seasonality: Sine wave with 24-hour period
        # Simulates peak hours (evening) and low-traffic periods (night)
        daily_pattern = 50 + 40 * np.sin(2 * np.pi * time / 24)

        # Stochastic network noise: Gaussian distribution
        # Represents random variations in network conditions
        noise = np.random.normal(0, 5, hours)

        # Linear growth trend: Simulates increasing network usage over time
        trend = time * 0.01

        traffic = daily_pattern + noise + trend

        # Anomaly injection: Simulates real-world events
        # Event spike: Represents mass events (concerts, sports events)
        # Site outage: Represents network failures or maintenance
        traffic[300:305] += 60
        traffic[1000:1005] -= 30

        # Ensure non-negative values (network throughput cannot be negative)
        traffic = np.maximum(traffic, 0)

        df = pd.DataFrame({"timestamp": time, "throughput_mbps": traffic})
        return df

    def prepare_sequences(self, data, lookback):
        """Convert time-series data into supervised learning sequences.

        Implements a sliding window approach to create input-output pairs
        for LSTM training. For each position i in the time-series, the
        method creates:
        - X[i]: A window of 'lookback' consecutive values [i, i+lookback)
        - y[i]: The next value at position i+lookback

        This sliding window mechanism preserves temporal dependencies
        required for LSTM to learn long-term patterns. The data is
        normalized using MinMaxScaler to [0, 1] range, which is critical
        for LSTM convergence and prevents gradient explosion during training.

        The scaler is serialized for use during inference, ensuring
        consistent preprocessing between training and prediction phases.

        Args:
            data (np.ndarray): 1D array of time-series values.
            lookback (int): Number of previous time steps to use as input.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Input sequences of shape
                    (n_samples, lookback) for LSTM training.
                - y (np.ndarray): Target values of shape (n_samples,).
                - scaler (MinMaxScaler): Fitted scaler for inverse transform
                    during evaluation and inference.
        """
        # Normalize data to [0, 1] range using MinMaxScaler
        # This is critical for LSTM: raw throughput values can vary widely
        # (e.g., 0-150 Mbps), causing gradient instability. Normalization
        # ensures all features are on the same scale, enabling stable
        # gradient descent and faster convergence.
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        # Persist scaler for consistent preprocessing during API inference
        joblib.dump(self.scaler, SCALER_PATH)

        # Sliding window: For each position i, create input sequence
        # [i, i+lookback) and target value at i+lookback
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback), 0])
            y.append(scaled_data[i + lookback, 0])

        return np.array(X), np.array(y), self.scaler