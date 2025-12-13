"""LSTM neural network architecture for time-series forecasting.

This module defines the deep learning architecture used for 5G traffic
prediction. The model employs a stacked LSTM architecture with dropout
regularization to capture long-term temporal dependencies while preventing
overfitting to the training data.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


class LSTMNetwork:
    """LSTM network architecture builder for traffic forecasting.

    Implements a two-layer stacked LSTM architecture optimized for
    time-series regression tasks. The architecture progressively reduces
    dimensionality (64 -> 32 units) to extract hierarchical temporal
    features while maintaining computational efficiency.
    """

    @staticmethod
    def build(lookback):
        """Construct the LSTM neural network architecture.

        Builds a sequential model with the following structure:
        - Input: (batch_size, lookback, 1) - sequences of historical values
        - Layer 1: LSTM(64) with return_sequences=True to preserve temporal
            structure for the next layer
        - Layer 2: LSTM(32) with return_sequences=False to compress temporal
            information into a single feature vector
        - Output: Dense layers for regression output

        The reduction from 64 to 32 units in the second layer enables
        the model to learn hierarchical features: the first layer captures
        short-term patterns, while the second layer abstracts these into
        longer-term trends. Dropout at 0.2 rate prevents overfitting by
        randomly deactivating 20% of neurons during training.

        Args:
            lookback (int): Number of time steps in the input sequence.
                Determines the input shape as (lookback, 1).

        Returns:
            tf.keras.Model: Compiled Sequential model with Adam optimizer
                and Mean Squared Error loss function. Input shape:
                (None, lookback, 1). Output shape: (None, 1).
        """
        model = Sequential([
            Input(shape=(lookback, 1)),

            # First LSTM layer: Captures long-term temporal dependencies
            # 64 units provides sufficient capacity for pattern recognition
            # return_sequences=True passes full sequence to next layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),

            # Second LSTM layer: Abstract temporal features
            # 32 units reduces dimensionality while preserving key patterns
            # return_sequences=False outputs single vector per sample
            LSTM(32, return_sequences=False),
            Dropout(0.2),

            # Dense layers: Map abstract features to regression output
            # ReLU activation in hidden layer introduces non-linearity
            Dense(16, activation='relu'),
            Dense(1)  # Linear output for throughput prediction (Mbps)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model