"""Model training and evaluation module with uncertainty quantification.

This module handles LSTM model training and provides evaluation capabilities
including residual analysis for confidence interval calculation. The
confidence intervals enable risk assessment in production network slicing
decisions by quantifying prediction uncertainty.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import IMG_PATH


class Trainer:
    """Handles model training and evaluation with uncertainty quantification.

    The Trainer class encapsulates the training process and provides
    evaluation methods that include uncertainty quantification through
    residual analysis. This enables production systems to assess prediction
    reliability and make informed decisions about resource allocation.
    """

    def __init__(self, model):
        """Initialize Trainer with a compiled model instance.

        Args:
            model (tf.keras.Model): Compiled LSTM model ready for training.
        """
        self.model = model

    def train(self, X_train, y_train, epochs, batch_size):
        """Train the LSTM model on prepared sequences.

        Executes the training process with validation split to monitor
        overfitting. The validation split reserves 10% of training data
        for performance evaluation during training epochs.

        Args:
            X_train (np.ndarray): Training input sequences of shape
                (n_samples, lookback, 1).
            y_train (np.ndarray): Training target values of shape
                (n_samples,).
            epochs (int): Number of training epochs.
            batch_size (int): Number of samples per gradient update.

        Returns:
            tf.keras.callbacks.History: Training history object containing
                loss and validation loss metrics for each epoch.
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history

    def evaluate_and_plot(self, X_test, y_test, scaler):
        """Evaluate model performance and generate forecast visualization.

        Performs inference on test data, calculates 95% confidence intervals
        using residual analysis, and generates a visualization showing
        actual traffic, predictions, and uncertainty bounds.

        The confidence interval calculation uses residual analysis: the
        standard deviation of prediction errors (residuals) on the test
        set is multiplied by 1.96 (the z-score for 95% confidence under
        normality assumption) to create upper and lower bounds around
        predictions. This approach quantifies model uncertainty and enables
        risk-aware decision making in network slicing applications.

        Args:
            X_test (np.ndarray): Test input sequences of shape
                (n_samples, lookback, 1).
            y_test (np.ndarray): Test target values (scaled to [0, 1]).
            scaler (MinMaxScaler): Scaler used during preprocessing for
                inverse transformation back to original scale (Mbps).
        """
        print("Generating Advanced Performance Report...")
        predictions = self.model.predict(X_test, verbose=0)

        # Inverse transform: Convert from normalized [0, 1] range back to
        # original throughput values (Mbps). The scaler was fitted during
        # data preparation, so inverse_transform recovers the original scale.
        y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(predictions).flatten()

        # Uncertainty Quantification via Residual Analysis
        # Compute residuals (prediction errors) on test set
        residuals = y_real - y_pred

        # Calculate standard deviation of residuals as measure of model
        # prediction uncertainty
        std_dev = np.std(residuals)

        # Compute 95% confidence interval bounds
        # Multiplier 1.96 corresponds to z-score for 95% confidence under
        # normal distribution assumption (two-tailed test)
        upper_bound = y_pred + (1.96 * std_dev)
        lower_bound = y_pred - (1.96 * std_dev)

        # Visualization
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))

        limit = 200

        # Plot actual traffic values
        plt.plot(y_real[:limit], color='#00ff00', label='Actual Traffic',
                 linewidth=2, alpha=0.8)

        # Shaded confidence interval area
        plt.fill_between(
            range(limit),
            lower_bound[:limit],
            upper_bound[:limit],
            color='#ff0055', alpha=0.2, label='95% Confidence Interval'
        )

        # Plot predicted values (mean forecast)
        plt.plot(y_pred[:limit], color='#ff0055',
                 label='AI Forecast (Mean)', linestyle='--', linewidth=2)

        plt.title(
            f'5G Traffic Forecast with Uncertainty Quantification '
            f'(σ={std_dev:.2f})',
            fontsize=16
        )
        plt.xlabel('Time (Hours)', fontsize=12)
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)

        plt.savefig(IMG_PATH)
        print(f"Advanced Graph saved to {IMG_PATH}")