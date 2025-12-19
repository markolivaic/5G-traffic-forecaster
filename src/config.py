"""Configuration module for 5G Traffic Forecaster.

This module centralizes all hyperparameters, data generation parameters,
and file system paths required for model training, inference, and visualization.
Using a centralized configuration approach ensures consistency across all
components of the system and simplifies deployment across environments.
"""

from pathlib import Path

# Data Generation Configuration
# Simulating 6 months of RAN traffic to capture seasonal trends
DAYS = 180
HOURS_PER_DAY = 24
TOTAL_STEPS = DAYS * HOURS_PER_DAY

# Model Hyperparameters
# Lookback Window: 24 hours. 
# Chosen to capture the diurnal (day/night) cycle inherent in telecom traffic.
LOOKBACK_WINDOW = 24
TRAIN_TEST_SPLIT = 0.8

# Training parameters tuned for LSTM convergence on stochastic data
EPOCHS = 20
BATCH_SIZE = 32

# File System Paths
# Using pathlib for robust path handling across OS environments
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "5g_lstm_v1.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.gz"
IMG_PATH = BASE_DIR / "reports" / "forecast_result.png"