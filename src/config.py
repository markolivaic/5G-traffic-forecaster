"""Configuration module for 5G Traffic Forecaster.

This module centralizes all hyperparameters, data generation parameters,
and file system paths required for model training, inference, and visualization.
Using a centralized configuration approach ensures consistency across all
components of the system and simplifies deployment across environments.
"""

import os

# Data Generation Configuration
DAYS = 180
HOURS_PER_DAY = 24
TOTAL_STEPS = DAYS * HOURS_PER_DAY

# Model Hyperparameters
LOOKBACK_WINDOW = 24
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 20
BATCH_SIZE = 32

# File System Paths
# BASE_DIR: Resolves to project root by traversing two directory levels
# (from src/config.py -> src/ -> project_root/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "5g_lstm_v1.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.gz")
IMG_PATH = os.path.join(BASE_DIR, "reports", "forecast_result.png")