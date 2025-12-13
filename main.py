"""Main training pipeline for 5G Traffic Forecaster.

This module orchestrates the complete machine learning pipeline from data
generation through model training to evaluation. It serves as the entry
point for model training and artifact generation (serialized model, scaler,
and evaluation visualizations).
"""

import os
from src.config import (
    DAYS, LOOKBACK_WINDOW, TRAIN_TEST_SPLIT, EPOCHS, BATCH_SIZE, MODEL_PATH
)
from src.data_loader import DataLoader
from src.lstm_model import LSTMNetwork
from src.trainer import Trainer

# Ensure required directories exist for model artifacts and reports
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


if __name__ == "__main__":
    """Execute the complete training pipeline.

    The pipeline consists of five stages:
    1. Data Engineering: Generate synthetic traffic data and prepare sequences
    2. Model Construction: Build LSTM architecture
    3. Training: Fit model on training data
    4. Persistence: Save trained model and scaler for inference
    5. Evaluation: Generate performance metrics and visualization
    """
    print("Initializing 5G Traffic Forecaster System...")

    # Stage 1: Data Engineering (ETL)
    loader = DataLoader()
    df = loader.generate_ran_traffic(days=DAYS)
    data = df['throughput_mbps'].values
    X, y, scaler = loader.prepare_sequences(data, LOOKBACK_WINDOW)

    # Reshape sequences for LSTM input format: [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/Test Split
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset Shape: {X.shape}, Training on {len(X_train)} samples.")

    # Stage 2: Model Construction
    model = LSTMNetwork.build(LOOKBACK_WINDOW)
    trainer = Trainer(model)

    # Stage 3: Model Training
    trainer.train(X_train, y_train, EPOCHS, BATCH_SIZE)

    # Stage 4: Persist Model Artifacts
    model.save(MODEL_PATH)
    print(f"Model serialized to {MODEL_PATH}")

    # Stage 5: Evaluation and Visualization
    trainer.evaluate_and_plot(X_test, y_test, scaler)