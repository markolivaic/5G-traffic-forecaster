"""Main training pipeline for 5G Traffic Forecaster.

Handles the end-to-end machine learning lifecycle: ETL, Model Training,
Persistence, and Performance Evaluation.
"""

import os
import sys
import logging
from src.config import (
    DAYS, LOOKBACK_WINDOW, TRAIN_TEST_SPLIT, EPOCHS, BATCH_SIZE, MODEL_PATH
)
from src.data_loader import DataLoader
from src.lstm_model import LSTMNetwork
from src.trainer import Trainer

# Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Executes the complete training workflow."""
    
    # 1. Setup Environment
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    logger.info("Initializing 5G AI Training Pipeline...")

    # 2. Data Engineering (ETL)
    loader = DataLoader()
    df = loader.generate_ran_traffic(days=DAYS)
    data = df['throughput_mbps'].values
    
    X, y, scaler = loader.prepare_sequences(data, LOOKBACK_WINDOW)

    # Reshape for LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/Test Split
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Dataset prepared. Input Shape: {X.shape}")
    logger.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Model Construction
    model = LSTMNetwork.build(LOOKBACK_WINDOW)
    trainer = Trainer(model)

    # 4. Training Loop
    logger.info(f"Starting LSTM training for {EPOCHS} epochs...")
    trainer.train(X_train, y_train, EPOCHS, BATCH_SIZE)

    # 5. Persistence
    model.save(MODEL_PATH)
    logger.info(f"Model artifact serialized to: {MODEL_PATH}")

    # 6. Evaluation
    logger.info("Generating performance visualization...")
    trainer.evaluate_and_plot(X_test, y_test, scaler)
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()