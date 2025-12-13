"""REST API service for 5G traffic forecasting and network slicing decisions.

This module provides a FastAPI-based microservice that exposes the trained
LSTM model for real-time inference. The API accepts historical traffic
measurements and returns predictions along with recommended network slicing
actions based on predicted throughput thresholds.
"""

import joblib

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import LOOKBACK_WINDOW, MODEL_PATH, SCALER_PATH

app = FastAPI(title="Ericsson 5G AI Node", version="1.2.0")


class TrafficPayload(BaseModel):
    """Request payload schema for traffic prediction endpoint.

    Attributes:
        history (list[float]): List of LOOKBACK_WINDOW (24) hourly throughput
            measurements in Mbps. Must contain exactly 24 values representing
            the most recent 24 hours of network traffic.
    """
    history: list[float]


@app.get("/")
def index():
    """Health check endpoint.

    Returns:
        dict: Service status and module identifier.
    """
    return {"status": "Online", "module": "Traffic Forecaster"}


@app.post("/predict")
def predict_traffic(payload: TrafficPayload):
    """Generate traffic forecast and network slicing recommendation.

    Accepts 24 hours of historical throughput data, applies the same
    preprocessing pipeline used during training (normalization via MinMaxScaler),
    performs LSTM inference, and returns the predicted throughput along with
    a recommended network action based on threshold-based business logic.

    The preprocessing pipeline mirrors training: input data is normalized
    using the same scaler instance saved during training, ensuring
    consistent feature scaling. The normalized sequence is reshaped to
    LSTM input format [1, LOOKBACK_WINDOW, 1] for inference.

    Business logic thresholds:
    - > 85 Mbps: SCALE_UP_RESOURCES - Proactive resource allocation to
        prevent congestion and maintain Quality of Service
    - < 20 Mbps: SCALE_DOWN_ENERGY_SAVE - Reduce resource allocation to
        optimize energy consumption during low-traffic periods
    - 20-85 Mbps: MAINTAIN - Current resource allocation is appropriate

    Args:
        payload (TrafficPayload): Request containing 24 historical measurements.

    Returns:
        dict: Dictionary containing:
            - forecast_mbps (float): Predicted throughput for next hour in Mbps
            - network_action (str): Recommended action (MAINTAIN, SCALE_UP_RESOURCES,
                or SCALE_DOWN_ENERGY_SAVE)

    Raises:
        HTTPException: If input history length does not equal LOOKBACK_WINDOW,
            or if model/scaler files are not found (status 400).
    """
    if len(payload.history) != LOOKBACK_WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Input requires exactly {LOOKBACK_WINDOW} data points."
        )

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except (FileNotFoundError, OSError, IOError) as e:
        return {
            "error": "Model not trained yet. Run main.py first.",
            "detail": str(e)
        }

    # Preprocessing: Apply same normalization as training
    input_data = np.array(payload.history).reshape(-1, 1)
    input_scaled = scaler.transform(input_data)
    input_final = input_scaled.reshape(1, LOOKBACK_WINDOW, 1)

    # LSTM Inference
    prediction_scaled = model.predict(input_final, verbose=0)
    prediction_mbps = scaler.inverse_transform(prediction_scaled)[0][0]

    # Business Logic: Network Slicing Decision
    # Thresholds determined by operational requirements:
    # - High threshold (85 Mbps): Prevents congestion-related latency spikes
    # - Low threshold (20 Mbps): Enables energy-efficient operation
    action = "MAINTAIN"
    if prediction_mbps > 85:
        action = "SCALE_UP_RESOURCES"
    elif prediction_mbps < 20:
        action = "SCALE_DOWN_ENERGY_SAVE"

    return {
        "forecast_mbps": float(round(prediction_mbps, 2)),
        "network_action": action
    }