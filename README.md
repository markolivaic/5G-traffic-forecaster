# 5G Traffic Forecaster: Proactive Network Slicing with LSTM Neural Networks

## Project Overview

The 5G Traffic Forecaster is a Proof-of-Concept system designed to address Radio Access Network (RAN) congestion challenges through predictive analytics. The system employs Long Short-Term Memory (LSTM) neural networks to forecast network traffic patterns and enable proactive resource allocation through network slicing mechanisms.

Traditional reactive approaches to network resource management introduce latency penalties when responding to traffic spikes. This system provides forward-looking predictions with uncertainty quantification, allowing network operators to preemptively scale resources before congestion occurs, thereby reducing latency and optimizing resource utilization.

## System Architecture

The system implements a complete machine learning pipeline from data generation through model deployment:

```
Data Generation → Preprocessing → LSTM Network → Uncertainty Quantification → REST API
```

### Pipeline Components

1. **Data Generation**: Synthetic 5G RAN traffic generation with realistic patterns including daily seasonality, stochastic noise, linear growth trends, and anomalous events.

2. **Preprocessing**: Time-series data is transformed into supervised learning sequences using a sliding window approach. Data normalization via MinMaxScaler ensures stable LSTM training.

3. **LSTM Network**: Two-layer stacked LSTM architecture (64 → 32 units) with dropout regularization to capture temporal dependencies while preventing overfitting.

4. **Uncertainty Quantification**: Residual analysis on test data enables calculation of 95% confidence intervals, providing risk assessment capabilities for production decision-making.

5. **REST API**: FastAPI-based microservice exposes the trained model for real-time inference with network slicing recommendations based on predicted throughput thresholds.

## Key Features

- **Time-Series Forecasting**: Multi-step ahead predictions using historical traffic patterns
- **Confidence Intervals**: 95% confidence bounds enable risk-aware network slicing decisions
- **Auto-scaling Logic**: Threshold-based recommendations for resource allocation optimization
- **Uncertainty Quantification**: Statistical analysis of prediction errors provides operational insight
- **Production-Ready API**: RESTful service for integration with network management systems

## Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- 4GB RAM minimum (8GB recommended for training)
- Internet connection for dependency installation

## Installation & Setup

### Virtual Environment Setup

Execute the following commands to establish an isolated Python environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Dependency Installation

Install required packages using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

Execute the unit test suite to verify system integrity:

```bash
python test_core.py
```

## Usage

### Model Training

Execute the training pipeline to generate model artifacts:

```bash
python main.py
```

This script performs the following operations:

1. Generates synthetic traffic data (default: 180 days)
2. Preprocesses data into LSTM-compatible sequences
3. Trains the LSTM model with configured hyperparameters
4. Saves model and scaler artifacts to `models/` directory
5. Generates evaluation visualization in `reports/` directory

Training artifacts include:

- `models/5g_lstm_v1.keras`: Serialized TensorFlow model
- `models/scaler.gz`: Preprocessing scaler for inference consistency
- `reports/forecast_result.png`: Performance visualization with confidence intervals

### Interactive Dashboard

Launch the Streamlit dashboard for interactive visualization:

```bash
streamlit run dashboard.py
```

The dashboard provides:

- Real-time traffic simulation
- Forecast visualization with confidence intervals
- Network slicing decision recommendations
- Adjustable simulation parameters

Access the dashboard at `http://localhost:8501` after execution.

### Microservice API

Start the FastAPI service for programmatic access:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API provides two endpoints:

#### Health Check

```
GET /
```

Returns service status and module identifier.

#### Traffic Prediction

```
POST /predict
Content-Type: application/json

{
    "history": [45.2, 48.1, 52.3, ..., 67.8]  // 24 values
}
```

Response format:

```json
{
  "forecast_mbps": 71.23,
  "network_action": "MAINTAIN"
}
```

Network actions:

- `SCALE_UP_RESOURCES`: Predicted throughput > 85 Mbps
- `MAINTAIN`: Predicted throughput between 20-85 Mbps
- `SCALE_DOWN_ENERGY_SAVE`: Predicted throughput < 20 Mbps

API documentation is available at `http://localhost:8000/docs` when the service is running.

## Docker Deployment

### Build Container Image

Construct the Docker image with the following command:

```bash
docker build -t 5g-traffic-forecaster:latest .
```

The build process automatically executes model training to ensure the container includes all required artifacts.

### Run Container

Execute the containerized service:

```bash
docker run -p 8000:8000 5g-traffic-forecaster:latest
```

The API service will be accessible at `http://localhost:8000`.

### Container Configuration

The Dockerfile configuration:

- Base image: `python:3.9-slim`
- Working directory: `/app`
- Training executed during build phase
- API service exposed on port 8000

## CI/CD & Testing

The project includes automated testing via GitHub Actions. The CI pipeline performs:

1. **Code Checkout**: Retrieves source code from repository
2. **Python Environment Setup**: Configures Python 3.9
3. **Dependency Installation**: Installs requirements with caching
4. **Unit Test Execution**: Runs test suite via unittest framework
5. **Docker Build Validation**: Verifies container build process

Test execution:

```bash
python -m unittest discover -s . -p "test_*.py"
```

## Technical Highlights

### Latency Reduction

Proactive resource allocation based on traffic forecasts eliminates reactive scaling delays. By predicting congestion events before they occur, the system reduces service interruption and maintains Quality of Service (QoS) metrics.

### Resource Optimization

Intelligent scaling recommendations optimize computational and energy resources. The system recommends scaling down during low-traffic periods, reducing operational costs while maintaining service availability during high-demand periods.

### Uncertainty Quantification

Statistical confidence intervals enable risk assessment in network slicing decisions. Operators can evaluate forecast reliability and make informed decisions about resource allocation, balancing service guarantees against resource costs.

## Project Structure

```
5G-Traffic-Forecaster/
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI microservice
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data_loader.py         # Data generation and preprocessing
│   ├── lstm_model.py          # LSTM architecture definition
│   └── trainer.py             # Training and evaluation logic
├── models/                    # Model artifacts (generated)
├── reports/                   # Visualizations (generated)
├── notebooks/
│   └── research_analysis.ipynb
├── main.py                    # Training pipeline entry point
├── dashboard.py               # Streamlit dashboard
├── test_core.py               # Unit test suite
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── README.md                  # This file
```

## Configuration

Key hyperparameters can be adjusted in `src/config.py`:

- `DAYS`: Data generation period (default: 180 days)
- `LOOKBACK_WINDOW`: Historical time steps for prediction (default: 24 hours)
- `TRAIN_TEST_SPLIT`: Training data proportion (default: 0.8)
- `EPOCHS`: Training iterations (default: 20)
- `BATCH_SIZE`: Gradient update batch size (default: 32)

## License

This project is provided as a Proof-of-Concept for research and evaluation purposes.

## Contact

For technical inquiries regarding architecture, implementation, or deployment, please refer to the inline documentation in source files or the API documentation interface.
