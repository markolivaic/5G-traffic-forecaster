"""Unit tests for 5G Traffic Forecaster core components.

This module contains test cases for data generation, preprocessing,
and model architecture validation. Tests ensure that data loaders
produce expected outputs and that the LSTM model architecture is
constructed correctly.
"""

import os
import sys
import unittest

import numpy as np

# Add project root to path to enable imports from src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.lstm_model import LSTMNetwork


class Test5GSystem(unittest.TestCase):
    """Test suite for 5G Traffic Forecaster system components."""

    def test_data_generation_shape(self):
        """Verify DataLoader generates data with correct structure.

        Validates that the data generation method produces a DataFrame
        with the expected columns and correct number of rows based on
        the specified number of days.

        Assertions:
            - Generated DataFrame is not empty
            - DataFrame contains 'throughput_mbps' column
            - DataFrame length matches expected value (days * 24 hours)
        """
        print("\nTesting Data Generation...")
        loader = DataLoader()
        df = loader.generate_ran_traffic(days=10)

        self.assertFalse(df.empty, "Dataset must not be empty")
        self.assertIn(
            'throughput_mbps',
            df.columns,
            "Column 'throughput_mbps' must exist"
        )
        self.assertEqual(
            len(df), 10 * 24,
            "Number of rows must equal days * 24 hours"
        )

    def test_model_architecture(self):
        """Verify LSTM model architecture is constructed correctly.

        Validates that the LSTM network is built with the expected
        input shape, ensuring compatibility with the preprocessing
        pipeline output format.

        Assertions:
            - Model input shape matches expected (None, lookback, 1)
            - Model can be instantiated without errors
        """
        print("\nTesting AI Model Architecture...")
        lookback = 24
        model = LSTMNetwork.build(lookback)

        # Verify input shape matches expected format for LSTM
        expected_shape = (None, lookback, 1)
        self.assertEqual(
            model.input_shape,
            expected_shape,
            "LSTM input shape is incorrect"
        )
        print("Model built successfully.")


if __name__ == '__main__':
    unittest.main()