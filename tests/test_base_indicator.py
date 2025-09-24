"""
Tests for the base indicator abstract class.

Tests the base functionality, helper methods, and ensures the abstract
interface is properly defined.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.base_indicator import BaseIndicator
from core.state_types import Detection, SignalDirection


class MockIndicator(BaseIndicator):
    """Mock implementation of BaseIndicator for testing."""

    def __init__(self, name: str, lookback: int = 20, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.lookback = lookback
        self.check_called = False

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """Mock check implementation."""
        self.check_called = True

        if not self.validate_data(data):
            return None

        # Return a detection if config says to
        if self.config.get('always_detect', False):
            current = data.iloc[-1]
            return self.create_detection(
                timestamp=current['datetime'],
                candle_index=candle_index,
                price=current['close'],
                direction=SignalDirection.LONG,
                metadata={'test': True}
            )
        return None

    def get_lookback_period(self) -> int:
        """Return the lookback period."""
        return self.lookback


class TestBaseIndicator(unittest.TestCase):
    """Test the base indicator functionality."""

    def setUp(self):
        """Set up test data."""
        self.sample_data = self._create_sample_data(100)

    def _create_sample_data(self, num_candles: int) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=num_candles, freq='1h')

        # Generate random price data
        np.random.seed(42)
        close_prices = 1.1000 + np.random.randn(num_candles) * 0.001

        data = pd.DataFrame({
            'datetime': dates,
            'open': close_prices + np.random.randn(num_candles) * 0.0001,
            'high': close_prices + abs(np.random.randn(num_candles) * 0.0002),
            'low': close_prices - abs(np.random.randn(num_candles) * 0.0002),
            'close': close_prices,
            'volume': np.random.randint(1000, 5000, num_candles)
        })

        return data

    def test_initialization(self):
        """Test indicator initialization."""
        indicator = MockIndicator("test_indicator", lookback=30)

        self.assertEqual(indicator.name, "test_indicator")
        self.assertEqual(indicator.get_lookback_period(), 30)
        self.assertEqual(indicator.config, {})
        self.assertEqual(indicator.internal_state, {})

    def test_initialization_with_config(self):
        """Test indicator initialization with config."""
        config = {'param1': 10, 'param2': 'value'}
        indicator = MockIndicator("test_indicator", config=config)

        self.assertEqual(indicator.config, config)

    def test_data_validation_valid(self):
        """Test data validation with valid data."""
        indicator = MockIndicator("test", lookback=20)

        valid = indicator.validate_data(self.sample_data)
        self.assertTrue(valid)

    def test_data_validation_empty(self):
        """Test data validation with empty data."""
        indicator = MockIndicator("test", lookback=20)

        empty_df = pd.DataFrame()
        valid = indicator.validate_data(empty_df)
        self.assertFalse(valid)

    def test_data_validation_missing_columns(self):
        """Test data validation with missing columns."""
        indicator = MockIndicator("test", lookback=20)

        bad_data = self.sample_data.drop(columns=['volume'])
        valid = indicator.validate_data(bad_data)
        self.assertFalse(valid)

    def test_data_validation_insufficient_data(self):
        """Test data validation with insufficient data."""
        indicator = MockIndicator("test", lookback=50)

        small_data = self.sample_data.head(30)
        valid = indicator.validate_data(small_data)
        self.assertFalse(valid)

    def test_data_validation_with_nan(self):
        """Test data validation with NaN values."""
        indicator = MockIndicator("test", lookback=20)

        bad_data = self.sample_data.copy()
        bad_data.loc[5, 'close'] = np.nan
        valid = indicator.validate_data(bad_data)
        self.assertFalse(valid)

    def test_check_method_called(self):
        """Test that check method is called."""
        indicator = MockIndicator("test", lookback=20)

        result = indicator.check(self.sample_data, 100)
        self.assertTrue(indicator.check_called)

    def test_check_returns_none_for_invalid_data(self):
        """Test check returns None for invalid data."""
        indicator = MockIndicator("test", lookback=20)

        bad_data = pd.DataFrame()
        result = indicator.check(bad_data, 100)
        self.assertIsNone(result)

    def test_check_returns_detection(self):
        """Test check can return a detection."""
        config = {'always_detect': True}
        indicator = MockIndicator("test", lookback=20, config=config)

        detection = indicator.check(self.sample_data, 100)

        self.assertIsNotNone(detection)
        self.assertIsInstance(detection, Detection)
        self.assertEqual(detection.indicator_name, "test")
        self.assertEqual(detection.candle_index, 100)
        self.assertEqual(detection.direction, SignalDirection.LONG)
        self.assertEqual(detection.metadata['test'], True)

    def test_create_detection_helper(self):
        """Test the create_detection helper method."""
        indicator = MockIndicator("test_indicator", lookback=20)

        timestamp = datetime.now()
        detection = indicator.create_detection(
            timestamp=timestamp,
            candle_index=50,
            price=1.2345,
            direction=SignalDirection.SHORT,
            metadata={'key': 'value'}
        )

        self.assertEqual(detection.indicator_name, "test_indicator")
        self.assertEqual(detection.timestamp, timestamp)
        self.assertEqual(detection.candle_index, 50)
        self.assertEqual(detection.price, 1.2345)
        self.assertEqual(detection.direction, SignalDirection.SHORT)
        self.assertEqual(detection.metadata['key'], 'value')

    def test_get_recent_highs(self):
        """Test the get_recent_highs helper method."""
        indicator = MockIndicator("test", lookback=20)

        highs = indicator.get_recent_highs(self.sample_data, window=5)

        self.assertIsInstance(highs, pd.Series)
        self.assertEqual(len(highs), len(self.sample_data))

        # Check that each value is the max of its window
        for i in range(5, len(highs)):
            window_max = self.sample_data['high'].iloc[i-4:i+1].max()
            self.assertAlmostEqual(highs.iloc[i], window_max)

    def test_get_recent_lows(self):
        """Test the get_recent_lows helper method."""
        indicator = MockIndicator("test", lookback=20)

        lows = indicator.get_recent_lows(self.sample_data, window=5)

        self.assertIsInstance(lows, pd.Series)
        self.assertEqual(len(lows), len(self.sample_data))

        # Check that each value is the min of its window
        for i in range(5, len(lows)):
            window_min = self.sample_data['low'].iloc[i-4:i+1].min()
            self.assertAlmostEqual(lows.iloc[i], window_min)

    def test_calculate_pip_distance(self):
        """Test pip distance calculation."""
        indicator = MockIndicator("test", lookback=20)

        distance = indicator.calculate_pip_distance(1.1050, 1.1000)
        self.assertAlmostEqual(distance, 50.0)

        distance = indicator.calculate_pip_distance(1.1000, 1.1050)
        self.assertAlmostEqual(distance, 50.0)

        distance = indicator.calculate_pip_distance(1.2345, 1.2345)
        self.assertAlmostEqual(distance, 0.0)

    def test_reset_method(self):
        """Test the reset method."""
        indicator = MockIndicator("test", lookback=20)

        # Add some state
        indicator.internal_state['key1'] = 'value1'
        indicator.internal_state['key2'] = 123

        # Reset should clear state
        indicator.reset()
        self.assertEqual(indicator.internal_state, {})

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        indicator = MockIndicator("my_indicator", lookback=25, config={'param': 10})

        str_rep = str(indicator)
        self.assertIn("MockIndicator", str_rep)
        self.assertIn("my_indicator", str_rep)

        repr_rep = repr(indicator)
        self.assertIn("MockIndicator", repr_rep)
        self.assertIn("my_indicator", repr_rep)
        self.assertIn("25", repr_rep)
        self.assertIn("param", repr_rep)

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        # Try to create a class without implementing abstract methods
        with self.assertRaises(TypeError):
            class BadIndicator(BaseIndicator):
                pass

            # This should fail
            bad = BadIndicator("bad")

    def test_lookback_period_consistency(self):
        """Test lookback period is consistent."""
        indicator = MockIndicator("test", lookback=15)

        # Should always return the same value
        for _ in range(10):
            self.assertEqual(indicator.get_lookback_period(), 15)


if __name__ == '__main__':
    unittest.main()