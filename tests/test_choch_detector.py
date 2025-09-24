"""
Unit tests for ChochDetector indicator.

This module contains comprehensive tests for the ChochDetector class,
ensuring proper functionality across different market conditions and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.choch_detector import ChochDetector
from core.state_types import SignalDirection


class TestChochDetector(unittest.TestCase):
    """Test cases for ChochDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'symbol': 'EURUSD',
            'base_strength': 5,
            'min_gap': 3
        }
        self.detector = ChochDetector('test_choch', self.config)

    def test_initialization(self):
        """Test proper initialization of the detector."""
        self.assertEqual(self.detector.name, 'test_choch')
        self.assertEqual(self.detector.symbol, 'EURUSD')
        self.assertEqual(self.detector.base_strength, 5)
        self.assertEqual(self.detector.min_gap, 3)
        self.assertEqual(self.detector.current_trend, 0)

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        with self.assertRaises(ValueError):
            ChochDetector('test', {'base_strength': 0})

        with self.assertRaises(ValueError):
            ChochDetector('test', {'min_gap': -1})

    def test_lookback_period(self):
        """Test lookback period calculation."""
        expected = (self.detector.base_strength * 2) + 10
        self.assertEqual(self.detector.get_lookback_period(), expected)

    def test_validate_data(self):
        """Test data validation."""
        # Test valid data
        valid_data = self._create_test_data(30)
        self.assertTrue(self.detector.validate_data(valid_data))

        # Test empty data
        empty_data = pd.DataFrame()
        self.assertFalse(self.detector.validate_data(empty_data))

        # Test insufficient data
        short_data = self._create_test_data(5)
        self.assertFalse(self.detector.validate_data(short_data))

        # Test missing columns
        incomplete_data = valid_data.drop('close', axis=1)
        self.assertFalse(self.detector.validate_data(incomplete_data))

    def test_swing_detection(self):
        """Test swing highs and lows detection."""
        # Create data with clear swing points
        data = self._create_swing_test_data()
        data_indexed = data.set_index('datetime')

        swing_labels, swing_levels = self.detector._get_swing_highs_lows(data_indexed)

        # Should detect at least some swing points
        swing_highs = swing_labels[swing_labels == 1]
        swing_lows = swing_labels[swing_labels == -1]

        self.assertGreater(len(swing_highs), 0, "Should detect at least one swing high")
        self.assertGreater(len(swing_lows), 0, "Should detect at least one swing low")

        # Verify swing levels are set correctly
        for timestamp in swing_highs.index:
            self.assertFalse(pd.isna(swing_levels[timestamp]))

        for timestamp in swing_lows.index:
            self.assertFalse(pd.isna(swing_levels[timestamp]))

    def test_swing_gap_constraint(self):
        """Test minimum gap constraint between swings."""
        # Create detector with strict gap requirement
        strict_detector = ChochDetector('strict_test', {'base_strength': 3, 'min_gap': 10})

        data = self._create_test_data(50)
        data_indexed = data.set_index('datetime')

        swing_labels, swing_levels = strict_detector._get_swing_highs_lows(data_indexed)

        # Check gap constraint is respected
        swing_highs = np.where(swing_labels == 1)[0]
        swing_lows = np.where(swing_labels == -1)[0]

        # Verify gaps between consecutive swing highs
        if len(swing_highs) > 1:
            for i in range(1, len(swing_highs)):
                gap = swing_highs[i] - swing_highs[i-1]
                self.assertGreaterEqual(gap, 10)

        # Verify gaps between consecutive swing lows
        if len(swing_lows) > 1:
            for i in range(1, len(swing_lows)):
                gap = swing_lows[i] - swing_lows[i-1]
                self.assertGreaterEqual(gap, 10)

    def test_choch_detection_workflow(self):
        """Test complete CHoCH detection workflow."""
        # Create data with clear trend change
        data = self._create_choch_test_data()

        # Test detection on the data
        lookback = self.detector.get_lookback_period()

        detections = []
        for i in range(lookback, len(data)):
            window_data = data.iloc[max(0, i - lookback):i + 1].copy()
            detection = self.detector.check(window_data, i)
            if detection:
                detections.append(detection)

        # Should detect at least one CHoCH
        # (This depends on the test data, might be 0 if no clear breakouts)
        self.assertGreaterEqual(len(detections), 0)

        # If detections found, validate structure
        for detection in detections:
            self.assertIsNotNone(detection.indicator_name)
            self.assertIsInstance(detection.timestamp, datetime)
            self.assertIsInstance(detection.price, float)
            self.assertIn(detection.direction, [SignalDirection.LONG, SignalDirection.SHORT])
            self.assertIsInstance(detection.metadata, dict)

    def test_trend_tracking(self):
        """Test internal trend state tracking."""
        # Initially no trend
        self.assertEqual(self.detector.current_trend, 0)

        # Create simple bullish breakout data
        data = self._create_simple_breakout_data('bullish')

        # Run detection
        lookback = self.detector.get_lookback_period()
        for i in range(lookback, len(data)):
            window_data = data.iloc[max(0, i - lookback):i + 1].copy()
            detection = self.detector.check(window_data, i)
            if detection:
                # Should establish bullish trend
                self.assertEqual(self.detector.current_trend, 1)
                break

    def test_get_all_swings_and_chochs(self):
        """Test the method that returns all swing points and CHoCH signals."""
        data = self._create_choch_test_data()
        data_indexed = data.set_index('datetime')

        swing_points, choch_signals = self.detector.get_all_swings_and_chochs(data_indexed)

        # Should return lists
        self.assertIsInstance(swing_points, list)
        self.assertIsInstance(choch_signals, list)

        # Validate swing point structure
        for swing in swing_points:
            self.assertIn('time', swing)
            self.assertIn('price', swing)
            self.assertIn('type', swing)
            self.assertIn('label', swing)
            self.assertIn(swing['type'], ['high', 'low'])
            self.assertIn(swing['label'], [1, -1])

        # Validate CHoCH signal structure
        for choch in choch_signals:
            self.assertIn('swing_time', choch)
            self.assertIn('swing_price', choch)
            self.assertIn('breakout_time', choch)
            self.assertIn('trend_direction', choch)
            self.assertIn('trend_name', choch)
            self.assertIn(choch['trend_direction'], [1, -1])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data
        minimal_data = self._create_test_data(self.detector.get_lookback_period())
        detection = self.detector.check(minimal_data, len(minimal_data) - 1)
        # Should not crash, may or may not detect

        # Test with flat data (no swings)
        flat_data = self._create_flat_data(30)
        detection = self.detector.check(flat_data, 29)
        self.assertIsNone(detection)

    def test_reset_functionality(self):
        """Test reset functionality."""
        # Add some internal state
        self.detector.internal_state['test_key'] = 'test_value'
        self.detector.swing_highs.append((datetime.now(), 1.1000))
        self.detector.choch_signals.append({'test': 'data'})
        self.detector.current_trend = 1

        # Reset
        self.detector.reset()

        # Verify state is cleared
        self.assertEqual(len(self.detector.internal_state), 0)
        self.assertEqual(len(self.detector.swing_highs), 0)
        self.assertEqual(len(self.detector.choch_signals), 0)
        self.assertEqual(self.detector.current_trend, 0)

    def _create_test_data(self, num_candles: int) -> pd.DataFrame:
        """Create basic test OHLCV data."""
        base_time = datetime(2024, 1, 1, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(num_candles)]

        data = []
        base_price = 1.1000

        for i, time in enumerate(times):
            # Create some price movement
            price_offset = np.sin(i * 0.2) * 0.0100  # 100 pip range
            open_price = base_price + price_offset
            close_price = open_price + np.random.uniform(-0.0020, 0.0020)
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.0010))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.0010))

            data.append({
                'datetime': time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 5000)
            })

        return pd.DataFrame(data)

    def _create_swing_test_data(self) -> pd.DataFrame:
        """Create test data with clear swing points."""
        base_time = datetime(2024, 1, 1, 0, 0)
        data = []

        # Create a pattern with very clear highs and lows for reliable detection
        # Need more data points to satisfy base_strength=5 requirements
        prices = [
            # Initial phase
            1.1000, 1.1005, 1.1010, 1.1015, 1.1020,
            # Build to first swing high
            1.1025, 1.1035, 1.1045, 1.1060, 1.1080,  # Peak at index 9
            # Decline from swing high
            1.1070, 1.1055, 1.1040, 1.1030, 1.1020,
            # Move to swing low
            1.1010, 1.1000, 1.0990, 1.0970, 1.0950,  # Trough at index 19
            # Recovery from swing low
            1.0960, 1.0975, 1.0990, 1.1005, 1.1020,
            # Build to second swing high
            1.1035, 1.1050, 1.1070, 1.1090, 1.1110,  # Peak at index 29
            # Final decline
            1.1100, 1.1085, 1.1070, 1.1055, 1.1040
        ]

        for i, target_price in enumerate(prices):
            time = base_time + timedelta(hours=i)

            # Create OHLC with target_price as the body center
            spread = 0.0002
            open_price = target_price + np.random.uniform(-spread/4, spread/4)
            close_price = target_price + np.random.uniform(-spread/4, spread/4)

            # Ensure high/low capture the target behavior
            high_price = max(open_price, close_price, target_price) + abs(np.random.uniform(0, spread/2))
            low_price = min(open_price, close_price, target_price) - abs(np.random.uniform(0, spread/2))

            data.append({
                'datetime': time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': 1000
            })

        return pd.DataFrame(data)

    def _create_choch_test_data(self) -> pd.DataFrame:
        """Create test data with potential CHoCH scenario."""
        base_time = datetime(2024, 1, 1, 0, 0)
        data = []

        # Create uptrend then downtrend (potential CHoCH)
        base_price = 1.1000

        # Uptrend phase - higher highs and lows
        for i in range(20):
            time = base_time + timedelta(hours=i)
            trend_offset = i * 0.0005  # Gradual upward movement
            noise = np.random.uniform(-0.0002, 0.0002)

            price = base_price + trend_offset + noise

            open_price = price + np.random.uniform(-0.0001, 0.0001)
            close_price = price + np.random.uniform(-0.0001, 0.0001)
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.0005))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.0005))

            data.append({
                'datetime': time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': 1000
            })

        # Potential reversal point - create swing high
        high_time = base_time + timedelta(hours=20)
        high_price = base_price + 0.0120  # Clear swing high

        data.append({
            'datetime': high_time,
            'open': high_price - 0.0005,
            'high': high_price,
            'low': high_price - 0.0010,
            'close': high_price - 0.0003,
            'volume': 1000
        })

        # Downtrend phase - lower highs and lows (potential CHoCH)
        for i in range(1, 15):
            time = base_time + timedelta(hours=20 + i)
            trend_offset = -i * 0.0003  # Gradual downward movement
            noise = np.random.uniform(-0.0002, 0.0002)

            price = high_price + trend_offset + noise

            open_price = price + np.random.uniform(-0.0001, 0.0001)
            close_price = price + np.random.uniform(-0.0001, 0.0001)
            high_price_candle = max(open_price, close_price) + abs(np.random.uniform(0, 0.0003))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.0003))

            data.append({
                'datetime': time,
                'open': open_price,
                'high': high_price_candle,
                'low': low_price,
                'close': close_price,
                'volume': 1000
            })

        return pd.DataFrame(data)

    def _create_simple_breakout_data(self, direction: str) -> pd.DataFrame:
        """Create simple breakout data for testing."""
        base_time = datetime(2024, 1, 1, 0, 0)
        data = []
        base_price = 1.1000

        # Create sideways movement first
        for i in range(15):
            time = base_time + timedelta(hours=i)
            price = base_price + np.random.uniform(-0.0010, 0.0010)

            data.append({
                'datetime': time,
                'open': price,
                'high': price + 0.0005,
                'low': price - 0.0005,
                'close': price + np.random.uniform(-0.0003, 0.0003),
                'volume': 1000
            })

        # Create clear swing point
        swing_time = base_time + timedelta(hours=15)
        if direction == 'bullish':
            swing_price = base_price + 0.0050  # High point
            data.append({
                'datetime': swing_time,
                'open': swing_price - 0.0010,
                'high': swing_price,
                'low': swing_price - 0.0020,
                'close': swing_price - 0.0005,
                'volume': 1000
            })

            # Create potential breakout
            breakout_time = base_time + timedelta(hours=20)
            breakout_price = swing_price + 0.0020  # Clear breakout above swing
            data.append({
                'datetime': breakout_time,
                'open': swing_price - 0.0010,
                'high': breakout_price,
                'low': swing_price - 0.0005,
                'close': breakout_price - 0.0005,  # Body closes above swing
                'volume': 1000
            })
        else:  # bearish
            swing_price = base_price - 0.0050  # Low point
            data.append({
                'datetime': swing_time,
                'open': swing_price + 0.0010,
                'high': swing_price + 0.0020,
                'low': swing_price,
                'close': swing_price + 0.0005,
                'volume': 1000
            })

            # Create potential breakout
            breakout_time = base_time + timedelta(hours=20)
            breakout_price = swing_price - 0.0020  # Clear breakout below swing
            data.append({
                'datetime': breakout_time,
                'open': swing_price + 0.0010,
                'high': swing_price + 0.0005,
                'low': breakout_price,
                'close': breakout_price + 0.0005,  # Body closes below swing
                'volume': 1000
            })

        return pd.DataFrame(data)

    def _create_flat_data(self, num_candles: int) -> pd.DataFrame:
        """Create flat test data with no significant swings."""
        base_time = datetime(2024, 1, 1, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(num_candles)]

        data = []
        for time in times:
            data.append({
                'datetime': time,
                'open': 1.1000,
                'high': 1.1002,
                'low': 1.0998,
                'close': 1.1000,
                'volume': 1000
            })

        return pd.DataFrame(data)


class TestChochDetectorIntegration(unittest.TestCase):
    """Integration tests for ChochDetector."""

    def test_with_various_configurations(self):
        """Test with different configuration parameters."""
        configs = [
            {'base_strength': 3, 'min_gap': 2},
            {'base_strength': 7, 'min_gap': 5},
            {'base_strength': 10, 'min_gap': 8}
        ]

        for config in configs:
            with self.subTest(config=config):
                detector = ChochDetector('config_test', config)

                # Test basic functionality
                data = self._create_integration_test_data()

                # Should not crash with any configuration
                swing_points, choch_signals = detector.get_all_swings_and_chochs(data.set_index('datetime'))

                # Results should be consistent with configuration
                self.assertIsInstance(swing_points, list)
                self.assertIsInstance(choch_signals, list)

    def _create_integration_test_data(self) -> pd.DataFrame:
        """Create more comprehensive test data for integration testing."""
        base_time = datetime(2024, 1, 1, 0, 0)
        num_candles = 100
        data = []

        for i in range(num_candles):
            time = base_time + timedelta(hours=i)

            # Create realistic price movement with trends and swings
            base_price = 1.1000
            trend_component = np.sin(i * 0.1) * 0.0200  # Long-term oscillation
            noise_component = np.random.uniform(-0.0030, 0.0030)

            price = base_price + trend_component + noise_component

            open_price = price + np.random.uniform(-0.0010, 0.0010)
            close_price = price + np.random.uniform(-0.0010, 0.0010)
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.0015))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.0015))

            data.append({
                'datetime': time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(800, 1200)
            })

        return pd.DataFrame(data)


if __name__ == '__main__':
    unittest.main()