"""
Unit tests for LiquidityGrabDetector indicator.

This module contains comprehensive tests for the LiquidityGrabDetector class,
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

from indicators.liquidity_grab_detector import LiquidityGrabDetector
from core.state_types import SignalDirection


class TestLiquidityGrabDetector(unittest.TestCase):
    """Test cases for LiquidityGrabDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'symbol': 'EURUSD',
            'enable_wick_extension_filter': True,
            'min_wick_extension_pips': 3.0,
            'detect_same_session': True
        }
        self.detector = LiquidityGrabDetector('test_liquidity_grab', self.config)

    def test_initialization(self):
        """Test proper initialization of the detector."""
        self.assertEqual(self.detector.name, 'test_liquidity_grab')
        self.assertEqual(self.detector.symbol, 'EURUSD')
        self.assertTrue(self.detector.enable_wick_extension_filter)
        self.assertEqual(self.detector.min_wick_extension_pips, 3.0)
        self.assertTrue(self.detector.detect_same_session)

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        with self.assertRaises(ValueError):
            LiquidityGrabDetector('test', {'min_wick_extension_pips': -1.0})

    def test_lookback_period(self):
        """Test lookback period requirement."""
        self.assertEqual(self.detector.get_lookback_period(), 48)

    def test_pip_size_calculation(self):
        """Test pip size calculation for different symbols."""
        # Test forex pairs
        self.assertEqual(self.detector._get_pip_size('EURUSD'), 0.0001)
        self.assertEqual(self.detector._get_pip_size('GBPUSD'), 0.0001)

        # Test JPY pairs
        self.assertEqual(self.detector._get_pip_size('USDJPY'), 0.01)
        self.assertEqual(self.detector._get_pip_size('EURJPY'), 0.01)

        # Test commodities
        self.assertEqual(self.detector._get_pip_size('XAUUSD'), 0.01)
        self.assertEqual(self.detector._get_pip_size('XAGUSD'), 0.001)

        # Test crypto
        self.assertEqual(self.detector._get_pip_size('BTCUSD'), 1.0)
        self.assertEqual(self.detector._get_pip_size('ETHUSD'), 0.1)

        # Test default
        self.assertEqual(self.detector._get_pip_size('UNKNOWN'), 0.0001)

    def test_validate_data(self):
        """Test data validation."""
        # Test valid data
        valid_data = self._create_test_data(50)
        self.assertTrue(self.detector.validate_data(valid_data))

        # Test empty data
        empty_data = pd.DataFrame()
        self.assertFalse(self.detector.validate_data(empty_data))

        # Test insufficient data
        short_data = self._create_test_data(10)
        self.assertFalse(self.detector.validate_data(short_data))

        # Test missing columns
        incomplete_data = valid_data.drop('close', axis=1)
        self.assertFalse(self.detector.validate_data(incomplete_data))

    def test_session_completeness(self):
        """Test session completeness validation."""
        # Create session data for testing
        base_time = datetime(2024, 1, 1, 21, 0)  # Asian session start

        # Complete Asian session (21:00 to 05:00 next day)
        asian_data = [
            (base_time + timedelta(hours=i), {'high': 1.1000, 'low': 1.0900})
            for i in range(8)
        ]
        self.assertTrue(self.detector._is_session_complete('asian', asian_data))

        # Incomplete Asian session (only 3 hours)
        incomplete_asian = asian_data[:3]
        self.assertFalse(self.detector._is_session_complete('asian', incomplete_asian))

        # Complete European session
        eur_base = datetime(2024, 1, 1, 5, 0)
        european_data = [
            (eur_base + timedelta(hours=i), {'high': 1.1000, 'low': 1.0900})
            for i in range(8)
        ]
        self.assertTrue(self.detector._is_session_complete('european', european_data))

    def test_liquidity_grab_detection(self):
        """Test liquidity grab pattern detection."""
        # Test high liquidity grab
        high_grab_candle = pd.Series({
            'open': 1.0950,
            'high': 1.1010,  # Breaks above 1.1000 level
            'low': 1.0940,
            'close': 1.0945,  # Closes below level
            'volume': 1000
        })

        self.assertTrue(
            self.detector._is_liquidity_grab(high_grab_candle, 1.1000, 'high')
        )

        # Test low liquidity grab
        low_grab_candle = pd.Series({
            'open': 1.0950,
            'high': 1.0960,
            'low': 1.0890,   # Breaks below 1.0900 level
            'close': 1.0955, # Closes above level
            'volume': 1000
        })

        self.assertTrue(
            self.detector._is_liquidity_grab(low_grab_candle, 1.0900, 'low')
        )

        # Test non-grab (body crosses level)
        non_grab_candle = pd.Series({
            'open': 1.0950,
            'high': 1.1010,
            'low': 1.0940,
            'close': 1.1005,  # Closes above level - not a grab
            'volume': 1000
        })

        self.assertFalse(
            self.detector._is_liquidity_grab(non_grab_candle, 1.1000, 'high')
        )

    def test_wick_extension_validation(self):
        """Test wick extension filtering."""
        # Create candle with 5 pip extension (should pass 3 pip filter)
        candle_5pip = pd.Series({
            'open': 1.0950,
            'high': 1.1005,  # 5 pips above 1.1000
            'low': 1.0940,
            'close': 1.0945,
            'volume': 1000
        })

        self.assertTrue(
            self.detector._meets_wick_extension(candle_5pip, 1.1000, 'high')
        )

        # Create candle with 2 pip extension (should fail 3 pip filter)
        candle_2pip = pd.Series({
            'open': 1.0950,
            'high': 1.1002,  # 2 pips above 1.1000
            'low': 1.0940,
            'close': 1.0945,
            'volume': 1000
        })

        self.assertFalse(
            self.detector._meets_wick_extension(candle_2pip, 1.1000, 'high')
        )

        # Test with filter disabled
        self.detector.enable_wick_extension_filter = False
        self.assertTrue(
            self.detector._meets_wick_extension(candle_2pip, 1.1000, 'high')
        )

    def test_clear_path_validation(self):
        """Test clear path validation between level and grab."""
        # Create test data with clear path - price stays below 1.1000
        data = self._create_test_data_range(
            start_time=datetime(2024, 1, 1, 10, 0),
            hours=5,
            base_price=1.0990  # Below the 1.1000 level we're testing
        )

        start_time = datetime(2024, 1, 1, 10, 0)
        end_time = datetime(2024, 1, 1, 14, 0)

        # Set datetime as index before testing
        data_indexed = data.set_index('datetime')

        # Test clear path first (should be true since highs are below 1.1000)
        self.assertTrue(
            self.detector._has_clear_path(data_indexed, start_time, end_time, 1.1000, 'high')
        )

        # Add violation in the path
        violation_time = datetime(2024, 1, 1, 12, 0)
        data_with_violation = data.copy()
        data_with_violation.loc[data_with_violation['datetime'] == violation_time, 'high'] = 1.1010
        data_with_violation = data_with_violation.set_index('datetime')

        self.assertFalse(
            self.detector._has_clear_path(data_with_violation, start_time, end_time, 1.1000, 'high')
        )

    def test_full_detection_workflow(self):
        """Test complete detection workflow with realistic data."""
        # Create realistic session data
        data = self._create_session_test_data()

        # Run detection
        detection = self.detector.check(data, 100)

        if detection:
            self.assertIsNotNone(detection.indicator_name)
            self.assertIsInstance(detection.timestamp, datetime)
            self.assertIsInstance(detection.price, float)
            self.assertIn(detection.direction, [SignalDirection.LONG, SignalDirection.SHORT])
            self.assertIsInstance(detection.metadata, dict)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data
        minimal_data = self._create_test_data(48)  # Exactly minimum required
        detection = self.detector.check(minimal_data, 47)
        # Should not crash, may or may not detect

        # Test with no session levels
        flat_data = self._create_flat_data(50)
        detection = self.detector.check(flat_data, 49)
        self.assertIsNone(detection)

    def test_reset_functionality(self):
        """Test reset functionality."""
        # Add some internal state
        self.detector.internal_state['test_key'] = 'test_value'
        self.detector.session_levels.append({'test': 'data'})
        self.detector.detected_grabs.append({'test': 'grab'})

        # Reset
        self.detector.reset()

        # Verify state is cleared
        self.assertEqual(len(self.detector.internal_state), 0)
        self.assertEqual(len(self.detector.session_levels), 0)
        self.assertEqual(len(self.detector.detected_grabs), 0)

    def _create_test_data(self, num_candles: int) -> pd.DataFrame:
        """Create test OHLCV data."""
        base_time = datetime(2024, 1, 1, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(num_candles)]

        data = []
        base_price = 1.1000

        for i, time in enumerate(times):
            # Create realistic price movement
            price_offset = np.sin(i * 0.1) * 0.0050  # 50 pip range
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

    def _create_test_data_range(self, start_time: datetime, hours: int, base_price: float) -> pd.DataFrame:
        """Create test data for a specific time range."""
        times = [start_time + timedelta(hours=i) for i in range(hours)]

        data = []
        for time in times:
            # Make sure highs and lows are realistic relative to the base price
            high = base_price + 0.0003  # Small spread above base
            low = base_price - 0.0003   # Small spread below base

            data.append({
                'datetime': time,
                'open': base_price,
                'high': high,
                'low': low,
                'close': base_price,
                'volume': 1000
            })

        return pd.DataFrame(data)

    def _create_session_test_data(self) -> pd.DataFrame:
        """Create realistic session test data with potential liquidity grab."""
        data = []
        base_time = datetime(2024, 1, 1, 21, 0)  # Asian session start

        # Asian session with high at 1.1050
        for i in range(8):
            time = base_time + timedelta(hours=i)
            high = 1.1050 if i == 4 else 1.1000 + np.random.uniform(0, 0.0020)
            low = 1.0950 + np.random.uniform(0, 0.0020)

            data.append({
                'datetime': time,
                'open': 1.1000 + np.random.uniform(-0.0010, 0.0010),
                'high': high,
                'low': low,
                'close': 1.1000 + np.random.uniform(-0.0010, 0.0010),
                'volume': 1000
            })

        # European session with potential grab
        european_start = base_time + timedelta(hours=8)
        for i in range(16):  # More data for better testing
            time = european_start + timedelta(hours=i)

            # Create a grab on candle 10
            if i == 10:
                data.append({
                    'datetime': time,
                    'open': 1.1020,
                    'high': 1.1055,  # Breaks Asian high with 5 pips
                    'low': 1.1010,
                    'close': 1.1015,  # Closes below Asian high
                    'volume': 1000
                })
            else:
                data.append({
                    'datetime': time,
                    'open': 1.1000 + np.random.uniform(-0.0010, 0.0010),
                    'high': 1.1000 + np.random.uniform(0, 0.0020),
                    'low': 1.0980 + np.random.uniform(0, 0.0015),
                    'close': 1.1000 + np.random.uniform(-0.0010, 0.0010),
                    'volume': 1000
                })

        return pd.DataFrame(data)

    def _create_flat_data(self, num_candles: int) -> pd.DataFrame:
        """Create flat test data with no significant levels."""
        base_time = datetime(2024, 1, 1, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(num_candles)]

        data = []
        for time in times:
            data.append({
                'datetime': time,
                'open': 1.1000,
                'high': 1.1001,
                'low': 1.0999,
                'close': 1.1000,
                'volume': 1000
            })

        return pd.DataFrame(data)


class TestLiquidityGrabDetectorIntegration(unittest.TestCase):
    """Integration tests for LiquidityGrabDetector."""

    def test_with_real_data_structure(self):
        """Test with data structure similar to real market data."""
        detector = LiquidityGrabDetector('integration_test', {
            'symbol': 'EURUSD',
            'enable_wick_extension_filter': True,
            'min_wick_extension_pips': 2.0
        })

        # Create data similar to CSV structure
        data = self._create_csv_like_data()

        # Test multiple detections
        detections = []
        lookback = detector.get_lookback_period()

        for i in range(lookback, len(data)):
            window = data.iloc[max(0, i - lookback):i + 1]
            detection = detector.check(window, i)
            if detection:
                detections.append(detection)

        # Validate results
        for detection in detections:
            self.assertIsInstance(detection.metadata.get('wick_extension_pips'), float)
            self.assertGreaterEqual(detection.metadata.get('wick_extension_pips'), 2.0)

    def _create_csv_like_data(self) -> pd.DataFrame:
        """Create data structure similar to loaded CSV."""
        base_time = datetime(2024, 1, 1, 0, 0)
        num_candles = 100

        data = []
        for i in range(num_candles):
            time = base_time + timedelta(hours=i)
            base_price = 1.1000 + np.sin(i * 0.05) * 0.0100  # Trending movement

            # Occasionally create grab patterns
            if i % 25 == 0 and i > 48:  # Every 25th candle after lookback
                # Create high grab
                data.append({
                    'datetime': time,
                    'open': base_price,
                    'high': base_price + 0.0008,  # 8 pip wick
                    'low': base_price - 0.0005,
                    'close': base_price - 0.0002,  # Close below level
                    'volume': 1500
                })
            else:
                data.append({
                    'datetime': time,
                    'open': base_price,
                    'high': base_price + np.random.uniform(0, 0.0003),
                    'low': base_price - np.random.uniform(0, 0.0003),
                    'close': base_price + np.random.uniform(-0.0002, 0.0002),
                    'volume': np.random.randint(800, 1200)
                })

        return pd.DataFrame(data)


if __name__ == '__main__':
    unittest.main()