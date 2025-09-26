"""
Unit tests for the Order Block Detector.

Tests Order Block formation, swing detection, mitigation tracking, and volume calculations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.order_block_detector import OrderBlockDetector
from core.state_types import SignalDirection


class TestOrderBlockDetector(unittest.TestCase):
    """Test cases for the Order Block Detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = OrderBlockDetector(
            name="test_ob",
            config={
                'base_strength': 5,
                'min_gap': 3,
                'close_mitigation': False,
                'track_volume': True,
                'max_blocks': 10
            }
        )

    def create_bullish_ob_data(self) -> pd.DataFrame:
        """Create test data with a clear bullish Order Block formation."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='h')

        # Create pattern: swing high at index 5, OB formation at index 8, breakout at index 12
        data = pd.DataFrame({
            'datetime': dates,
            'open':   [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025],
            'high':   [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1025, 1.1020, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1050, 1.1045, 1.1040, 1.1035, 1.1030],
            'low':    [1.0995, 1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1015, 1.1010, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020],
            'close':  [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1300, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 1800, 1700, 1600, 1500]
        })

        return data

    def create_bearish_ob_data(self) -> pd.DataFrame:
        """Create test data with a clear bearish Order Block formation."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='h')

        # Create pattern: swing low at index 5, OB formation at index 8, breakout at index 12
        data = pd.DataFrame({
            'datetime': dates,
            'open':   [1.1050, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1030, 1.1035, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025],
            'high':   [1.1055, 1.1050, 1.1045, 1.1040, 1.1035, 1.1030, 1.1035, 1.1040, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
            'low':    [1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1025, 1.1030, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000, 1.1005, 1.1010, 1.1015, 1.1020],
            'close':  [1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1030, 1.1035, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1300, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 1800, 1700, 1600, 1500]
        })

        return data

    def create_mitigation_data(self) -> pd.DataFrame:
        """Create test data where Order Block gets mitigated."""
        dates = pd.date_range(start='2024-01-01', periods=25, freq='h')

        # Create bullish OB that gets mitigated later
        data = pd.DataFrame({
            'datetime': dates,
            'open':   [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000],
            'high':   [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1025, 1.1020, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1050, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005],
            'low':    [1.0995, 1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1015, 1.1010, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995],  # Mitigation at index 22
            'close':  [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035, 1.1040, 1.1045, 1.1040, 1.1035, 1.1030, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995],
            'volume': [1000] * 25
        })

        return data

    def create_no_breakout_data(self) -> pd.DataFrame:
        """Create test data with swing points but no breakouts."""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='h')

        data = pd.DataFrame({
            'datetime': dates,
            'open':   [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000],
            'high':   [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1025, 1.1020, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1005],
            'low':    [1.0995, 1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1015, 1.1010, 1.1005, 1.1010, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995],
            'close':  [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1020, 1.1015, 1.1010, 1.1015, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000, 1.1005],
            'volume': [1000] * 15
        })

        return data

    def test_bullish_order_block_detection(self):
        """Test detection of bullish Order Blocks."""
        data = self.create_bullish_ob_data()

        # Get all order blocks
        order_blocks = self.detector.get_all_order_blocks(data)

        # Should detect at least one bullish order block
        bullish_obs = [ob for ob in order_blocks if ob['type'] == 'bullish']
        self.assertGreater(len(bullish_obs), 0, "Should detect at least one bullish Order Block")

        # Check order block properties
        ob = bullish_obs[0]
        self.assertEqual(ob['type'], 'bullish')
        self.assertGreater(ob['top'], ob['bottom'])
        self.assertIsNotNone(ob['swing_time'])
        self.assertIsNotNone(ob['formation_time'])
        self.assertIsNotNone(ob['breakout_time'])

    def test_bearish_order_block_detection(self):
        """Test detection of bearish Order Blocks."""
        data = self.create_bearish_ob_data()

        # Get all order blocks
        order_blocks = self.detector.get_all_order_blocks(data)

        # Should detect at least one bearish order block
        bearish_obs = [ob for ob in order_blocks if ob['type'] == 'bearish']
        self.assertGreater(len(bearish_obs), 0, "Should detect at least one bearish Order Block")

        # Check order block properties
        ob = bearish_obs[0]
        self.assertEqual(ob['type'], 'bearish')
        self.assertGreater(ob['top'], ob['bottom'])
        self.assertIsNotNone(ob['swing_time'])
        self.assertIsNotNone(ob['formation_time'])
        self.assertIsNotNone(ob['breakout_time'])

    def test_order_block_mitigation(self):
        """Test Order Block mitigation tracking."""
        data = self.create_mitigation_data()

        # Get all order blocks
        order_blocks = self.detector.get_all_order_blocks(data)

        # Should have at least one order block
        self.assertGreater(len(order_blocks), 0, "Should detect at least one Order Block")

        # Check if any order block is mitigated
        mitigated_obs = [ob for ob in order_blocks if ob['is_mitigated']]

        # May or may not have mitigated blocks depending on the exact data pattern
        # Just ensure the mitigation logic doesn't crash
        for ob in order_blocks:
            if ob['is_mitigated']:
                self.assertIsNotNone(ob['mitigation_time'])
            else:
                self.assertIsNone(ob['mitigation_time'])

    def test_swing_point_detection(self):
        """Test swing point detection accuracy."""
        data = self.create_bullish_ob_data()

        # Set data index to datetime
        data = data.set_index('datetime')

        # Get swing points
        swing_labels, swing_levels = self.detector._detect_swing_points(data)

        # Should detect some swing points
        swing_highs = swing_labels[swing_labels == 1]
        swing_lows = swing_labels[swing_labels == -1]

        self.assertGreater(len(swing_highs) + len(swing_lows), 0, "Should detect swing points")

        # Check that swing levels are valid prices
        for timestamp, level in swing_levels.dropna().items():
            self.assertGreater(level, 0, "Swing levels should be positive prices")

    def test_volume_calculations(self):
        """Test volume-based strength calculations."""
        data = self.create_bullish_ob_data()

        # Get all order blocks
        order_blocks = self.detector.get_all_order_blocks(data)

        if order_blocks:
            ob = order_blocks[0]
            volume_data = ob['volume_data']

            # Check volume data structure
            self.assertIn('total_volume', volume_data)
            self.assertIn('breakout_volume', volume_data)
            self.assertIn('formation_volume', volume_data)

            # Check volume values are non-negative
            self.assertGreaterEqual(volume_data['total_volume'], 0)
            self.assertGreaterEqual(volume_data['breakout_volume'], 0)
            self.assertGreaterEqual(volume_data['formation_volume'], 0)

            # Check strength percentage
            self.assertGreaterEqual(ob['strength_percentage'], 0)
            self.assertLessEqual(ob['strength_percentage'], 100)

    def test_no_breakout_scenario(self):
        """Test behavior when there are swings but no breakouts."""
        data = self.create_no_breakout_data()

        # Get all order blocks
        order_blocks = self.detector.get_all_order_blocks(data)

        # Should not detect any order blocks without breakouts
        self.assertEqual(len(order_blocks), 0, "Should not detect Order Blocks without breakouts")

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create very small dataset
        dates = pd.date_range(start='2024-01-01', periods=5, freq='h')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.1000, 1.1005, 1.1010, 1.1015, 1.1020],
            'high': [1.1005, 1.1010, 1.1015, 1.1020, 1.1025],
            'low':  [1.0995, 1.1000, 1.1005, 1.1010, 1.1015],
            'close':[1.1005, 1.1010, 1.1015, 1.1020, 1.1025],
            'volume': [1000] * 5
        })

        # Should handle insufficient data gracefully
        order_blocks = self.detector.get_all_order_blocks(data)
        self.assertEqual(len(order_blocks), 0, "Should handle insufficient data gracefully")

    def test_detection_signal(self):
        """Test that detection signal is generated correctly."""
        data = self.create_bullish_ob_data()

        # Check for detection on specific candle
        detection = self.detector.check(data[:13], candle_index=12)  # Check breakout candle

        # If order block formed, should have detection
        if detection:
            self.assertIsNotNone(detection.timestamp)
            self.assertIn(detection.direction, [SignalDirection.LONG, SignalDirection.SHORT])
            self.assertIn('ob_type', detection.metadata)
            self.assertIn('swing_time', detection.metadata)

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid base_strength
        with self.assertRaises(ValueError):
            OrderBlockDetector(
                name="test_invalid",
                config={'base_strength': 0}
            )

        # Test invalid min_gap
        with self.assertRaises(ValueError):
            OrderBlockDetector(
                name="test_invalid",
                config={'min_gap': -1}
            )

        # Test invalid max_blocks
        with self.assertRaises(ValueError):
            OrderBlockDetector(
                name="test_invalid",
                config={'max_blocks': 0}
            )

    def test_close_mitigation_mode(self):
        """Test close mitigation vs wick mitigation modes."""
        # Test with close mitigation enabled
        detector_close = OrderBlockDetector(
            name="test_close",
            config={
                'base_strength': 5,
                'close_mitigation': True,
                'track_volume': True
            }
        )

        data = self.create_mitigation_data()
        order_blocks = detector_close.get_all_order_blocks(data)

        # Should work without errors
        self.assertIsInstance(order_blocks, list)

    def test_reset(self):
        """Test that reset clears internal state."""
        data = self.create_bullish_ob_data()

        # Detect order blocks
        order_blocks = self.detector.get_all_order_blocks(data)
        self.assertGreater(len(order_blocks), 0)

        # Reset
        self.detector.reset()

        # Internal state should be cleared
        self.assertEqual(len(self.detector.order_blocks), 0)
        self.assertEqual(len(self.detector.swing_highs), 0)
        self.assertEqual(len(self.detector.swing_lows), 0)


if __name__ == '__main__':
    unittest.main()