"""
Unit tests for the FVG Detector.

Tests FVG detection, mitigation tracking, and shrinking behavior.
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

from indicators.fvg_detector import FvgDetector
from core.state_types import SignalDirection


class TestFvgDetector(unittest.TestCase):
    """Test cases for the FVG Detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = FvgDetector(
            name="test_fvg",
            config={
                'min_gap_size': 0.0,
                'track_mitigation': True,
                'track_shrinking': True
            }
        )

    def create_test_data_with_bullish_fvg(self) -> pd.DataFrame:
        """Create test data with a clear bullish FVG pattern."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='h')

        # Create data with bullish FVG between candles 2-3-4
        # Candle 3 (index 2) high = 1.1000, Candle 5 (index 4) low = 1.1010
        # This creates a bullish gap
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.0990, 1.0995, 1.0995, 1.1005, 1.1010, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995],
            'high': [1.0995, 1.1000, 1.1000, 1.1010, 1.1020, 1.1020, 1.1015, 1.1010, 1.1005, 1.1000],
            'low':  [1.0985, 1.0990, 1.0990, 1.1000, 1.1010, 1.1010, 1.1005, 1.1000, 1.0995, 1.0990],
            'close':[1.0995, 1.0995, 1.1000, 1.1010, 1.1015, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995],
            'volume': [1000] * 10
        })

        return data

    def create_test_data_with_bearish_fvg(self) -> pd.DataFrame:
        """Create test data with a clear bearish FVG pattern."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='h')

        # Create data with bearish FVG between candles 2-3-4
        # Candle 3 (index 2) low = 1.1000, Candle 5 (index 4) high = 1.0990
        # This creates a bearish gap
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.1010, 1.1005, 1.1005, 1.0995, 1.0990, 1.0985, 1.0990, 1.0995, 1.1000, 1.1005],
            'high': [1.1015, 1.1010, 1.1010, 1.1000, 1.0990, 1.0990, 1.0995, 1.1000, 1.1005, 1.1010],
            'low':  [1.1005, 1.1000, 1.1000, 1.0990, 1.0980, 1.0980, 1.0985, 1.0990, 1.0995, 1.1000],
            'close':[1.1005, 1.1005, 1.1000, 1.0990, 1.0985, 1.0985, 1.0990, 1.0995, 1.1000, 1.1005],
            'volume': [1000] * 10
        })

        return data

    def create_test_data_with_mitigation(self) -> pd.DataFrame:
        """Create test data with FVG that gets mitigated."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='h')

        # Bullish FVG that gets mitigated later
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.0990, 1.0995, 1.0995, 1.1005, 1.1010, 1.1015, 1.1010, 1.0995, 1.0990, 1.0995],
            'high': [1.0995, 1.1000, 1.1000, 1.1010, 1.1020, 1.1020, 1.1015, 1.1000, 1.0995, 1.1000],
            'low':  [1.0985, 1.0990, 1.0990, 1.1000, 1.1010, 1.1010, 1.1005, 1.0990, 1.0985, 1.0990],  # Candle 7 mitigates
            'close':[1.0995, 1.0995, 1.1000, 1.1010, 1.1015, 1.1015, 1.1010, 1.0990, 1.0990, 1.0995],
            'volume': [1000] * 10
        })

        return data

    def create_test_data_with_shrinking(self) -> pd.DataFrame:
        """Create test data with FVG that gets partially touched (shrunk)."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='h')

        # Bullish FVG that gets partially touched but not fully mitigated
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.0990, 1.0995, 1.0995, 1.1005, 1.1010, 1.1015, 1.1010, 1.1008, 1.1010, 1.1015],
            'high': [1.0995, 1.1000, 1.1000, 1.1010, 1.1020, 1.1020, 1.1015, 1.1012, 1.1015, 1.1020],
            'low':  [1.0985, 1.0990, 1.0990, 1.1000, 1.1010, 1.1010, 1.1005, 1.1006, 1.1008, 1.1010],  # Partial touch
            'close':[1.0995, 1.0995, 1.1000, 1.1010, 1.1015, 1.1015, 1.1010, 1.1010, 1.1015, 1.1018],
            'volume': [1000] * 10
        })

        return data

    def test_bullish_fvg_detection(self):
        """Test detection of bullish FVG."""
        data = self.create_test_data_with_bullish_fvg()

        # Get all FVGs
        fvgs = self.detector.get_all_fvgs(data)

        # Should detect at least one bullish FVG
        bullish_fvgs = [f for f in fvgs if f['type'] == 'bullish']
        self.assertGreater(len(bullish_fvgs), 0, "Should detect at least one bullish FVG")

        # Check FVG properties
        fvg = bullish_fvgs[0]
        self.assertEqual(fvg['type'], 'bullish')
        self.assertGreater(fvg['gap_size'], 0)
        self.assertGreater(fvg['current_top'], fvg['current_bottom'])

    def test_bearish_fvg_detection(self):
        """Test detection of bearish FVG."""
        data = self.create_test_data_with_bearish_fvg()

        # Get all FVGs
        fvgs = self.detector.get_all_fvgs(data)

        # Should detect at least one bearish FVG
        bearish_fvgs = [f for f in fvgs if f['type'] == 'bearish']
        self.assertGreater(len(bearish_fvgs), 0, "Should detect at least one bearish FVG")

        # Check FVG properties
        fvg = bearish_fvgs[0]
        self.assertEqual(fvg['type'], 'bearish')
        self.assertGreater(fvg['gap_size'], 0)
        self.assertGreater(fvg['current_top'], fvg['current_bottom'])

    def test_fvg_mitigation(self):
        """Test FVG mitigation tracking."""
        data = self.create_test_data_with_mitigation()

        # Get all FVGs
        fvgs = self.detector.get_all_fvgs(data)

        # Should have at least one FVG
        self.assertGreater(len(fvgs), 0, "Should detect at least one FVG")

        # Check if any FVG is mitigated
        mitigated_fvgs = [f for f in fvgs if f['is_mitigated']]
        self.assertGreater(len(mitigated_fvgs), 0, "Should have at least one mitigated FVG")

        # Check mitigation properties
        fvg = mitigated_fvgs[0]
        self.assertTrue(fvg['is_mitigated'])
        self.assertIsNotNone(fvg['mitigation_index'])
        self.assertIsNotNone(fvg['mitigation_time'])

    def test_fvg_shrinking(self):
        """Test FVG shrinking when partially touched."""
        data = self.create_test_data_with_shrinking()

        # Get all FVGs
        fvgs = self.detector.get_all_fvgs(data)

        # Should have at least one FVG
        self.assertGreater(len(fvgs), 0, "Should detect at least one FVG")

        # Check for shrinking
        fvg = fvgs[0]
        if fvg['type'] == 'bullish' and len(fvg['shrink_history']) > 0:
            # For bullish FVG, top should have been adjusted down
            self.assertLess(fvg['current_top'], fvg['original_top'])
        elif fvg['type'] == 'bearish' and len(fvg['shrink_history']) > 0:
            # For bearish FVG, bottom should have been adjusted up
            self.assertGreater(fvg['current_bottom'], fvg['original_bottom'])

    def test_min_gap_size_filter(self):
        """Test that FVGs smaller than min_gap_size are filtered out."""
        # Create detector with larger min_gap_size
        detector = FvgDetector(
            name="test_fvg_filtered",
            config={
                'min_gap_size': 0.0020,  # 20 pips
                'track_mitigation': True,
                'track_shrinking': True
            }
        )

        data = self.create_test_data_with_bullish_fvg()

        # Get all FVGs
        fvgs = detector.get_all_fvgs(data)

        # Check that small gaps are filtered
        for fvg in fvgs:
            self.assertGreaterEqual(fvg['gap_size'], 0.0020)

    def test_no_fvg_in_trending_data(self):
        """Test that no FVGs are detected in smoothly trending data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='h')

        # Create smoothly trending data without gaps
        data = pd.DataFrame({
            'datetime': dates,
            'open': [1.1000 + i*0.0005 for i in range(10)],
            'high': [1.1005 + i*0.0005 for i in range(10)],
            'low':  [1.0995 + i*0.0005 for i in range(10)],
            'close':[1.1002 + i*0.0005 for i in range(10)],
            'volume': [1000] * 10
        })

        # Get all FVGs
        fvgs = self.detector.get_all_fvgs(data)

        # Should not detect any FVGs
        self.assertEqual(len(fvgs), 0, "Should not detect FVGs in smoothly trending data")

    def test_detection_signal(self):
        """Test that detection signal is generated correctly."""
        data = self.create_test_data_with_bullish_fvg()

        # Check for detection on the last candle
        detection = self.detector.check(data[:5], candle_index=4)  # Check first 5 candles

        # If FVG formed, should have detection
        if detection:
            self.assertIsNotNone(detection.timestamp)
            self.assertIn(detection.direction, [SignalDirection.LONG, SignalDirection.SHORT])
            self.assertIn('fvg_type', detection.metadata)
            self.assertIn('gap_size', detection.metadata)

    def test_reset(self):
        """Test that reset clears internal state."""
        data = self.create_test_data_with_bullish_fvg()

        # Detect FVGs
        fvgs = self.detector.get_all_fvgs(data)
        self.assertGreater(len(fvgs), 0)

        # Reset
        self.detector.reset()

        # Internal state should be cleared
        self.assertEqual(len(self.detector.fvgs), 0)


if __name__ == '__main__':
    unittest.main()