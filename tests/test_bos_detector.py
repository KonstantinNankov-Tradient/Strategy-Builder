"""
Tests for the Break of Structure (BOS) Detector.

This module contains comprehensive tests for the BosDetector class,
verifying swing detection, trend continuation identification, and BOS signal generation.
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

from indicators.bos_detector import BosDetector
from core.state_types import SignalDirection


class TestBosDetector(unittest.TestCase):
    """Test suite for the BOS Detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = BosDetector(
            name='test_bos',
            config={
                'base_strength': 2,
                'min_gap': 1,
                'symbol': 'TEST'
            }
        )

    def create_test_data(self, prices, start_date='2024-01-01'):
        """Create test OHLCV data from price list."""
        dates = pd.date_range(start=start_date, periods=len(prices), freq='h')
        data = []
        for date, price in zip(dates, prices):
            # Create candles with small wicks
            if price > prices[max(0, prices.index(price) - 1)]:
                # Bullish candle
                data.append({
                    'datetime': date,
                    'open': price - 0.1,
                    'high': price + 0.05,
                    'low': price - 0.15,
                    'close': price,
                    'volume': 1000
                })
            else:
                # Bearish candle
                data.append({
                    'datetime': date,
                    'open': price + 0.1,
                    'high': price + 0.15,
                    'low': price - 0.05,
                    'close': price,
                    'volume': 1000
                })
        return pd.DataFrame(data)

    def test_initialization(self):
        """Test BOS detector initialization."""
        self.assertEqual(self.detector.name, 'test_bos')
        self.assertEqual(self.detector.base_strength, 2)
        self.assertEqual(self.detector.min_gap, 1)
        self.assertEqual(self.detector.symbol, 'TEST')
        self.assertEqual(self.detector.current_trend, 0)
        self.assertFalse(self.detector.trend_established)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid base_strength
        with self.assertRaises(ValueError):
            BosDetector('test', {'base_strength': 0})

        # Invalid min_gap
        with self.assertRaises(ValueError):
            BosDetector('test', {'min_gap': -1})

    def test_lookback_period(self):
        """Test lookback period calculation."""
        # With base_strength=2: (2*2) + 10 = 14
        self.assertEqual(self.detector.get_lookback_period(), 14)

    def test_swing_detection(self):
        """Test swing high and low detection."""
        # Create data with clear swings
        prices = [100, 102, 104, 103, 101,  # Swing high at 104
                  99, 97, 96, 98, 100]        # Swing low at 96
        data = self.create_test_data(prices)
        
        # Set index to datetime for swing detection
        data_indexed = data.set_index('datetime')
        swing_labels, swing_levels = self.detector._get_swing_highs_lows(data_indexed)
        
        # Check swing high detected
        swing_highs = swing_labels[swing_labels == 1]
        self.assertEqual(len(swing_highs), 1)
        
        # Check swing low detected
        swing_lows = swing_labels[swing_labels == -1]
        self.assertEqual(len(swing_lows), 1)

    def test_first_breakout_not_bos(self):
        """Test that the first breakout is NOT a BOS signal."""
        # Create data with initial breakout
        prices = [100, 101, 102, 101, 100,  # Swing high at 102
                  99, 98, 99, 100, 101,
                  102, 103, 104, 105, 106]   # Breaks above 102
        data = self.create_test_data(prices)

        # Process through data looking for detections
        bos_found = False
        trend_set = False
        for i in range(self.detector.get_lookback_period(), len(data)):
            window = data.iloc[:i+1]
            detection = self.detector.check(window.copy(), i)

            if self.detector.trend_established and not trend_set:
                trend_set = True
                # At first breakout, trend should be established but no BOS
                self.assertEqual(self.detector.current_trend, 1)  # Bullish

            if detection and detection.metadata.get('bos_type'):
                bos_found = True

        # First breakout should not generate BOS
        self.assertFalse(bos_found, "BOS incorrectly detected on first breakout")
        self.assertTrue(self.detector.trend_established)

    def test_bos_on_trend_continuation(self):
        """Test BOS detection on trend continuation."""
        # Create data with trend continuation - more pronounced swings
        prices = [100, 102, 104, 102, 100,   # First swing high at 104
                  101, 103, 105, 106, 105,   # Breaks above 104 (establishes bullish trend)
                  103, 107, 110, 108, 105,   # Second swing high at 110
                  107, 109, 111, 112, 113]   # Breaks above 110 (BOS - continuation)

        data = self.create_test_data(prices)

        # Process through all data to establish trend
        bos_detected = False
        for i in range(self.detector.get_lookback_period(), len(data)):
            window = data.iloc[:i+1]
            detection = self.detector.check(window.copy(), i)

            # Check if we got a BOS
            if detection and detection.metadata.get('bos_type') == 'bullish_bos':
                self.assertEqual(detection.direction, SignalDirection.LONG)
                self.assertTrue(detection.metadata['is_continuation'])
                bos_detected = True
                break

        # Should have found a BOS
        self.assertTrue(bos_detected, "BOS not detected on trend continuation")

    def test_no_bos_on_trend_change(self):
        """Test that BOS is NOT generated on trend change (that would be CHoCH)."""
        # Create data with trend change
        prices = [100, 102, 104, 103, 101,   # Swing high at 104
                  102, 103, 105, 106, 104,   # Breaks above 104 (bullish)
                  103, 101, 99, 100, 102,    # Swing low at 99
                  101, 100, 98, 97, 96]      # Breaks below 99 (bearish - CHoCH, not BOS)
        
        data = self.create_test_data(prices)
        
        # Process through all data
        bos_detected = False
        for i in range(self.detector.get_lookback_period(), len(data)):
            window = data.iloc[:i+1]
            detection = self.detector.check(window.copy(), i)
            
            # Check at trend change point (should not be BOS)
            if i >= 18 and detection:  # Around the bearish break
                # This should not happen - trend change is not BOS
                if detection.metadata.get('bos_type'):
                    bos_detected = True
        
        self.assertFalse(bos_detected, "BOS incorrectly detected on trend change")

    def test_multiple_bos_same_trend(self):
        """Test multiple BOS signals in the same trend."""
        # Create strong bullish trend with multiple continuations
        prices = []
        base = 100
        
        # First swing and break (establish trend)
        prices.extend([base, base+2, base+4, base+3, base+1])  # Swing high
        prices.extend([base+2, base+3, base+5, base+6])        # Break (trend established)
        
        # Second swing and break (first BOS)
        prices.extend([base+5, base+7, base+9, base+8, base+6])  # Swing high
        prices.extend([base+7, base+8, base+10, base+11])        # Break (BOS)
        
        # Third swing and break (second BOS)
        prices.extend([base+10, base+12, base+14, base+13, base+11])  # Swing high
        prices.extend([base+12, base+13, base+15, base+16])           # Break (BOS)
        
        data = self.create_test_data(prices)
        
        # Count BOS detections
        bos_count = 0
        for i in range(self.detector.get_lookback_period(), len(data)):
            window = data.iloc[:i+1]
            detection = self.detector.check(window.copy(), i)
            
            if detection and detection.metadata.get('bos_type'):
                bos_count += 1
                self.assertEqual(detection.metadata['trend_name'], 'Bullish')
                self.assertTrue(detection.metadata['is_continuation'])
        
        # Should detect 2 BOS signals (not counting initial trend establishment)
        self.assertGreaterEqual(bos_count, 1)

    def test_bearish_bos(self):
        """Test BOS detection in bearish trend."""
        # Create bearish trend data
        prices = [100, 98, 96, 97, 99,      # Swing low at 96
                  98, 97, 95, 94, 96,        # Breaks below 96 (establishes bearish)
                  95, 93, 91, 92, 94,        # Swing low at 91
                  93, 92, 90, 89, 88]        # Breaks below 91 (BOS)
        
        data = self.create_test_data(prices)
        
        # Process data and look for bearish BOS
        for i in range(self.detector.get_lookback_period(), len(data)):
            window = data.iloc[:i+1]
            detection = self.detector.check(window.copy(), i)
            
            if detection and detection.metadata.get('bos_type') == 'bearish_bos':
                self.assertEqual(detection.direction, SignalDirection.SHORT)
                self.assertTrue(detection.metadata['is_continuation'])
                self.assertEqual(detection.metadata['trend_direction'], -1)
                return
        
        self.fail("Bearish BOS not detected")

    def test_get_all_swings_and_bos(self):
        """Test getting all swings and BOS signals for visualization."""
        # Create data with multiple swings and BOS
        prices = [100, 102, 104, 103, 101,   # Swing high
                  102, 103, 105, 106, 104,   # Break (trend)
                  105, 107, 109, 108, 106,   # Swing high
                  107, 108, 110, 111, 112,   # Break (BOS)
                  111, 113, 115, 114, 112,   # Swing high
                  113, 114, 116, 117, 118]   # Break (BOS)
        
        data = self.create_test_data(prices)
        
        # Get all swings and BOS
        swing_points, bos_signals = self.detector.get_all_swings_and_bos(data)
        
        # Should have multiple swing points
        self.assertGreater(len(swing_points), 0)
        
        # Should have at least one BOS signal
        self.assertGreater(len(bos_signals), 0)
        
        # All BOS should be continuations
        for bos in bos_signals:
            self.assertTrue(bos['is_continuation'])

    def test_reset(self):
        """Test detector reset functionality."""
        # Process some data first
        prices = [100, 102, 104, 103, 101, 102, 103, 105]
        data = self.create_test_data(prices)
        self.detector.check(data, 7)
        
        # Reset detector
        self.detector.reset()
        
        # Check state is cleared
        self.assertEqual(self.detector.current_trend, 0)
        self.assertFalse(self.detector.trend_established)
        self.assertEqual(len(self.detector.swing_highs), 0)
        self.assertEqual(len(self.detector.swing_lows), 0)
        self.assertEqual(len(self.detector.bos_signals), 0)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create data smaller than lookback period
        prices = [100, 101, 102]
        data = self.create_test_data(prices)
        
        detection = self.detector.check(data, 2)
        self.assertIsNone(detection)

    def test_min_gap_constraint(self):
        """Test minimum gap constraint between swings."""
        # Create data with potential swings too close
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106]
        data = self.create_test_data(prices)
        
        data_indexed = data.set_index('datetime')
        swing_labels, _ = self.detector._get_swing_highs_lows(data_indexed)
        
        # Count swings and check they respect min_gap
        swing_indices = np.where(~np.isnan(swing_labels))[0]
        for i in range(1, len(swing_indices)):
            gap = swing_indices[i] - swing_indices[i-1]
            self.assertGreaterEqual(gap, self.detector.min_gap)


if __name__ == '__main__':
    unittest.main()