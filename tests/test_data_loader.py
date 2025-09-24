"""
Comprehensive test suite for DataLoader class.

Tests cover:
- CSV loading and parsing
- Data validation
- Timeframe resampling
- Data caching
- Window-based data access
- Streaming functionality
- Error handling
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test suite for DataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader(cache_enabled=True)
        # Get the parent directory (strategy_builder) and then the data path
        self.test_data_path = Path(__file__).parent.parent / 'data' / 'EURUSD_20200101_20250809.csv'

    def test_initialization(self):
        """Test DataLoader initialization."""
        # Test with cache enabled
        loader = DataLoader(cache_enabled=True)
        self.assertTrue(loader.cache_enabled)
        self.assertEqual(len(loader._data_cache), 0)
        self.assertEqual(len(loader._timeframe_cache), 0)

        # Test with cache disabled
        loader = DataLoader(cache_enabled=False)
        self.assertFalse(loader.cache_enabled)

    def test_load_csv_success(self):
        """Test successful CSV loading."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        df = self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertListEqual(
            list(df.columns),
            ['open', 'high', 'low', 'close', 'volume']
        )

        # Check that data is cached
        self.assertIn('EURUSD', self.loader._data_cache)
        self.assertIn('EURUSD_H1', self.loader._timeframe_cache)

    def test_load_csv_missing_file(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_csv('nonexistent.csv')

    def test_data_validation(self):
        """Test data validation logic."""
        # Create test data with valid OHLC
        dates = pd.date_range('2024-01-01', periods=5, freq='h')
        valid_data = pd.DataFrame({
            'open': [1.10, 1.11, 1.12, 1.13, 1.14],
            'high': [1.11, 1.12, 1.13, 1.14, 1.15],
            'low': [1.09, 1.10, 1.11, 1.12, 1.13],
            'close': [1.105, 1.115, 1.125, 1.135, 1.145],
            'volume': [100, 200, 300, 400, 500]
        }, index=dates)

        # Should pass validation
        self.loader._validate_data(valid_data)

        # Test invalid high < low
        invalid_data = valid_data.copy()
        invalid_data.loc[invalid_data.index[0], 'high'] = 1.08  # Less than low

        # Should log warning but not raise
        self.loader._validate_data(invalid_data)

        # Test missing values
        invalid_data = valid_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = np.nan

        with self.assertRaises(ValueError) as context:
            self.loader._validate_data(invalid_data)
        self.assertIn("missing values", str(context.exception))

        # Test non-chronological order
        invalid_data = valid_data.copy()
        invalid_data = invalid_data.sort_index(ascending=False)

        with self.assertRaises(ValueError) as context:
            self.loader._validate_data(invalid_data)
        self.assertIn("chronological order", str(context.exception))

    def test_resample_timeframe(self):
        """Test timeframe resampling."""
        # Create hourly test data
        dates = pd.date_range('2024-01-01', periods=24, freq='h')
        hourly_data = pd.DataFrame({
            'open': np.random.randn(24) + 1.10,
            'high': np.random.randn(24) + 1.11,
            'low': np.random.randn(24) + 1.09,
            'close': np.random.randn(24) + 1.105,
            'volume': np.random.randint(100, 1000, 24)
        }, index=dates)

        # Ensure high/low relationships are correct
        hourly_data['high'] = hourly_data[['open', 'high', 'close']].max(axis=1)
        hourly_data['low'] = hourly_data[['open', 'low', 'close']].min(axis=1)

        # Test H1 to H4 resampling
        h4_data = self.loader.resample_timeframe(hourly_data, 'H4')

        # Check shape (24 hours / 4 = 6 H4 candles)
        self.assertEqual(len(h4_data), 6)

        # Verify aggregation rules
        first_4_hours = hourly_data.iloc[:4]
        self.assertEqual(h4_data.iloc[0]['open'], first_4_hours['open'].iloc[0])
        self.assertEqual(h4_data.iloc[0]['high'], first_4_hours['high'].max())
        self.assertEqual(h4_data.iloc[0]['low'], first_4_hours['low'].min())
        self.assertEqual(h4_data.iloc[0]['close'], first_4_hours['close'].iloc[-1])
        self.assertEqual(h4_data.iloc[0]['volume'], first_4_hours['volume'].sum())

        # Test H1 to D1 resampling
        d1_data = self.loader.resample_timeframe(hourly_data, 'D1')
        self.assertEqual(len(d1_data), 1)  # 24 hours = 1 day

        # Test invalid timeframe
        with self.assertRaises(ValueError):
            self.loader.resample_timeframe(hourly_data, 'INVALID')

    def test_get_data_with_filtering(self):
        """Test data retrieval with date filtering."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        # Load data first
        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Test basic retrieval
        df = self.loader.get_data('EURUSD', 'H1')
        self.assertIsInstance(df, pd.DataFrame)

        # Test with date filtering
        df_filtered = self.loader.get_data(
            'EURUSD', 'H1',
            start='2024-01-01',
            end='2024-12-31'
        )

        # Check that filtering worked
        self.assertGreaterEqual(df_filtered.index[0], pd.to_datetime('2024-01-01'))
        self.assertLessEqual(df_filtered.index[-1], pd.to_datetime('2024-12-31'))

        # Test retrieval of non-loaded symbol
        with self.assertRaises(ValueError):
            self.loader.get_data('GBPUSD', 'H1')

    def test_get_latest_candles(self):
        """Test retrieval of latest candles."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Get last 50 candles
        latest = self.loader.get_latest_candles('EURUSD', 'H1', count=50)
        self.assertEqual(len(latest), 50)

        # Verify it's actually the latest data
        all_data = self.loader.get_data('EURUSD', 'H1')
        pd.testing.assert_frame_equal(latest, all_data.tail(50))

    def test_get_window(self):
        """Test window-based data retrieval."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Test window retrieval
        window = self.loader.get_window(
            'EURUSD', 'H1',
            center_time='2024-06-01',
            lookback=100,
            lookforward=20
        )

        # Window should have at most 121 candles (100 back + 1 center + 20 forward)
        self.assertLessEqual(len(window), 121)

    def test_stream_data(self):
        """Test data streaming for backtesting."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Test streaming
        stream = self.loader.stream_data(
            'EURUSD', 'H1',
            start='2024-01-01',
            end='2024-01-31',
            lookback=50
        )

        # Collect first few windows
        windows = []
        for i, (timestamp, window) in enumerate(stream):
            windows.append((timestamp, window))
            if i >= 5:  # Get first 5 windows
                break

        # Verify windows
        self.assertEqual(len(windows), 6)

        for timestamp, window in windows:
            # Each window should have at most lookback + 1 candles
            self.assertLessEqual(len(window), 51)
            # Current timestamp should be the last in the window
            self.assertEqual(window.index[-1], timestamp)

    def test_cache_operations(self):
        """Test cache management."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        # Load data to populate cache
        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Get different timeframes to populate cache
        self.loader.get_data('EURUSD', 'H4')
        self.loader.get_data('EURUSD', 'D1')

        # Check cache is populated
        self.assertIn('EURUSD', self.loader._data_cache)
        self.assertIn('EURUSD_H1', self.loader._timeframe_cache)
        self.assertIn('EURUSD_H4', self.loader._timeframe_cache)
        self.assertIn('EURUSD_D1', self.loader._timeframe_cache)

        # Clear specific symbol cache
        self.loader.clear_cache('EURUSD')
        self.assertNotIn('EURUSD', self.loader._data_cache)
        self.assertNotIn('EURUSD_H1', self.loader._timeframe_cache)
        self.assertNotIn('EURUSD_H4', self.loader._timeframe_cache)

        # Reload and clear all cache
        self.loader.load_csv(self.test_data_path, symbol='EURUSD')
        self.loader.clear_cache()
        self.assertEqual(len(self.loader._data_cache), 0)
        self.assertEqual(len(self.loader._timeframe_cache), 0)

    def test_get_info(self):
        """Test information retrieval."""
        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        self.loader.load_csv(self.test_data_path, symbol='EURUSD')

        info = self.loader.get_info()

        # Check info structure
        self.assertIn('cache_enabled', info)
        self.assertIn('loaded_symbols', info)
        self.assertIn('cached_timeframes', info)
        self.assertIn('memory_usage_mb', info)
        self.assertIn('EURUSD_stats', info)

        # Verify values
        self.assertTrue(info['cache_enabled'])
        self.assertIn('EURUSD', info['loaded_symbols'])
        self.assertGreater(info['memory_usage_mb'], 0)

    def test_cache_disabled(self):
        """Test behavior with caching disabled."""
        loader = DataLoader(cache_enabled=False)

        if not self.test_data_path.exists():
            self.skipTest(f"Test data file not found: {self.test_data_path}")

        loader.load_csv(self.test_data_path, symbol='EURUSD')

        # Cache should be empty even after loading
        self.assertEqual(len(loader._data_cache), 0)
        self.assertEqual(len(loader._timeframe_cache), 0)

    def test_create_sample_data(self):
        """Test with programmatically created sample data."""
        # Create sample data for testing without file dependency
        dates = pd.date_range('2024-01-01', periods=1000, freq='h')
        sample_data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.randn(1000) * 0.01 + 1.10,
            'high': np.random.randn(1000) * 0.01 + 1.11,
            'low': np.random.randn(1000) * 0.01 + 1.09,
            'close': np.random.randn(1000) * 0.01 + 1.105,
            'volume': np.random.randint(100, 5000, 1000)
        })

        # Fix high/low relationships
        sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
        sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load the temporary file
            df = self.loader.load_csv(temp_path, symbol='TEST')

            # Verify loaded correctly
            self.assertEqual(len(df), 1000)
            self.assertIn('TEST', self.loader._data_cache)

            # Test various operations
            h4_data = self.loader.get_data('TEST', 'H4')
            self.assertGreater(len(h4_data), 0)

            latest = self.loader.get_latest_candles('TEST', 'H1', count=10)
            self.assertEqual(len(latest), 10)

        finally:
            # Clean up temporary file
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()