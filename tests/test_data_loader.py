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

from core.data_loader import DataLoader, TimeframeConverter


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


class TestTimeframeConverter(unittest.TestCase):
    """Test suite for TimeframeConverter utility class."""

    def test_get_timeframe_ratio(self):
        """Test timeframe ratio calculation."""
        # Test common ratios
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('H4', 'H1'), 4.0)
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('H1', 'M15'), 4.0)
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('D1', 'H4'), 6.0)
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('M15', 'M5'), 3.0)

        # Test reverse ratios
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('H1', 'H4'), 0.25)
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('M5', 'M15'), 1/3)

        # Test same timeframe
        self.assertEqual(TimeframeConverter.get_timeframe_ratio('H1', 'H1'), 1.0)

        # Test invalid timeframes
        with self.assertRaises(ValueError):
            TimeframeConverter.get_timeframe_ratio('INVALID', 'H1')

        with self.assertRaises(ValueError):
            TimeframeConverter.get_timeframe_ratio('H1', 'INVALID')

    def test_convert_timeframe_index(self):
        """Test timeframe index conversion."""
        # H4 to H1 conversion
        self.assertEqual(TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'start'), 400)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'middle'), 402)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'end'), 403)

        # H1 to H4 conversion
        self.assertEqual(TimeframeConverter.convert_timeframe_index(400, 'H1', 'H4'), 100)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(401, 'H1', 'H4'), 100)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(403, 'H1', 'H4'), 100)

        # M15 to M5 conversion
        self.assertEqual(TimeframeConverter.convert_timeframe_index(10, 'M15', 'M5', 'start'), 30)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(10, 'M15', 'M5', 'middle'), 31)
        self.assertEqual(TimeframeConverter.convert_timeframe_index(10, 'M15', 'M5', 'end'), 32)

        # Test invalid position
        with self.assertRaises(ValueError):
            TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'invalid')

    def test_h4_to_h1_index(self):
        """Test specific H4 to H1 conversion methods."""
        # Test start position
        self.assertEqual(TimeframeConverter.h4_to_h1_index(0, 'start'), 0)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(1, 'start'), 4)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(100, 'start'), 400)

        # Test middle position
        self.assertEqual(TimeframeConverter.h4_to_h1_index(0, 'middle'), 2)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(1, 'middle'), 6)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(100, 'middle'), 402)

        # Test end position
        self.assertEqual(TimeframeConverter.h4_to_h1_index(0, 'end'), 3)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(1, 'end'), 7)
        self.assertEqual(TimeframeConverter.h4_to_h1_index(100, 'end'), 403)

    def test_h1_to_h4_index(self):
        """Test specific H1 to H4 conversion method."""
        # Test various H1 indices that map to same H4
        self.assertEqual(TimeframeConverter.h1_to_h4_index(400), 100)
        self.assertEqual(TimeframeConverter.h1_to_h4_index(401), 100)
        self.assertEqual(TimeframeConverter.h1_to_h4_index(402), 100)
        self.assertEqual(TimeframeConverter.h1_to_h4_index(403), 100)

        # Test boundary cases
        self.assertEqual(TimeframeConverter.h1_to_h4_index(0), 0)
        self.assertEqual(TimeframeConverter.h1_to_h4_index(4), 1)
        self.assertEqual(TimeframeConverter.h1_to_h4_index(8), 2)

    def test_get_equivalent_candle_range(self):
        """Test getting equivalent candle ranges."""
        # H4 to H1 range
        start, end = TimeframeConverter.get_equivalent_candle_range(100, 'H4', 'H1')
        self.assertEqual(start, 400)
        self.assertEqual(end, 403)

        # H1 to M15 range
        start, end = TimeframeConverter.get_equivalent_candle_range(10, 'H1', 'M15')
        self.assertEqual(start, 40)
        self.assertEqual(end, 43)

        # Test edge case with index 0
        start, end = TimeframeConverter.get_equivalent_candle_range(0, 'H4', 'H1')
        self.assertEqual(start, 0)
        self.assertEqual(end, 3)

    def test_validate_timeframe_alignment(self):
        """Test timeframe alignment validation."""
        # Create aligned test data
        dates_h1 = pd.date_range('2024-01-01', periods=400, freq='h')
        h1_data = pd.DataFrame({
            'open': np.random.randn(400) + 1.10,
            'high': np.random.randn(400) + 1.11,
            'low': np.random.randn(400) + 1.09,
            'close': np.random.randn(400) + 1.105,
            'volume': np.random.randint(100, 1000, 400)
        }, index=dates_h1)

        dates_h4 = pd.date_range('2024-01-01', periods=100, freq='4h')
        h4_data = pd.DataFrame({
            'open': np.random.randn(100) + 1.10,
            'high': np.random.randn(100) + 1.11,
            'low': np.random.randn(100) + 1.09,
            'close': np.random.randn(100) + 1.105,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates_h4)

        # Test perfect alignment
        result = TimeframeConverter.validate_timeframe_alignment(h1_data, 'H1', h4_data, 'H4')
        self.assertTrue(result['aligned'])
        self.assertEqual(result['ratio'], 0.25)  # H1 to H4 ratio
        self.assertEqual(result['data1_length'], 400)
        self.assertEqual(result['data2_length'], 100)
        self.assertEqual(len(result['issues']), 0)

        # Test misaligned data
        misaligned_h4 = h4_data.iloc[:60]  # Remove more candles to create significant misalignment
        result = TimeframeConverter.validate_timeframe_alignment(h1_data, 'H1', misaligned_h4, 'H4')
        self.assertFalse(result['aligned'])
        self.assertGreater(len(result['issues']), 0)


class TestMultiTimeframeDataLoader(unittest.TestCase):
    """Test suite for multi-timeframe DataLoader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader(cache_enabled=True)

        # Create comprehensive test data
        dates = pd.date_range('2024-01-01', periods=1000, freq='h')
        self.sample_data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.randn(1000) * 0.01 + 1.10,
            'high': np.random.randn(1000) * 0.01 + 1.11,
            'low': np.random.randn(1000) * 0.01 + 1.09,
            'close': np.random.randn(1000) * 0.01 + 1.105,
            'volume': np.random.randint(100, 5000, 1000)
        })

        # Fix high/low relationships
        self.sample_data['high'] = self.sample_data[['open', 'high', 'close']].max(axis=1)
        self.sample_data['low'] = self.sample_data[['open', 'low', 'close']].min(axis=1)

        # Create temporary file and load data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file, index=False)
        self.temp_file.close()

        self.loader.load_csv(self.temp_file.name, symbol='EURUSD')

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_load_multi_timeframe_data(self):
        """Test loading multiple timeframes simultaneously."""
        # Test default timeframes
        multi_data = self.loader.load_multi_timeframe_data('EURUSD')
        self.assertIn('H4', multi_data)
        self.assertIn('H1', multi_data)

        # Test custom timeframes
        timeframes = ['H4', 'H1', 'M30', 'M15']
        multi_data = self.loader.load_multi_timeframe_data('EURUSD', timeframes)

        for tf in timeframes:
            self.assertIn(tf, multi_data)
            self.assertIsInstance(multi_data[tf], pd.DataFrame)
            self.assertGreater(len(multi_data[tf]), 0)

        # Test with date filtering
        multi_data_filtered = self.loader.load_multi_timeframe_data(
            'EURUSD', ['H4', 'H1'],
            start='2024-01-15',
            end='2024-01-25'
        )

        # Check that filtering was applied
        for tf in ['H4', 'H1']:
            self.assertGreaterEqual(multi_data_filtered[tf].index[0], pd.to_datetime('2024-01-15'))
            self.assertLessEqual(multi_data_filtered[tf].index[-1], pd.to_datetime('2024-01-25'))

        # Test with non-loaded symbol
        with self.assertRaises(ValueError):
            self.loader.load_multi_timeframe_data('GBPUSD')

    def test_sync_timeframe_data(self):
        """Test timeframe data synchronization."""
        multi_data = self.loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1', 'M30'])

        # Test auto-selection of reference timeframe
        synced = self.loader.sync_timeframe_data(multi_data)

        # H4 should be selected as reference (largest timeframe)
        self.assertIn('H4', synced)
        self.assertIn('H1', synced)
        self.assertIn('M30', synced)

        # Test explicit reference timeframe
        synced_h1 = self.loader.sync_timeframe_data(multi_data, reference_timeframe='H1')
        self.assertEqual(len(synced_h1['H1']), len(multi_data['H1']))  # Reference unchanged

        # Test empty data
        empty_synced = self.loader.sync_timeframe_data({})
        self.assertEqual(len(empty_synced), 0)

        # Test invalid reference timeframe
        with self.assertRaises(ValueError):
            self.loader.sync_timeframe_data(multi_data, reference_timeframe='INVALID')

    def test_get_timeframe_mapping(self):
        """Test creating timeframe index mapping."""
        multi_data = self.loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1'])

        mapping = self.loader.get_timeframe_mapping(multi_data, 'H4', 'H1')

        # Check mapping structure
        self.assertIsInstance(mapping, dict)
        self.assertGreater(len(mapping), 0)

        # Test some mapping values make sense
        for h4_idx, h1_idx in list(mapping.items())[:10]:  # Check first 10 mappings
            self.assertIsInstance(h4_idx, int)
            self.assertIsInstance(h1_idx, int)
            self.assertGreaterEqual(h4_idx, 0)
            self.assertGreaterEqual(h1_idx, 0)

        # Test invalid timeframes
        with self.assertRaises(ValueError):
            self.loader.get_timeframe_mapping(multi_data, 'INVALID', 'H1')

    def test_find_equivalent_candle(self):
        """Test finding equivalent candles across timeframes."""
        multi_data = self.loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1'])

        # Test H4 to H1 conversion
        h1_idx = self.loader.find_equivalent_candle(multi_data, 'H4', 10, 'H1', 'start')
        self.assertIsNotNone(h1_idx)
        self.assertIsInstance(h1_idx, int)
        self.assertGreaterEqual(h1_idx, 0)

        # Test different positions
        h1_start = self.loader.find_equivalent_candle(multi_data, 'H4', 10, 'H1', 'start')
        h1_middle = self.loader.find_equivalent_candle(multi_data, 'H4', 10, 'H1', 'middle')
        h1_end = self.loader.find_equivalent_candle(multi_data, 'H4', 10, 'H1', 'end')

        self.assertLessEqual(h1_start, h1_middle)
        self.assertLessEqual(h1_middle, h1_end)

        # Test reverse conversion (H1 to H4)
        h4_idx = self.loader.find_equivalent_candle(multi_data, 'H1', 40, 'H4')
        self.assertIsNotNone(h4_idx)

        # Test invalid cases
        result = self.loader.find_equivalent_candle(multi_data, 'INVALID', 10, 'H1')
        self.assertIsNone(result)

        result = self.loader.find_equivalent_candle(multi_data, 'H4', 999999, 'H1')
        self.assertIsNone(result)

    def test_timeframe_data_consistency(self):
        """Test that multi-timeframe data maintains consistency."""
        multi_data = self.loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1'])

        h4_data = multi_data['H4']
        h1_data = multi_data['H1']

        # Basic consistency checks
        self.assertGreater(len(h1_data), len(h4_data))  # H1 should have more candles

        # Check that timeframes are roughly aligned (allowing for some variance due to resampling)
        expected_ratio = TimeframeConverter.get_timeframe_ratio('H4', 'H1')
        actual_ratio = len(h1_data) / len(h4_data)

        # Allow 20% variance for real-world data imperfections
        self.assertGreater(actual_ratio, expected_ratio * 0.8)
        self.assertLess(actual_ratio, expected_ratio * 1.2)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very small dataset
        small_dates = pd.date_range('2024-01-01', periods=10, freq='h')
        small_data = pd.DataFrame({
            'datetime': small_dates,
            'open': [1.10] * 10,
            'high': [1.11] * 10,
            'low': [1.09] * 10,
            'close': [1.105] * 10,
            'volume': [1000] * 10
        })

        small_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        small_data.to_csv(small_temp, index=False)
        small_temp.close()

        try:
            small_loader = DataLoader()
            small_loader.load_csv(small_temp.name, symbol='SMALL')

            multi_data = small_loader.load_multi_timeframe_data('SMALL', ['H4', 'H1'])

            # Should handle small datasets gracefully
            self.assertIn('H4', multi_data)
            self.assertIn('H1', multi_data)

            # H4 might have very few or zero candles
            self.assertGreaterEqual(len(multi_data['H4']), 0)
            self.assertGreater(len(multi_data['H1']), 0)

        finally:
            os.unlink(small_temp.name)


if __name__ == '__main__':
    unittest.main()