"""
DataLoader module for handling OHLCV data loading and management.

This module provides the DataLoader class which is responsible for:
- Loading CSV data files
- Multi-timeframe resampling
- Efficient data caching
- Data validation
- Streaming data for backtesting
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class DataLoader:
    """
    DataLoader handles loading, caching, and managing OHLCV data from CSV files.

    Key features:
    - Load CSV data with datetime parsing
    - Multi-timeframe support with resampling
    - Efficient caching mechanism
    - Data validation and integrity checks
    - Window-based data access for backtesting
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize DataLoader with optional caching.

        Args:
            cache_enabled: Enable in-memory caching for performance
        """
        self.cache_enabled = cache_enabled
        self._data_cache: Dict[str, pd.DataFrame] = {}  # Original data cache
        self._timeframe_cache: Dict[str, pd.DataFrame] = {}  # Resampled data cache
        self.logger = logging.getLogger(__name__)

        # Define supported timeframes and their pandas resample rules
        self.timeframe_map = {
            'M1': '1min',   # 1 minute
            'M5': '5min',   # 5 minutes
            'M15': '15min', # 15 minutes
            'M30': '30min', # 30 minutes
            'H1': '1h',     # 1 hour
            'H4': '4h',     # 4 hours
            'D1': '1D',     # 1 day
            'W1': '1W',     # 1 week
        }

    def load_csv(self, filepath: Union[str, Path], symbol: str = 'EURUSD') -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        The CSV should have columns: datetime, open, high, low, close, volume

        Args:
            filepath: Path to CSV file
            symbol: Symbol identifier for caching

        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        # Convert to Path object
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Read CSV with datetime parsing
        df = pd.read_csv(
            filepath,
            parse_dates=['datetime'],
            index_col='datetime'
        )

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data integrity
        self._validate_data(df)

        # Cache if enabled
        if self.cache_enabled:
            self._data_cache[symbol] = df
            cache_key = f"{symbol}_H1"  # Assume original data is H1
            self._timeframe_cache[cache_key] = df

        self.logger.info(f"Loaded {len(df)} rows for {symbol}")
        self.logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate OHLC data integrity.

        Checks:
        - High >= Low
        - High >= Open and Close
        - Low <= Open and Close
        - No missing values in OHLC
        - Chronological order
        """
        # Check high >= low
        invalid_hl = df[df['high'] < df['low']]
        if not invalid_hl.empty:
            self.logger.warning(f"Found {len(invalid_hl)} rows where high < low")

        # Check high/low relationships with open/close
        invalid_high = df[(df['high'] < df['open']) | (df['high'] < df['close'])]
        if not invalid_high.empty:
            self.logger.warning(f"Found {len(invalid_high)} rows with invalid high values")

        invalid_low = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
        if not invalid_low.empty:
            self.logger.warning(f"Found {len(invalid_low)} rows with invalid low values")

        # Check for missing values
        if df[['open', 'high', 'low', 'close']].isnull().any().any():
            raise ValueError("OHLC data contains missing values")

        # Check chronological order
        if not df.index.is_monotonic_increasing:
            raise ValueError("Data is not in chronological order")

    def resample_timeframe(self,
                          data: pd.DataFrame,
                          target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLC data to a different timeframe.

        Uses proper aggregation rules:
        - Open: First value in period
        - High: Maximum value in period
        - Low: Minimum value in period
        - Close: Last value in period
        - Volume: Sum of all volumes in period

        Args:
            data: Source DataFrame with OHLC data
            target_timeframe: Target timeframe (e.g., 'H4', 'D1')

        Returns:
            Resampled DataFrame
        """
        if target_timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        rule = self.timeframe_map[target_timeframe]

        # Resample with proper OHLC aggregation
        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove periods with no data
        resampled = resampled.dropna(how='all')

        return resampled

    def get_data(self,
                 symbol: str = 'EURUSD',
                 timeframe: str = 'H1',
                 start: Optional[Union[str, datetime]] = None,
                 end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get data for a specific symbol and timeframe.

        Args:
            symbol: Symbol to retrieve
            timeframe: Desired timeframe
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with requested data
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe}"

        if cache_key in self._timeframe_cache:
            df = self._timeframe_cache[cache_key]
        elif symbol in self._data_cache:
            # Resample from cached original data
            original = self._data_cache[symbol]
            df = self.resample_timeframe(original, timeframe)

            # Cache the resampled data
            if self.cache_enabled:
                self._timeframe_cache[cache_key] = df
        else:
            raise ValueError(f"No data loaded for symbol: {symbol}")

        # Apply date filtering if specified
        if start is not None:
            if isinstance(start, str):
                start = pd.to_datetime(start)
            df = df[df.index >= start]

        if end is not None:
            if isinstance(end, str):
                end = pd.to_datetime(end)
            df = df[df.index <= end]

        return df

    def get_latest_candles(self,
                          symbol: str = 'EURUSD',
                          timeframe: str = 'H1',
                          count: int = 100) -> pd.DataFrame:
        """
        Get the most recent N candles.

        Args:
            symbol: Symbol to retrieve
            timeframe: Desired timeframe
            count: Number of candles to return

        Returns:
            DataFrame with latest N candles
        """
        df = self.get_data(symbol, timeframe)
        return df.tail(count)

    def get_window(self,
                   symbol: str = 'EURUSD',
                   timeframe: str = 'H1',
                   center_time: Union[str, datetime] = None,
                   lookback: int = 100,
                   lookforward: int = 0) -> pd.DataFrame:
        """
        Get a window of data around a specific time.

        Useful for backtesting when you need historical context
        but don't want future data leakage.

        Args:
            symbol: Symbol to retrieve
            timeframe: Desired timeframe
            center_time: Center point for the window
            lookback: Number of candles before center_time
            lookforward: Number of candles after center_time

        Returns:
            DataFrame with windowed data
        """
        df = self.get_data(symbol, timeframe)

        if center_time is None:
            center_time = df.index[-1]
        elif isinstance(center_time, str):
            center_time = pd.to_datetime(center_time)

        # Find the closest index to center_time
        idx = df.index.get_indexer([center_time], method='nearest')[0]

        # Calculate window bounds
        start_idx = max(0, idx - lookback)
        end_idx = min(len(df), idx + lookforward + 1)

        return df.iloc[start_idx:end_idx]

    def stream_data(self,
                    symbol: str = 'EURUSD',
                    timeframe: str = 'H1',
                    start: Optional[Union[str, datetime]] = None,
                    end: Optional[Union[str, datetime]] = None,
                    lookback: int = 100):
        """
        Generator that streams data for backtesting.

        Yields data windows that simulate real-time data arrival,
        ensuring no future data leakage.

        Args:
            symbol: Symbol to stream
            timeframe: Timeframe to use
            start: Start date for streaming
            end: End date for streaming
            lookback: Number of historical candles to include

        Yields:
            Tuple of (current_time, data_window)
        """
        df = self.get_data(symbol, timeframe, start, end)

        for i in range(lookback, len(df)):
            current_time = df.index[i]
            # Window includes current candle and lookback history
            window = df.iloc[max(0, i - lookback):i + 1]
            yield current_time, window

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            # Clear specific symbol
            self._data_cache.pop(symbol, None)
            # Clear all timeframe caches for this symbol
            keys_to_remove = [k for k in self._timeframe_cache if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                self._timeframe_cache.pop(key, None)
        else:
            # Clear all caches
            self._data_cache.clear()
            self._timeframe_cache.clear()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.

        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_enabled': self.cache_enabled,
            'loaded_symbols': list(self._data_cache.keys()),
            'cached_timeframes': list(self._timeframe_cache.keys()),
            'memory_usage_mb': sum(
                df.memory_usage(deep=True).sum() / 1024**2
                for df in self._data_cache.values()
            )
        }

        for symbol, df in self._data_cache.items():
            info[f'{symbol}_stats'] = {
                'rows': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}",
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
            }

        return info

    def load_multi_timeframe_data(self,
                                 symbol: str = 'EURUSD',
                                 timeframes: List[str] = None,
                                 start: Optional[Union[str, datetime]] = None,
                                 end: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple timeframes simultaneously for a symbol.

        Args:
            symbol: Symbol to load
            timeframes: List of timeframes to load (e.g., ['H4', 'H1', 'M15'])
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Dictionary mapping timeframes to DataFrames

        Example:
            >>> loader = DataLoader()
            >>> data = loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1'])
            >>> print(f"H4 candles: {len(data['H4'])}, H1 candles: {len(data['H1'])}")
        """
        if timeframes is None:
            timeframes = ['H4', 'H1']

        if symbol not in self._data_cache:
            raise ValueError(f"No data loaded for symbol: {symbol}")

        multi_data = {}

        for timeframe in timeframes:
            try:
                data = self.get_data(symbol, timeframe, start, end)
                multi_data[timeframe] = data
                self.logger.info(f"Loaded {len(data)} {timeframe} candles for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to load {timeframe} data for {symbol}: {e}")
                raise

        return multi_data

    def sync_timeframe_data(self,
                           multi_data: Dict[str, pd.DataFrame],
                           reference_timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """
        Synchronize multiple timeframes to ensure proper alignment.

        Args:
            multi_data: Dictionary of timeframe DataFrames
            reference_timeframe: Timeframe to use as reference (None = auto-select)

        Returns:
            Dictionary of synchronized timeframe DataFrames

        Example:
            >>> data = loader.load_multi_timeframe_data('EURUSD', ['H4', 'H1'])
            >>> synced = loader.sync_timeframe_data(data, 'H4')
        """
        if not multi_data:
            return multi_data

        # Auto-select reference timeframe (largest available)
        if reference_timeframe is None:
            timeframe_order = ['W1', 'D1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1']
            for tf in timeframe_order:
                if tf in multi_data:
                    reference_timeframe = tf
                    break

        if reference_timeframe not in multi_data:
            raise ValueError(f"Reference timeframe {reference_timeframe} not in data")

        reference_data = multi_data[reference_timeframe]
        synced_data = {reference_timeframe: reference_data}

        # Sync other timeframes to reference
        for timeframe, data in multi_data.items():
            if timeframe == reference_timeframe:
                continue

            # Find overlapping time range
            ref_start, ref_end = reference_data.index[0], reference_data.index[-1]
            data_start, data_end = data.index[0], data.index[-1]

            common_start = max(ref_start, data_start)
            common_end = min(ref_end, data_end)

            # Filter to common time range
            synced = data[(data.index >= common_start) & (data.index <= common_end)]
            synced_data[timeframe] = synced

            self.logger.info(f"Synchronized {timeframe}: {len(synced)} candles in common range")

        return synced_data

    def get_timeframe_mapping(self,
                            multi_data: Dict[str, pd.DataFrame],
                            from_timeframe: str,
                            to_timeframe: str) -> Dict[int, int]:
        """
        Create index mapping between two timeframes.

        Args:
            multi_data: Dictionary of timeframe DataFrames
            from_timeframe: Source timeframe
            to_timeframe: Target timeframe

        Returns:
            Dictionary mapping source indices to target indices

        Example:
            >>> mapping = loader.get_timeframe_mapping(data, 'H4', 'H1')
            >>> h1_index = mapping[100]  # H4 index 100 -> H1 index
        """
        if from_timeframe not in multi_data or to_timeframe not in multi_data:
            raise ValueError("Both timeframes must be in multi_data")

        from_data = multi_data[from_timeframe]
        to_data = multi_data[to_timeframe]

        mapping = {}

        for from_idx, from_time in enumerate(from_data.index):
            # Find closest timestamp in target timeframe
            time_diffs = np.abs(to_data.index - from_time)
            closest_idx = time_diffs.argmin()
            mapping[from_idx] = int(closest_idx)

        return mapping

    def find_equivalent_candle(self,
                             multi_data: Dict[str, pd.DataFrame],
                             source_timeframe: str,
                             source_index: int,
                             target_timeframe: str,
                             position: str = 'start') -> Optional[int]:
        """
        Find equivalent candle index in target timeframe.

        Args:
            multi_data: Dictionary of timeframe DataFrames
            source_timeframe: Source timeframe
            source_index: Source candle index
            target_timeframe: Target timeframe
            position: Which part of period to return ('start', 'middle', 'end')

        Returns:
            Equivalent candle index in target timeframe, or None if not found

        Example:
            >>> idx = loader.find_equivalent_candle(data, 'H4', 100, 'H1', 'start')
        """
        if source_timeframe not in multi_data or target_timeframe not in multi_data:
            return None

        source_data = multi_data[source_timeframe]
        target_data = multi_data[target_timeframe]

        if source_index >= len(source_data):
            return None

        # Get timestamp of source candle
        source_time = source_data.index[source_index]

        # Convert using TimeframeConverter for better accuracy
        try:
            converter_index = TimeframeConverter.convert_timeframe_index(
                source_index, source_timeframe, target_timeframe, position
            )

            # Validate against actual data
            if converter_index < len(target_data):
                return converter_index
        except:
            pass

        # Fallback to timestamp-based search
        time_diffs = np.abs(target_data.index - source_time)
        closest_idx = time_diffs.argmin()

        return int(closest_idx) if closest_idx < len(target_data) else None


class TimeframeConverter:
    """
    Utility class for converting between different timeframe indices.

    Handles index conversion between timeframes for multi-timeframe strategies.
    For example, H4 candle index 100 corresponds to H1 candle indices 400-403.
    """

    # Timeframe ratios relative to M1 (1 minute)
    TIMEFRAME_MINUTES = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'M30': 30,
        'H1': 60,
        'H4': 240,
        'D1': 1440,
        'W1': 10080
    }

    @classmethod
    def get_timeframe_ratio(cls, from_timeframe: str, to_timeframe: str) -> float:
        """
        Get the ratio between two timeframes.

        Args:
            from_timeframe: Source timeframe (e.g., 'H4')
            to_timeframe: Target timeframe (e.g., 'H1')

        Returns:
            Conversion ratio (e.g., H4 to H1 = 4.0)

        Example:
            >>> TimeframeConverter.get_timeframe_ratio('H4', 'H1')
            4.0
        """
        if from_timeframe not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {from_timeframe}")
        if to_timeframe not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {to_timeframe}")

        from_minutes = cls.TIMEFRAME_MINUTES[from_timeframe]
        to_minutes = cls.TIMEFRAME_MINUTES[to_timeframe]

        return from_minutes / to_minutes

    @classmethod
    def convert_timeframe_index(cls,
                              index: int,
                              from_timeframe: str,
                              to_timeframe: str,
                              position: str = 'start') -> int:
        """
        Convert candle index between timeframes.

        Args:
            index: Source candle index
            from_timeframe: Source timeframe (e.g., 'H4')
            to_timeframe: Target timeframe (e.g., 'H1')
            position: Which part of the period to return ('start', 'middle', 'end')

        Returns:
            Converted candle index in target timeframe

        Example:
            >>> TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'start')
            400
            >>> TimeframeConverter.convert_timeframe_index(100, 'H4', 'H1', 'end')
            403
        """
        if position not in ['start', 'middle', 'end']:
            raise ValueError("position must be 'start', 'middle', or 'end'")

        ratio = cls.get_timeframe_ratio(from_timeframe, to_timeframe)

        if ratio >= 1:
            # Converting to smaller timeframe (H4 -> H1)
            base_index = int(index * ratio)

            if position == 'start':
                return base_index
            elif position == 'middle':
                return base_index + int(ratio // 2)
            else:  # end
                return base_index + int(ratio - 1)
        else:
            # Converting to larger timeframe (H1 -> H4)
            return int(index / (1 / ratio))

    @classmethod
    def h4_to_h1_index(cls, h4_index: int, position: str = 'start') -> int:
        """
        Convert H4 candle index to H1 candle index.

        Args:
            h4_index: H4 candle index
            position: Which H1 candle within the H4 period ('start', 'middle', 'end')

        Returns:
            Corresponding H1 candle index

        Example:
            >>> TimeframeConverter.h4_to_h1_index(100, 'start')
            400
            >>> TimeframeConverter.h4_to_h1_index(100, 'end')
            403
        """
        return cls.convert_timeframe_index(h4_index, 'H4', 'H1', position)

    @classmethod
    def h1_to_h4_index(cls, h1_index: int) -> int:
        """
        Convert H1 candle index to H4 candle index.

        Args:
            h1_index: H1 candle index

        Returns:
            Corresponding H4 candle index

        Example:
            >>> TimeframeConverter.h1_to_h4_index(400)
            100
            >>> TimeframeConverter.h1_to_h4_index(403)
            100
        """
        return cls.convert_timeframe_index(h1_index, 'H1', 'H4')

    @classmethod
    def get_equivalent_candle_range(cls,
                                  index: int,
                                  from_timeframe: str,
                                  to_timeframe: str) -> Tuple[int, int]:
        """
        Get the range of candles in target timeframe that correspond to
        a single candle in source timeframe.

        Args:
            index: Source candle index
            from_timeframe: Source timeframe
            to_timeframe: Target timeframe

        Returns:
            Tuple of (start_index, end_index) in target timeframe

        Example:
            >>> TimeframeConverter.get_equivalent_candle_range(100, 'H4', 'H1')
            (400, 403)
        """
        start_idx = cls.convert_timeframe_index(index, from_timeframe, to_timeframe, 'start')
        end_idx = cls.convert_timeframe_index(index, from_timeframe, to_timeframe, 'end')
        return (start_idx, end_idx)

    @classmethod
    def validate_timeframe_alignment(cls,
                                   data1: pd.DataFrame,
                                   timeframe1: str,
                                   data2: pd.DataFrame,
                                   timeframe2: str) -> Dict[str, Any]:
        """
        Validate that two timeframes are properly aligned.

        Args:
            data1: First DataFrame
            timeframe1: Timeframe of first DataFrame
            data2: Second DataFrame
            timeframe2: Timeframe of second DataFrame

        Returns:
            Dictionary with validation results
        """
        ratio = cls.get_timeframe_ratio(timeframe1, timeframe2)

        result = {
            'aligned': True,
            'ratio': ratio,
            'data1_length': len(data1),
            'data2_length': len(data2),
            'expected_ratio': None,
            'actual_ratio': None,
            'issues': []
        }

        if ratio >= 1:
            # data1 has smaller timeframe, so data1 should be longer
            expected_ratio = ratio  # How much longer data1 should be
            actual_ratio = len(data1) / len(data2) if len(data2) > 0 else 0
            result['expected_ratio'] = expected_ratio
            result['actual_ratio'] = actual_ratio

            # Allow larger tolerance for real data (up to 50%)
            if abs(actual_ratio - expected_ratio) > expected_ratio * 0.5:
                result['aligned'] = False
                result['issues'].append(f"Length ratio mismatch: expected ~{expected_ratio}, got {actual_ratio:.2f}")
        else:
            # data1 has smaller timeframe (ratio < 1), so data1 should be longer
            expected_ratio = 1 / ratio  # How much longer data1 should be
            actual_ratio = len(data1) / len(data2) if len(data2) > 0 else 0
            result['expected_ratio'] = expected_ratio
            result['actual_ratio'] = actual_ratio

            # Allow larger tolerance for real data (up to 50%)
            if abs(actual_ratio - expected_ratio) > expected_ratio * 0.5:
                result['aligned'] = False
                result['issues'].append(f"Length ratio mismatch: expected ~{expected_ratio}, got {actual_ratio:.2f}")

        return result