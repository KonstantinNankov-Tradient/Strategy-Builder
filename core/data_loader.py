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