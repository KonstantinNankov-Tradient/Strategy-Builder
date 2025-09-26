""" 
Break of Structure (BOS) Detection Indicator for Strategy Builder.

This module implements swing-based BOS detection, identifying when price breaks
swing highs/lows with body closes in the same trend direction, confirming trend continuations.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from indicators.base_indicator import BaseIndicator
from core.state_types import Detection, SignalDirection


class BosDetector(BaseIndicator):
    """
    Swing-based Break of Structure (BOS) detector.

    Detects trend continuations by identifying when price breaks swing highs/lows
    with body closes in the same trend direction, confirming market structure continuations.

    Features:
    - Body-based swing detection with configurable strength and gap parameters
    - Vectorized breakout detection for performance
    - Trend direction tracking (bullish/bearish)
    - BOS signal generation on trend continuations (opposite of CHoCH)
    - First swing does not generate BOS, only subsequent same-trend breaks
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BOS Detector.

        Args:
            name: Unique identifier for this indicator instance
            config: Configuration parameters including:
                - base_strength: Swing detection strength (default: 5)
                - min_gap: Minimum gap between swings (default: 3)
                - symbol: Trading symbol for display (default: 'EURUSD')
        """
        # Configuration - set before calling super
        if config is None:
            config = {}
        self.base_strength = config.get('base_strength', 5)
        self.min_gap = config.get('min_gap', 3)
        self.symbol = config.get('symbol', 'EURUSD')

        # Call parent initialization
        super().__init__(name, config)

        # Internal state
        self.swing_highs: List[Tuple[datetime, float]] = []
        self.swing_lows: List[Tuple[datetime, float]] = []
        self.bos_signals: List[Dict] = []
        self.current_trend: int = 0  # 1 = bullish, -1 = bearish, 0 = no trend
        self.trend_established: bool = False  # Track if initial trend is set

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.base_strength < 1:
            raise ValueError("base_strength must be at least 1")
        if self.min_gap < 0:
            raise ValueError("min_gap must be non-negative")

    def get_lookback_period(self) -> int:
        """
        Minimum number of candles needed for swing analysis.
        Need enough candles for swing detection plus breakout confirmation.
        """
        return (self.base_strength * 2) + 10  # Extra buffer for swing detection

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check for BOS patterns in the provided data.

        Args:
            data: DataFrame with OHLCV data (datetime, open, high, low, close, volume)
            candle_index: Current candle number in backtest sequence

        Returns:
            Detection object if BOS detected, None otherwise
        """
        if not self.validate_data(data):
            return None

        # Reset data index to datetime for analysis
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        # Ensure data is sorted by datetime
        data = data.sort_index()

        # Get swing points from the data
        swing_labels, swing_levels = self._get_swing_highs_lows(data)

        # Detect BOS based on swing breakouts
        bos_detection = self._detect_bos(data, swing_labels, swing_levels, candle_index)

        return bos_detection

    def _get_swing_highs_lows(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect swing highs and lows using body-based analysis.

        Args:
            data: OHLCV DataFrame indexed by datetime

        Returns:
            Tuple of (swing_labels, swing_levels) where:
            - swing_labels: 1 = swing high, -1 = swing low, NaN = no swing
            - swing_levels: Price level where swing occurs
        """
        data_len = len(data)

        # Pre-compute body highs and lows for entire dataset
        body_highs = np.maximum(data['open'].values, data['close'].values)
        body_lows = np.minimum(data['open'].values, data['close'].values)

        # Initialize result arrays
        swing_labels = np.full(data_len, np.nan)
        swing_levels = np.full(data_len, np.nan)

        # Create rolling windows for vectorized comparison
        valid_range = range(self.base_strength, data_len - self.base_strength)

        if not valid_range:
            # Return empty series if not enough data
            return (pd.Series(swing_labels, index=data.index, name="Swing Labels"),
                   pd.Series(swing_levels, index=data.index, name="Swing Levels"))

        # Vectorized swing detection
        for i in valid_range:
            current_high = body_highs[i]
            current_low = body_lows[i]

            # Get surrounding data ranges
            start_past = max(0, i - self.base_strength)
            end_future = min(data_len, i + 1 + self.base_strength)

            # Check if current point is local maximum (swing high)
            past_highs = body_highs[start_past:i]
            future_highs = body_highs[i+1:end_future]

            is_swing_high = (len(past_highs) == 0 or current_high >= np.max(past_highs)) and \
                           (len(future_highs) == 0 or current_high >= np.max(future_highs))

            # Check if current point is local minimum (swing low)
            past_lows = body_lows[start_past:i]
            future_lows = body_lows[i+1:end_future]

            is_swing_low = (len(past_lows) == 0 or current_low <= np.min(past_lows)) and \
                          (len(future_lows) == 0 or current_low <= np.min(future_lows))

            # Apply min_gap constraint
            if is_swing_high:
                # Check gap from previous swing highs
                prev_swing_highs = np.where(swing_labels[:i] == 1)[0]
                if len(prev_swing_highs) == 0 or (i - prev_swing_highs[-1]) >= self.min_gap:
                    swing_labels[i] = 1
                    swing_levels[i] = current_high

            if is_swing_low:
                # Check gap from previous swing lows
                prev_swing_lows = np.where(swing_labels[:i] == -1)[0]
                if len(prev_swing_lows) == 0 or (i - prev_swing_lows[-1]) >= self.min_gap:
                    swing_labels[i] = -1
                    swing_levels[i] = current_low

        # Convert to pandas Series
        swing_labels_series = pd.Series(swing_labels, index=data.index, name="Swing Labels")
        swing_levels_series = pd.Series(swing_levels, index=data.index, name="Swing Levels")

        return swing_labels_series, swing_levels_series

    def _detect_bos(self, data: pd.DataFrame, swing_labels: pd.Series,
                    swing_levels: pd.Series, candle_index: int) -> Optional[Detection]:
        """
        Detect BOS based on swing point breakouts.

        Args:
            data: OHLCV data
            swing_labels: Swing point labels (1=high, -1=low)
            swing_levels: Swing point price levels
            candle_index: Current candle index

        Returns:
            Detection if BOS found on current candle
        """
        # Process all historical breakouts to establish state
        all_historical_breakouts = []

        # Get all swing highs and lows
        swing_highs = swing_labels[swing_labels == 1]
        swing_lows = swing_labels[swing_labels == -1]

        # Check for bullish breakouts throughout the data
        for swing_time in swing_highs.index:
            swing_level = swing_levels[swing_time]
            # Find first breakout after this swing
            future_data = data[data.index > swing_time]
            if not future_data.empty:
                for idx, row in future_data.iterrows():
                    body_top = max(row['open'], row['close'])
                    if body_top > swing_level:
                        all_historical_breakouts.append((swing_time, swing_level, idx, 1, 'Bullish'))
                        break

        # Check for bearish breakouts throughout the data
        for swing_time in swing_lows.index:
            swing_level = swing_levels[swing_time]
            # Find first breakout after this swing
            future_data = data[data.index > swing_time]
            if not future_data.empty:
                for idx, row in future_data.iterrows():
                    body_bottom = min(row['open'], row['close'])
                    if body_bottom < swing_level:
                        all_historical_breakouts.append((swing_time, swing_level, idx, -1, 'Bearish'))
                        break

        # Sort by breakout time
        all_historical_breakouts.sort(key=lambda x: x[2])

        # Process historical breakouts to establish current state
        local_trend = 0
        local_trend_established = False

        for i, (swing_time, swing_price, breakout_time, trend, trend_name) in enumerate(all_historical_breakouts):
            if i == 0:
                # First breakout establishes trend
                local_trend = trend
                local_trend_established = True
            elif trend != local_trend:
                # Trend change (CHoCH)
                local_trend = trend

        # Update detector state
        self.current_trend = local_trend
        self.trend_established = local_trend_established

        # Now check if current candle creates a BOS
        current_time = data.index[-1]
        current_candle = data.iloc[-1]

        current_breakouts = []

        # Check for bullish breakouts on current candle
        for swing_time in swing_highs.index:
            if swing_time >= current_time:  # Skip future swings
                continue

            swing_level = swing_levels[swing_time]
            body_top = max(current_candle['open'], current_candle['close'])

            if body_top > swing_level:
                # Check if this breakout hasn't happened before current candle
                already_broken = False
                for idx, row in data[data.index > swing_time].iloc[:-1].iterrows():
                    if max(row['open'], row['close']) > swing_level:
                        already_broken = True
                        break

                if not already_broken:
                    current_breakouts.append((swing_time, swing_level, current_time, 'up'))

        # Check for bearish breakouts on current candle
        for swing_time in swing_lows.index:
            if swing_time >= current_time:  # Skip future swings
                continue

            swing_level = swing_levels[swing_time]
            body_bottom = min(current_candle['open'], current_candle['close'])

            if body_bottom < swing_level:
                # Check if this breakout hasn't happened before current candle
                already_broken = False
                for idx, row in data[data.index > swing_time].iloc[:-1].iterrows():
                    if min(row['open'], row['close']) < swing_level:
                        already_broken = True
                        break

                if not already_broken:
                    current_breakouts.append((swing_time, swing_level, current_time, 'down'))

        if not current_breakouts:
            return None

        # Sort by swing time to get most recent breakout
        current_breakouts.sort(key=lambda x: x[0], reverse=True)
        most_recent_swing = current_breakouts[0]

        start_time, swing_price, breakout_time, trend_str = most_recent_swing
        new_trend = 1 if trend_str == 'up' else -1

        # Check if this is a BOS
        if not self.trend_established:
            # First breakout establishes trend but is not a BOS
            self.current_trend = new_trend
            self.trend_established = True
            return None
        elif new_trend != self.current_trend:
            # Trend change (CHoCH, not BOS)
            self.current_trend = new_trend
            return None
        elif new_trend == self.current_trend:
            # This is a BOS - trend continuation
            direction = SignalDirection.LONG if new_trend == 1 else SignalDirection.SHORT
            trend_name = "Bullish" if new_trend == 1 else "Bearish"

            metadata = {
                'swing_time': start_time,
                'swing_price': swing_price,
                'breakout_time': breakout_time,
                'trend_direction': new_trend,
                'trend_name': trend_name,
                'bos_type': f"{trend_name.lower()}_bos",
                'is_continuation': True
            }

            return self.create_detection(
                timestamp=current_time,
                candle_index=candle_index,
                price=swing_price,
                direction=direction,
                metadata=metadata
            )

        return None

    def get_all_swings_and_bos(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Get all swing points and BOS signals from the data.
        Used for visualization purposes.

        Args:
            data: OHLCV DataFrame

        Returns:
            Tuple of (swing_points, bos_signals)
        """
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        data = data.sort_index()

        # Get swing points
        swing_labels, swing_levels = self._get_swing_highs_lows(data)

        # Collect all swings
        swing_points = []
        for timestamp, label in swing_labels.items():
            if not pd.isna(label):
                swing_points.append({
                    'time': timestamp,
                    'price': swing_levels[timestamp],
                    'type': 'high' if label == 1 else 'low',
                    'label': int(label)
                })

        # Detect all BOS signals chronologically
        bos_signals = []
        current_trend = 0
        trend_established = False

        # Get all swing highs and lows
        swing_highs = swing_labels[swing_labels == 1]
        swing_lows = swing_labels[swing_labels == -1]

        # Combine and sort all swings chronologically
        all_swings = []

        for swing_time in swing_highs.index:
            swing_level = swing_levels[swing_time]
            # Find first breakout after this swing
            future_data = data[data.index > swing_time]
            if not future_data.empty:
                breakout_mask = np.maximum(future_data['open'], future_data['close']) > swing_level
                if breakout_mask.any():
                    breakout_time = future_data.index[breakout_mask][0]
                    all_swings.append((swing_time, swing_level, breakout_time, 1, 'Bullish'))

        for swing_time in swing_lows.index:
            swing_level = swing_levels[swing_time]
            # Find first breakout after this swing
            future_data = data[data.index > swing_time]
            if not future_data.empty:
                breakout_mask = np.minimum(future_data['open'], future_data['close']) < swing_level
                if breakout_mask.any():
                    breakout_time = future_data.index[breakout_mask][0]
                    all_swings.append((swing_time, swing_level, breakout_time, -1, 'Bearish'))

        # Sort by breakout time
        all_swings.sort(key=lambda x: x[2])

        # Generate BOS signals
        for i, (swing_time, swing_price, breakout_time, trend, trend_name) in enumerate(all_swings):
            # BOS logic: NOT first swing and same trend as current
            is_bos = (i != 0) and (trend == current_trend)

            if i == 0:
                # First swing establishes trend but is not a BOS
                current_trend = trend
                trend_established = True
            elif trend != current_trend:
                # Trend change (CHoCH, not BOS)
                current_trend = trend
            elif is_bos:
                # This is a BOS - trend continuation
                bos_signals.append({
                    'swing_time': swing_time,
                    'swing_price': swing_price,
                    'breakout_time': breakout_time,
                    'trend_direction': trend,
                    'trend_name': trend_name,
                    'signal': trend,
                    'is_continuation': True
                })

        return swing_points, bos_signals

    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.bos_signals.clear()
        self.current_trend = 0
        self.trend_established = False