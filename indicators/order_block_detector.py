"""
Order Block (OB) Detection Indicator for Strategy Builder.

This module implements sophisticated Order Block detection, identifying institutional
supply and demand zones based on swing breakouts with volume analysis and mitigation tracking.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from indicators.base_indicator import BaseIndicator
from core.state_types import Detection, SignalDirection


class OrderBlockDetector(BaseIndicator):
    """
    Order Block detector with swing-based formation and volume analysis.

    Detects institutional supply/demand zones that form when price breaks through
    swing highs/lows, creating areas where large volumes of orders were placed.

    Features:
    - Swing-based Order Block formation detection
    - Volume-based strength calculations
    - Mitigation tracking (full and partial)
    - Dynamic block management (active vs mitigated)
    - Bullish and bearish block identification
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Order Block Detector.

        Args:
            name: Unique identifier for this indicator instance
            config: Configuration parameters including:
                - base_strength: Swing detection strength (default: 5)
                - min_gap: Minimum gap between swings (default: 3)
                - close_mitigation: Use close price for mitigation (default: False)
                - track_volume: Calculate volume-based strength (default: True)
                - max_blocks: Maximum active blocks to track (default: 10)
                - symbol: Trading symbol for display (default: 'EURUSD')
        """
        # Configuration - set before calling super
        if config is None:
            config = {}
        self.base_strength = config.get('base_strength', 5)
        self.min_gap = config.get('min_gap', 3)
        self.close_mitigation = config.get('close_mitigation', False)
        self.track_volume = config.get('track_volume', True)
        self.max_blocks = config.get('max_blocks', 10)
        self.symbol = config.get('symbol', 'EURUSD')

        # Call parent initialization
        super().__init__(name, config)

        # Internal state for Order Blocks
        self.order_blocks: List[Dict] = []  # All detected order blocks
        self.swing_highs: List[Tuple[datetime, float]] = []
        self.swing_lows: List[Tuple[datetime, float]] = []

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.base_strength < 1:
            raise ValueError("base_strength must be at least 1")
        if self.min_gap < 0:
            raise ValueError("min_gap must be non-negative")
        if self.max_blocks < 1:
            raise ValueError("max_blocks must be at least 1")

    def get_lookback_period(self) -> int:
        """
        Minimum number of candles needed for Order Block analysis.
        Need enough candles for swing detection plus breakout confirmation.
        """
        return (self.base_strength * 2) + 10

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check for Order Block patterns in the provided data.

        Args:
            data: DataFrame with OHLCV data (datetime, open, high, low, close, volume)
            candle_index: Current candle number in backtest sequence

        Returns:
            Detection object if new Order Block detected, None otherwise
        """
        if not self.validate_data(data):
            return None

        # Reset data index to datetime for analysis
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        # Ensure data is sorted by datetime
        data = data.sort_index()

        # Need sufficient data for swing analysis
        if len(data) < self.get_lookback_period():
            return None

        # Detect all order blocks up to current point
        self._detect_all_order_blocks(data)

        # Update mitigation status
        self._update_mitigation_status(data)

        # Check if a new order block was formed on current candle
        return self._check_current_order_block(data, candle_index)

    def _detect_swing_points(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
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

        # Pre-compute body highs and lows
        body_highs = np.maximum(data['open'].values, data['close'].values)
        body_lows = np.minimum(data['open'].values, data['close'].values)

        # Initialize result arrays
        swing_labels = np.full(data_len, np.nan)
        swing_levels = np.full(data_len, np.nan)

        # Vectorized swing detection
        valid_range = range(self.base_strength, data_len - self.base_strength)

        for i in valid_range:
            current_high = body_highs[i]
            current_low = body_lows[i]

            # Get surrounding data ranges
            start_past = max(0, i - self.base_strength)
            end_future = min(data_len, i + 1 + self.base_strength)

            # Check for swing high
            past_highs = body_highs[start_past:i]
            future_highs = body_highs[i+1:end_future]

            is_swing_high = (len(past_highs) == 0 or current_high >= np.max(past_highs)) and \
                           (len(future_highs) == 0 or current_high >= np.max(future_highs))

            # Check for swing low
            past_lows = body_lows[start_past:i]
            future_lows = body_lows[i+1:end_future]

            is_swing_low = (len(past_lows) == 0 or current_low <= np.min(past_lows)) and \
                          (len(future_lows) == 0 or current_low <= np.min(future_lows))

            # Apply min_gap constraint
            if is_swing_high:
                prev_swing_highs = np.where(swing_labels[:i] == 1)[0]
                if len(prev_swing_highs) == 0 or (i - prev_swing_highs[-1]) >= self.min_gap:
                    swing_labels[i] = 1
                    swing_levels[i] = data['high'].iloc[i]  # Use actual high, not body high

            if is_swing_low:
                prev_swing_lows = np.where(swing_labels[:i] == -1)[0]
                if len(prev_swing_lows) == 0 or (i - prev_swing_lows[-1]) >= self.min_gap:
                    swing_labels[i] = -1
                    swing_levels[i] = data['low'].iloc[i]  # Use actual low, not body low

        # Convert to pandas Series
        swing_labels_series = pd.Series(swing_labels, index=data.index, name="Swing Labels")
        swing_levels_series = pd.Series(swing_levels, index=data.index, name="Swing Levels")

        return swing_labels_series, swing_levels_series

    def _detect_all_order_blocks(self, data: pd.DataFrame) -> None:
        """
        Detect all Order Block formations in the data.

        Args:
            data: OHLCV DataFrame indexed by datetime
        """
        # Clear existing order blocks for fresh detection
        self.order_blocks = []

        # Get swing points
        swing_labels, swing_levels = self._detect_swing_points(data)

        # Track which swings have been used for order block formation
        used_swings = set()

        # Process each candle looking for breakouts
        for i in range(2, len(data)):
            current_time = data.index[i]
            current_candle = data.iloc[i]

            # Check for bullish order block formation (break above swing high)
            self._check_bullish_ob_formation(
                data, swing_labels, swing_levels, i, current_time, current_candle, used_swings
            )

            # Check for bearish order block formation (break below swing low)
            self._check_bearish_ob_formation(
                data, swing_labels, swing_levels, i, current_time, current_candle, used_swings
            )

    def _check_bullish_ob_formation(self, data: pd.DataFrame, swing_labels: pd.Series,
                                  swing_levels: pd.Series, candle_idx: int,
                                  current_time: datetime, current_candle: pd.Series,
                                  used_swings: set) -> None:
        """Check for bullish order block formation at current candle."""
        # Find most recent unused swing high before current candle
        swing_highs_before = swing_labels[:candle_idx][swing_labels[:candle_idx] == 1]

        if swing_highs_before.empty:
            return

        # Get the most recent swing high
        last_swing_time = swing_highs_before.index[-1]

        if last_swing_time in used_swings:
            return

        swing_high_price = swing_levels[last_swing_time]

        # Check if current candle closes above swing high (breakout)
        if current_candle['close'] > swing_high_price:
            used_swings.add(last_swing_time)

            # Find the optimal order block zone between swing and breakout
            swing_idx = data.index.get_loc(last_swing_time)
            ob_zone = self._find_bullish_ob_zone(data, swing_idx, candle_idx)

            if ob_zone:
                ob_idx, ob_top, ob_bottom = ob_zone

                # Calculate volume data if enabled
                volume_data = self._calculate_volume_data(data, candle_idx) if self.track_volume else {}

                # Create bullish order block
                order_block = {
                    'type': 'bullish',
                    'formation_time': data.index[ob_idx],
                    'swing_time': last_swing_time,
                    'breakout_time': current_time,
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'is_mitigated': False,
                    'mitigation_time': None,
                    'volume_data': volume_data,
                    'strength_percentage': self._calculate_strength_percentage(volume_data) if volume_data else 0
                }

                self.order_blocks.append(order_block)

    def _check_bearish_ob_formation(self, data: pd.DataFrame, swing_labels: pd.Series,
                                   swing_levels: pd.Series, candle_idx: int,
                                   current_time: datetime, current_candle: pd.Series,
                                   used_swings: set) -> None:
        """Check for bearish order block formation at current candle."""
        # Find most recent unused swing low before current candle
        swing_lows_before = swing_labels[:candle_idx][swing_labels[:candle_idx] == -1]

        if swing_lows_before.empty:
            return

        # Get the most recent swing low
        last_swing_time = swing_lows_before.index[-1]

        if last_swing_time in used_swings:
            return

        swing_low_price = swing_levels[last_swing_time]

        # Check if current candle closes below swing low (breakout)
        if current_candle['close'] < swing_low_price:
            used_swings.add(last_swing_time)

            # Find the optimal order block zone between swing and breakout
            swing_idx = data.index.get_loc(last_swing_time)
            ob_zone = self._find_bearish_ob_zone(data, swing_idx, candle_idx)

            if ob_zone:
                ob_idx, ob_top, ob_bottom = ob_zone

                # Calculate volume data if enabled
                volume_data = self._calculate_volume_data(data, candle_idx) if self.track_volume else {}

                # Create bearish order block
                order_block = {
                    'type': 'bearish',
                    'formation_time': data.index[ob_idx],
                    'swing_time': last_swing_time,
                    'breakout_time': current_time,
                    'top': ob_top,
                    'bottom': ob_bottom,
                    'is_mitigated': False,
                    'mitigation_time': None,
                    'volume_data': volume_data,
                    'strength_percentage': self._calculate_strength_percentage(volume_data) if volume_data else 0
                }

                self.order_blocks.append(order_block)

    def _find_bullish_ob_zone(self, data: pd.DataFrame, swing_idx: int, breakout_idx: int) -> Optional[Tuple[int, float, float]]:
        """
        Find the optimal bullish order block zone between swing and breakout.
        Returns (candle_index, top_price, bottom_price) of the best OB zone.
        """
        best_low = float('inf')
        best_idx = None

        # Analyze candles between swing high and breakout
        for i in range(swing_idx + 1, breakout_idx):
            candle = data.iloc[i]

            # For bullish OB, find candle with lowest low
            if candle['low'] < best_low:
                best_low = candle['low']
                best_idx = i

        if best_idx is not None:
            candle = data.iloc[best_idx]
            # OB top = body top, bottom = candle low
            ob_top = max(candle['open'], candle['close'])
            ob_bottom = candle['low']
            return best_idx, ob_top, ob_bottom

        return None

    def _find_bearish_ob_zone(self, data: pd.DataFrame, swing_idx: int, breakout_idx: int) -> Optional[Tuple[int, float, float]]:
        """
        Find the optimal bearish order block zone between swing and breakout.
        Returns (candle_index, top_price, bottom_price) of the best OB zone.
        """
        best_high = float('-inf')
        best_idx = None

        # Analyze candles between swing low and breakout
        for i in range(swing_idx + 1, breakout_idx):
            candle = data.iloc[i]

            # For bearish OB, find candle with highest high
            if candle['high'] > best_high:
                best_high = candle['high']
                best_idx = i

        if best_idx is not None:
            candle = data.iloc[best_idx]
            # OB top = candle high, bottom = body bottom
            ob_top = candle['high']
            ob_bottom = min(candle['open'], candle['close'])
            return best_idx, ob_top, ob_bottom

        return None

    def _calculate_volume_data(self, data: pd.DataFrame, breakout_idx: int) -> Dict[str, float]:
        """Calculate volume-based data for order block strength."""
        if breakout_idx < 2:
            return {'total_volume': 0, 'breakout_volume': 0, 'formation_volume': 0}

        # Get volumes from breakout candle and 2 preceding candles
        vol_current = data.iloc[breakout_idx]['volume']
        vol_prev1 = data.iloc[breakout_idx - 1]['volume']
        vol_prev2 = data.iloc[breakout_idx - 2]['volume']

        # Handle NaN volumes
        vol_current = vol_current if not pd.isna(vol_current) else 0
        vol_prev1 = vol_prev1 if not pd.isna(vol_prev1) else 0
        vol_prev2 = vol_prev2 if not pd.isna(vol_prev2) else 0

        return {
            'total_volume': vol_current + vol_prev1 + vol_prev2,
            'breakout_volume': vol_current + vol_prev1,
            'formation_volume': vol_prev2
        }

    def _calculate_strength_percentage(self, volume_data: Dict[str, float]) -> float:
        """Calculate strength percentage based on volume ratios."""
        if not volume_data or volume_data['breakout_volume'] == 0:
            return 0.0

        # Calculate ratio of formation to breakout volume
        formation_vol = volume_data['formation_volume']
        breakout_vol = volume_data['breakout_volume']

        max_vol = max(formation_vol, breakout_vol)
        min_vol = min(formation_vol, breakout_vol)

        if max_vol == 0:
            return 0.0

        return (min_vol / max_vol) * 100.0

    def _update_mitigation_status(self, data: pd.DataFrame) -> None:
        """Update mitigation status for all order blocks."""
        for ob in self.order_blocks:
            if ob['is_mitigated']:
                continue  # Skip already mitigated blocks

            # Find candles after order block formation
            formation_time = ob['formation_time']
            future_candles = data[data.index > formation_time]

            for timestamp, candle in future_candles.iterrows():
                if ob['type'] == 'bullish':
                    # Bullish OB mitigated when price goes below OB bottom
                    if (not self.close_mitigation and candle['low'] < ob['bottom']) or \
                       (self.close_mitigation and min(candle['open'], candle['close']) < ob['bottom']):
                        ob['is_mitigated'] = True
                        ob['mitigation_time'] = timestamp
                        break
                else:  # bearish
                    # Bearish OB mitigated when price goes above OB top
                    if (not self.close_mitigation and candle['high'] > ob['top']) or \
                       (self.close_mitigation and max(candle['open'], candle['close']) > ob['top']):
                        ob['is_mitigated'] = True
                        ob['mitigation_time'] = timestamp
                        break

    def _check_current_order_block(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check if a new order block was formed on the current candle.

        Args:
            data: OHLCV DataFrame
            candle_index: Current candle index

        Returns:
            Detection if new order block formed, None otherwise
        """
        if not self.order_blocks:
            return None

        # Check if the most recent order block was formed on current candle
        latest_ob = self.order_blocks[-1]
        current_time = data.index[-1]

        if latest_ob['breakout_time'] == current_time:
            # New order block detected
            direction = SignalDirection.LONG if latest_ob['type'] == 'bullish' else SignalDirection.SHORT

            metadata = {
                'ob_type': latest_ob['type'],
                'swing_time': latest_ob['swing_time'],
                'formation_time': latest_ob['formation_time'],
                'ob_top': latest_ob['top'],
                'ob_bottom': latest_ob['bottom'],
                'strength_percentage': latest_ob['strength_percentage'],
                'volume_data': latest_ob['volume_data']
            }

            return self.create_detection(
                timestamp=current_time,
                candle_index=candle_index,
                price=(latest_ob['top'] + latest_ob['bottom']) / 2,  # OB midpoint
                direction=direction,
                metadata=metadata
            )

        return None

    def get_all_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """
        Get all order blocks with their current states from the data.
        Used for visualization purposes.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of all order blocks with their properties
        """
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        data = data.sort_index()

        # Detect all order blocks
        self._detect_all_order_blocks(data)

        # Update mitigation status
        self._update_mitigation_status(data)

        return self.order_blocks.copy()

    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self.order_blocks.clear()
        self.swing_highs.clear()
        self.swing_lows.clear()