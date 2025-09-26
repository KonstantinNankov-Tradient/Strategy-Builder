"""
Fair Value Gap (FVG) Detection Indicator for Strategy Builder.

This module implements FVG detection, identifying price imbalances that occur
when there's a gap between three consecutive candles, with dynamic mitigation
and shrinking tracking.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from indicators.base_indicator import BaseIndicator
from core.state_types import Detection, SignalDirection


class FvgDetector(BaseIndicator):
    """
    Fair Value Gap (FVG) detector with mitigation and shrinking tracking.

    Detects price imbalances (gaps) between three consecutive candles and tracks
    their mitigation status and size adjustments when partially touched by price.

    Features:
    - Detects bullish and bearish FVGs using 3-candle patterns
    - Tracks complete mitigation (gap fully filled)
    - Dynamically shrinks gaps when partially touched
    - Maintains history of gap states for visualization
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FVG Detector.

        Args:
            name: Unique identifier for this indicator instance
            config: Configuration parameters including:
                - min_gap_size: Minimum gap size to consider (default: 0.0)
                - track_mitigation: Track gap filling (default: True)
                - track_shrinking: Track partial gap fills (default: True)
                - symbol: Trading symbol for display (default: 'EURUSD')
        """
        # Configuration - set before calling super
        if config is None:
            config = {}
        self.min_gap_size = config.get('min_gap_size', 0.0)
        self.track_mitigation = config.get('track_mitigation', True)
        self.track_shrinking = config.get('track_shrinking', True)
        self.symbol = config.get('symbol', 'EURUSD')

        # Call parent initialization
        super().__init__(name, config)

        # Internal state for FVGs
        self.fvgs: List[Dict] = []  # All detected FVGs with their state

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.min_gap_size < 0:
            raise ValueError("min_gap_size must be non-negative")

    def get_lookback_period(self) -> int:
        """
        Minimum number of candles needed for FVG analysis.
        Need at least 3 candles for gap detection.
        """
        return 3

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check for FVG patterns in the provided data.

        Args:
            data: DataFrame with OHLCV data (datetime, open, high, low, close, volume)
            candle_index: Current candle number in backtest sequence

        Returns:
            Detection object if new FVG detected on current candle, None otherwise
        """
        if not self.validate_data(data):
            return None

        # Reset data index to datetime for analysis
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        # Ensure data is sorted by datetime
        data = data.sort_index()

        # Need at least 3 candles to detect FVG
        if len(data) < 3:
            return None

        # Detect all FVGs up to current point
        self._detect_all_fvgs(data)

        # Update mitigation and shrinking status
        if self.track_mitigation or self.track_shrinking:
            self._update_fvg_states(data)

        # Check if a new FVG was formed on the current (last) candle
        # FVGs are formed on the middle candle of the 3-candle pattern
        # So we check if the second-to-last candle forms an FVG
        if len(data) >= 3:
            detection = self._check_current_fvg(data, candle_index)
            return detection

        return None

    def _detect_all_fvgs(self, data: pd.DataFrame) -> None:
        """
        Detect all FVG patterns in the data.

        Args:
            data: OHLCV DataFrame indexed by datetime
        """
        # Clear existing FVGs for fresh detection
        self.fvgs = []

        # Need at least 3 candles
        if len(data) < 3:
            return

        # Iterate through all possible 3-candle patterns
        for i in range(len(data) - 2):
            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]

            # Check for bullish FVG (gap up)
            # Candle 3's low > Candle 1's high
            if candle3['low'] > candle1['high']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size >= self.min_gap_size:
                    self.fvgs.append({
                        'type': 'bullish',
                        'start_index': i + 1,  # Middle candle index
                        'start_time': data.index[i + 1],
                        'candle1_time': data.index[i],
                        'candle3_time': data.index[i + 2],
                        'original_top': candle3['low'],
                        'original_bottom': candle1['high'],
                        'current_top': candle3['low'],
                        'current_bottom': candle1['high'],
                        'gap_size': gap_size,
                        'is_mitigated': False,
                        'mitigation_index': None,
                        'mitigation_time': None,
                        'shrink_history': []
                    })

            # Check for bearish FVG (gap down)
            # Candle 3's high < Candle 1's low
            elif candle3['high'] < candle1['low']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= self.min_gap_size:
                    self.fvgs.append({
                        'type': 'bearish',
                        'start_index': i + 1,  # Middle candle index
                        'start_time': data.index[i + 1],
                        'candle1_time': data.index[i],
                        'candle3_time': data.index[i + 2],
                        'original_top': candle1['low'],
                        'original_bottom': candle3['high'],
                        'current_top': candle1['low'],
                        'current_bottom': candle3['high'],
                        'gap_size': gap_size,
                        'is_mitigated': False,
                        'mitigation_index': None,
                        'mitigation_time': None,
                        'shrink_history': []
                    })

    def _update_fvg_states(self, data: pd.DataFrame) -> None:
        """
        Update mitigation and shrinking status for all FVGs.

        Args:
            data: OHLCV DataFrame indexed by datetime
        """
        for fvg in self.fvgs:
            if fvg['is_mitigated']:
                continue  # Skip already mitigated FVGs

            # Check each candle after FVG formation
            start_idx = fvg['start_index'] + 2  # Start checking from candle after pattern

            for i in range(start_idx, len(data)):
                candle = data.iloc[i]
                candle_time = data.index[i]

                if fvg['type'] == 'bullish':
                    # Check for complete mitigation (bearish move fills gap)
                    if self.track_mitigation and candle['low'] <= fvg['original_bottom']:
                        fvg['is_mitigated'] = True
                        fvg['mitigation_index'] = i
                        fvg['mitigation_time'] = candle_time
                        break

                    # Check for shrinking (partial touch from above)
                    if self.track_shrinking and not fvg['is_mitigated']:
                        if candle['low'] < fvg['current_top'] and candle['low'] > fvg['current_bottom']:
                            old_top = fvg['current_top']
                            fvg['current_top'] = candle['low']
                            fvg['shrink_history'].append({
                                'candle_index': i,
                                'candle_time': candle_time,
                                'old_boundary': old_top,
                                'new_boundary': candle['low'],
                                'side': 'top'
                            })

                else:  # bearish FVG
                    # Check for complete mitigation (bullish move fills gap)
                    if self.track_mitigation and candle['high'] >= fvg['original_top']:
                        fvg['is_mitigated'] = True
                        fvg['mitigation_index'] = i
                        fvg['mitigation_time'] = candle_time
                        break

                    # Check for shrinking (partial touch from below)
                    if self.track_shrinking and not fvg['is_mitigated']:
                        if candle['high'] > fvg['current_bottom'] and candle['high'] < fvg['current_top']:
                            old_bottom = fvg['current_bottom']
                            fvg['current_bottom'] = candle['high']
                            fvg['shrink_history'].append({
                                'candle_index': i,
                                'candle_time': candle_time,
                                'old_boundary': old_bottom,
                                'new_boundary': candle['high'],
                                'side': 'bottom'
                            })

    def _check_current_fvg(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check if a new FVG was formed on the current candle pattern.

        Args:
            data: OHLCV DataFrame
            candle_index: Current candle index in backtest

        Returns:
            Detection if new FVG formed, None otherwise
        """
        # FVGs are detected when we have the complete 3-candle pattern
        # The signal is generated on the third candle
        if len(data) < 3:
            return None

        # Get the last three candles
        candle1 = data.iloc[-3]
        candle2 = data.iloc[-2]
        candle3 = data.iloc[-1]

        # Check for bullish FVG
        if candle3['low'] > candle1['high']:
            gap_size = candle3['low'] - candle1['high']
            if gap_size >= self.min_gap_size:
                direction = SignalDirection.LONG
                metadata = {
                    'fvg_type': 'bullish',
                    'gap_top': candle3['low'],
                    'gap_bottom': candle1['high'],
                    'gap_size': gap_size,
                    'formation_candle': data.index[-2],  # Middle candle
                    'signal_candle': data.index[-1]  # Third candle
                }

                return self.create_detection(
                    timestamp=data.index[-1],
                    candle_index=candle_index,
                    price=(candle3['low'] + candle1['high']) / 2,  # Gap midpoint
                    direction=direction,
                    metadata=metadata
                )

        # Check for bearish FVG
        elif candle3['high'] < candle1['low']:
            gap_size = candle1['low'] - candle3['high']
            if gap_size >= self.min_gap_size:
                direction = SignalDirection.SHORT
                metadata = {
                    'fvg_type': 'bearish',
                    'gap_top': candle1['low'],
                    'gap_bottom': candle3['high'],
                    'gap_size': gap_size,
                    'formation_candle': data.index[-2],  # Middle candle
                    'signal_candle': data.index[-1]  # Third candle
                }

                return self.create_detection(
                    timestamp=data.index[-1],
                    candle_index=candle_index,
                    price=(candle1['low'] + candle3['high']) / 2,  # Gap midpoint
                    direction=direction,
                    metadata=metadata
                )

        return None

    def get_all_fvgs(self, data: pd.DataFrame) -> List[Dict]:
        """
        Get all FVGs with their current states from the data.
        Used for visualization purposes.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of all FVGs with their properties and states
        """
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        data = data.sort_index()

        # Detect all FVGs
        self._detect_all_fvgs(data)

        # Update their states
        if self.track_mitigation or self.track_shrinking:
            self._update_fvg_states(data)

        return self.fvgs.copy()

    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self.fvgs.clear()