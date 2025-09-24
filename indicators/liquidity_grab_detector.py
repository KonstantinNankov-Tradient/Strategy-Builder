"""
Liquidity Grab Detection Indicator for Strategy Builder.

This module implements session-based liquidity grab detection, identifying when price
briefly breaks session highs/lows with wicks but fails to close beyond the level,
indicating institutional liquidity harvesting.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from indicators.base_indicator import BaseIndicator
from core.state_types import Detection, SignalDirection


class LiquidityGrabDetector(BaseIndicator):
    """
    Session-based liquidity grab detector.

    Detects when price wicks break session highs/lows but bodies fail to close
    beyond the level, indicating failed breakouts and liquidity grabs.

    Features:
    - Session identification (Asian: 21:00-05:00, European: 05:00-13:00, NY: 13:00-21:00 UTC)
    - Same-session and cross-session grab detection
    - Configurable wick extension filtering
    - Clear path validation
    - Multi-symbol pip size support
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Liquidity Grab Detector.

        Args:
            name: Unique identifier for this indicator instance
            config: Configuration parameters including:
                - enable_wick_extension_filter: Require minimum wick extension (default: True)
                - min_wick_extension_pips: Minimum pips beyond level (default: 3.0)
                - symbol: Trading symbol for pip calculation (default: 'EURUSD')
                - detect_same_session: Enable same-session grab detection (default: True)
        """
        # Configuration - set before calling super to ensure _validate_config has access
        if config is None:
            config = {}
        self.enable_wick_extension_filter = config.get('enable_wick_extension_filter', True)
        self.min_wick_extension_pips = config.get('min_wick_extension_pips', 3.0)
        self.symbol = config.get('symbol', 'EURUSD')
        self.detect_same_session = config.get('detect_same_session', True)

        # Call parent initialization (which calls _validate_config)
        super().__init__(name, config)

        # Internal state
        self.session_levels: List[Dict] = []
        self.detected_grabs: List[Dict] = []

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.min_wick_extension_pips < 0:
            raise ValueError("min_wick_extension_pips must be non-negative")

    def get_lookback_period(self) -> int:
        """
        Minimum number of candles needed for session analysis.
        Need at least 24 hours of 1H data to detect complete sessions.
        """
        return 48  # 2 days of hourly data

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check for liquidity grab patterns in the provided data.

        Args:
            data: DataFrame with OHLCV data (datetime, open, high, low, close, volume)
            candle_index: Current candle number in backtest sequence

        Returns:
            Detection object if liquidity grab detected, None otherwise
        """
        if not self.validate_data(data):
            return None

        # Reset data index to datetime for session analysis
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        # Ensure data is sorted by datetime
        data = data.sort_index()

        # Get session ranges from the data
        session_levels = self._get_session_ranges(data)

        # Check for liquidity grabs
        grab_detection = self._detect_liquidity_grabs(data, session_levels, candle_index)

        return grab_detection

    def _get_session_ranges(self, data: pd.DataFrame) -> List[Dict]:
        """
        Identify trading session ranges with session highs and lows.

        Sessions (UTC):
        - Asian: 21:00 - 05:00
        - European: 05:00 - 13:00
        - NY: 13:00 - 21:00

        Returns:
            List of session dictionaries with high/low levels
        """
        # Vectorized session assignment
        hours = data.index.hour
        session_mask_asian = (hours >= 21) | (hours < 5)
        session_mask_european = (hours >= 5) & (hours < 13)
        session_mask_ny = (hours >= 13) & (hours < 21)

        # Create session series
        sessions_series = pd.Series('', index=data.index)
        sessions_series[session_mask_asian] = 'asian'
        sessions_series[session_mask_european] = 'european'
        sessions_series[session_mask_ny] = 'ny'

        # Group sessions
        session_groups = []
        current_session = None
        current_group = []

        for timestamp, session_type in sessions_series.items():
            if not session_type:
                continue

            # Check for session change
            if current_session != session_type:
                # Special handling for Asian session continuity across midnight
                if (current_session == 'asian' and session_type == 'asian' and
                    timestamp.hour <= 5 and len(current_group) > 0 and
                    current_group[-1][0].hour >= 21):
                    # Continue Asian session across midnight
                    pass
                elif session_type == 'asian' and timestamp.hour != 21:
                    # Don't start new Asian session unless at 21:00
                    if current_session == 'asian':
                        pass  # Continue current Asian session
                    else:
                        continue  # Skip this timestamp
                else:
                    # Finalize previous group
                    if current_group:
                        session_groups.append((current_session, current_group))
                    current_group = []
                    current_session = session_type

            current_group.append((timestamp, data.loc[timestamp]))

        # Finalize last group
        if current_group:
            session_groups.append((current_session, current_group))

        # Process session groups to find high/low levels
        session_levels = []

        for session_type, session_data in session_groups:
            if not session_data or not self._is_session_complete(session_type, session_data):
                continue

            # Find session high and low
            highs = np.array([item[1]['high'] for item in session_data])
            lows = np.array([item[1]['low'] for item in session_data])
            timestamps = [item[0] for item in session_data]

            high_idx = np.argmax(highs)
            low_idx = np.argmin(lows)

            session_high = highs[high_idx]
            session_low = lows[low_idx]
            high_time = timestamps[high_idx]
            low_time = timestamps[low_idx]

            session_levels.extend([
                {
                    'level': session_high,
                    'time': high_time,
                    'session': session_type,
                    'type': 'high',
                    'start_time': timestamps[0],
                    'end_time': timestamps[-1]
                },
                {
                    'level': session_low,
                    'time': low_time,
                    'session': session_type,
                    'type': 'low',
                    'start_time': timestamps[0],
                    'end_time': timestamps[-1]
                }
            ])

        return session_levels

    def _is_session_complete(self, session_type: str, session_data: List) -> bool:
        """
        Check if a session is complete within its defined boundaries.

        Args:
            session_type: 'asian', 'european', or 'ny'
            session_data: List of (timestamp, row) tuples

        Returns:
            True if session is complete
        """
        if not session_data:
            return False

        start_time = session_data[0][0]
        end_time = session_data[-1][0]

        # Expected session durations and start hours
        session_info = {
            'asian': {'duration': 8, 'start_hour': 21, 'end_hour': 5},
            'european': {'duration': 8, 'start_hour': 5, 'end_hour': 13},
            'ny': {'duration': 8, 'start_hour': 13, 'end_hour': 21}
        }

        info = session_info.get(session_type, {})
        expected_duration = info.get('duration', 8)

        # Check session duration
        if session_type == 'asian' and end_time.hour < start_time.hour:
            # Asian session spans midnight
            actual_duration = (24 - start_time.hour) + end_time.hour
        else:
            actual_duration = end_time.hour - start_time.hour

        min_required_duration = expected_duration - 1  # Allow 1 hour tolerance

        return actual_duration >= min_required_duration

    def _detect_liquidity_grabs(self, data: pd.DataFrame, session_levels: List[Dict],
                               candle_index: int) -> Optional[Detection]:
        """
        Detect liquidity grabs against session levels.

        Args:
            data: OHLCV data
            session_levels: List of session high/low levels
            candle_index: Current candle index

        Returns:
            Detection if grab found on current candle
        """
        if not session_levels:
            return None

        current_time = data.index[-1]  # Most recent candle
        current_candle = data.iloc[-1]

        # Check each level for potential grabs
        for level_info in session_levels:
            level = level_info['level']
            level_time = level_info['time']
            level_type = level_info['type']

            # Only check levels created before current candle
            if level_time >= current_time:
                continue

            # Check if current candle creates a liquidity grab
            if self._is_liquidity_grab(current_candle, level, level_type):
                # Validate clear path and wick extension
                if (self._has_clear_path(data, level_time, current_time, level, level_type) and
                    self._meets_wick_extension(current_candle, level, level_type)):

                    # Determine direction
                    direction = SignalDirection.SHORT if level_type == 'high' else SignalDirection.LONG

                    # Determine grab session
                    grab_session = self._get_session_for_time(current_time)

                    # Determine session relationship
                    is_next_session = self._is_next_session(level_info, grab_session, current_time)
                    is_same_session = level_info['session'] == grab_session

                    if is_same_session:
                        session_relationship = "same"
                    elif is_next_session:
                        session_relationship = "next"
                    else:
                        session_relationship = "following"

                    # Create detection with metadata
                    metadata = {
                        'level_price': level,
                        'level_time': level_time,
                        'session': level_info['session'],
                        'grab_session': grab_session,
                        'grab_type': f"{level_type}_grab",
                        'session_relationship': session_relationship,
                        'is_next_session': is_next_session,
                        'is_same_session': is_same_session,
                        'wick_extension_pips': self._calculate_wick_extension_pips(
                            current_candle, level, level_type
                        )
                    }

                    return self.create_detection(
                        timestamp=current_time,
                        candle_index=candle_index,
                        price=level,
                        direction=direction,
                        metadata=metadata
                    )

        return None

    def _is_liquidity_grab(self, candle: pd.Series, level: float, level_type: str) -> bool:
        """
        Check if candle creates a liquidity grab against the level.

        Args:
            candle: OHLC candle data
            level: Price level to check
            level_type: 'high' or 'low'

        Returns:
            True if candle creates liquidity grab
        """
        body_top = max(candle['open'], candle['close'])
        body_bottom = min(candle['open'], candle['close'])

        if level_type == 'high':
            # High grab: wick breaks above level, body stays below
            return candle['high'] > level and body_top < level
        else:  # low
            # Low grab: wick breaks below level, body stays above
            return candle['low'] < level and body_bottom > level

    def _has_clear_path(self, data: pd.DataFrame, start_time: datetime,
                       end_time: datetime, level: float, level_type: str) -> bool:
        """
        Check if there's a clear path between level creation and grab.

        Args:
            data: OHLCV data
            start_time: When level was created
            end_time: When grab occurred
            level: Price level
            level_type: 'high' or 'low'

        Returns:
            True if path is clear (no prior violations)
        """
        # Get path data between times (exclusive)
        path_data = data[(data.index > start_time) & (data.index < end_time)]

        if path_data.empty:
            return True

        if level_type == 'high':
            # Check if any wick or body crosses above level
            body_tops = np.maximum(path_data['open'], path_data['close'])
            return not ((path_data['high'] >= level) | (body_tops >= level)).any()
        else:
            # Check if any wick or body crosses below level
            body_bottoms = np.minimum(path_data['open'], path_data['close'])
            return not ((path_data['low'] <= level) | (body_bottoms <= level)).any()

    def _meets_wick_extension(self, candle: pd.Series, level: float, level_type: str) -> bool:
        """
        Check if wick extension meets minimum pip requirement.

        Args:
            candle: OHLC candle data
            level: Price level
            level_type: 'high' or 'low'

        Returns:
            True if wick extension is sufficient
        """
        if not self.enable_wick_extension_filter:
            return True

        pip_size = self._get_pip_size()

        if level_type == 'high':
            wick_extension = candle['high'] - level
        else:
            wick_extension = level - candle['low']

        required_extension = self.min_wick_extension_pips * pip_size
        return wick_extension >= required_extension

    def _calculate_wick_extension_pips(self, candle: pd.Series, level: float, level_type: str) -> float:
        """Calculate wick extension in pips."""
        pip_size = self._get_pip_size()

        if level_type == 'high':
            wick_extension = candle['high'] - level
        else:
            wick_extension = level - candle['low']

        return wick_extension / pip_size

    def _get_pip_size(self, pair: Optional[str] = None) -> float:
        """
        Determine pip size based on currency pair.

        Args:
            pair: Symbol to check, defaults to self.symbol

        Returns:
            Pip size for the symbol
        """
        if pair is None:
            pair = self.symbol

        # Remove .swd suffix if present
        pair = pair.upper().replace('.SWD', '')

        pip_sizes = {
            # Forex pairs - Standard pairs (4 decimal places)
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDCHF': 0.0001,
            'AUDUSD': 0.0001,
            'USDCAD': 0.0001,
            'NZDUSD': 0.0001,
            'AUDCAD': 0.0001,
            # JPY pairs (2 decimal places)
            'USDJPY': 0.01,
            'EURJPY': 0.01,
            'GBPJPY': 0.01,
            'AUDJPY': 0.01,
            'CADJPY': 0.01,
            'CHFJPY': 0.01,
            # Commodities
            'XAUUSD': 0.01,      # Gold
            'XAGUSD': 0.001,     # Silver
            'US30': 1,           # Dow Jones
            # Cryptocurrencies
            'BTCUSD': 1.0,
            'ETHUSD': 0.1,
            'XRPUSD': 0.0001,
            'SOLUSD': 0.01,
            # Stocks
            'AAPL': 0.01,
            'MSFT': 0.01,
            'NVDA': 0.01,
            'GOOG': 0.01,
            'AMZN': 0.01
        }

        return pip_sizes.get(pair, 0.0001)  # Default to 0.0001 for unknown pairs

    def _get_session_for_time(self, timestamp: datetime) -> str:
        """Determine which session a timestamp belongs to."""
        hour = timestamp.hour
        if hour >= 21 or hour < 5:
            return 'asian'
        elif 5 <= hour < 13:
            return 'european'
        elif 13 <= hour < 21:
            return 'ny'
        else:
            return 'asian'  # Default to Asian session

    def _is_next_session(self, level_info: Dict, grab_session: str, grab_time: datetime) -> bool:
        """Determine if the grab happened in the next session or a following session."""
        level_session = level_info['session']
        level_time = level_info['time']

        # Define next session relationships based on time and session flow
        if level_session == 'asian':
            # Asian session (21:00-05:00), next is European (05:00-13:00)
            if grab_session == 'european':
                # Same day: Asian level at 21:00+ grabbed by European at 05:00+ same day (rare)
                if (grab_time.date() == level_time.date() and
                    grab_time.hour >= 5):
                    return True
                # Next day: Asian level at 21:00+ grabbed by European next day (common)
                elif ((grab_time.date() - level_time.date()).days == 1 and
                      level_time.hour >= 21 and
                      grab_time.hour >= 5):
                    return True

        elif level_session == 'european':
            # European session (05:00-13:00), next is NY (13:00-21:00)
            # Check if grab happens in the immediate NY session (same day 13:00+)
            if (grab_session == 'ny' and
                grab_time.date() == level_time.date() and
                grab_time.hour >= 13):
                return True

        elif level_session == 'ny':
            # NY session (13:00-21:00), next is Asian (21:00+ same day or next day)
            # Check if grab happens in the immediate Asian session
            if grab_session == 'asian':
                # Same day Asian session (21:00+)
                if (grab_time.date() == level_time.date() and
                    grab_time.hour >= 21):
                    return True
                # Next day Asian session (any time, since Asian starts at 21:00 previous day)
                elif ((grab_time.date() - level_time.date()).days == 1):
                    return True

        return False

    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self.session_levels.clear()
        self.detected_grabs.clear()