"""
Base indicator abstract class for all trading indicators.

This module provides the abstract base class that all indicators must inherit from.
It defines the standard interface for indicator implementations and provides
common functionality for data validation and error handling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_types import Detection, SignalDirection


class BaseIndicator(ABC):
    """
    Abstract base class for all trading indicators.

    This class defines the interface that all indicators must implement,
    ensuring consistent behavior across the system. Indicators are responsible
    for analyzing market data and detecting specific patterns or conditions.

    Attributes:
        name: Unique identifier for this indicator instance
        config: Configuration parameters for the indicator
        logger: Logger instance for this indicator
        internal_state: Dictionary for storing indicator-specific state
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base indicator.

        Args:
            name: Unique name for this indicator instance
            config: Optional configuration dictionary with indicator-specific parameters
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.internal_state: Dict[str, Any] = {}

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.

        Subclasses can override this to add specific validation.
        """
        pass

    @abstractmethod
    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """
        Check for indicator signal in the provided data.

        This is the main method that analyzes market data and detects patterns.
        It should be deterministic - same input should always produce same output.

        Args:
            data: DataFrame with OHLCV data. Expected columns:
                  - datetime: Timestamp of the candle
                  - open: Opening price
                  - high: Highest price
                  - low: Lowest price
                  - close: Closing price
                  - volume: Trading volume
                  The DataFrame should contain at least get_lookback_period() rows.
                  The most recent candle is at index -1.

            candle_index: Current candle number in the overall backtest sequence.
                         Used for tracking when the detection occurred.

        Returns:
            Detection object if a pattern is detected, None otherwise.
            The Detection should include:
            - indicator_name: Name of this indicator
            - timestamp: When the detection occurred
            - candle_index: The provided candle_index
            - price: Relevant price level for the detection
            - direction: LONG, SHORT, or NONE
            - metadata: Optional dict with indicator-specific data
        """
        pass

    @abstractmethod
    def get_lookback_period(self) -> int:
        """
        Get the number of historical candles required for this indicator.

        This tells the system how much historical data the indicator needs
        to perform its analysis. The system will ensure at least this many
        candles are provided to the check() method.

        Returns:
            Minimum number of candles required for indicator to function.
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state of the indicator.

        Called between different setups or when the indicator needs to be
        reset to initial conditions. Subclasses should override if they
        maintain internal state that needs clearing.
        """
        self.internal_state.clear()
        self.logger.debug(f"Indicator {self.name} reset")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the provided data meets requirements.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning(f"{self.name}: Empty or None data provided")
            return False

        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.logger.warning(f"{self.name}: Missing columns: {missing_columns}")
            return False

        if len(data) < self.get_lookback_period():
            self.logger.warning(
                f"{self.name}: Insufficient data. "
                f"Need {self.get_lookback_period()}, got {len(data)}"
            )
            return False

        # Check for NaN values
        if data[required_columns].isnull().any().any():
            self.logger.warning(f"{self.name}: Data contains NaN values")
            return False

        return True

    def create_detection(
        self,
        timestamp: datetime,
        candle_index: int,
        price: float,
        direction: SignalDirection,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Detection:
        """
        Helper method to create a Detection object.

        Args:
            timestamp: When the detection occurred
            candle_index: Candle number in backtest
            price: Relevant price for the detection
            direction: Signal direction (LONG/SHORT/NONE)
            metadata: Optional additional data

        Returns:
            Configured Detection object
        """
        return Detection(
            indicator_name=self.name,
            timestamp=timestamp,
            candle_index=candle_index,
            price=price,
            direction=direction,
            metadata=metadata or {}
        )

    def get_recent_highs(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Get rolling window highs.

        Args:
            data: OHLCV DataFrame
            window: Number of candles for rolling window

        Returns:
            Series of rolling maximum high values
        """
        return data['high'].rolling(window=window, min_periods=1).max()

    def get_recent_lows(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Get rolling window lows.

        Args:
            data: OHLCV DataFrame
            window: Number of candles for rolling window

        Returns:
            Series of rolling minimum low values
        """
        return data['low'].rolling(window=window, min_periods=1).min()

    def calculate_pip_distance(self, price1: float, price2: float) -> float:
        """
        Calculate distance in pips between two prices.

        Args:
            price1: First price
            price2: Second price

        Returns:
            Distance in pips (assumes 4 decimal places for forex)
        """
        return abs(price1 - price2) * 10000

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed representation of the indicator."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"config={self.config}, "
            f"lookback={self.get_lookback_period()})"
        )