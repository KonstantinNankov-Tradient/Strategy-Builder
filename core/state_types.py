"""
State type definitions for the sequential trading strategy state machine.

This module defines all the enums, types, and data structures for a
single-setup-at-a-time trading system. Any indicator can start a setup,
and the sequence is determined by the strategy configuration.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExecutionState(Enum):
    """
    Generic state progression for any strategy.

    States are generic - not tied to specific indicators:
    - SCANNING: Looking for first signal (could be any indicator)
    - SIGNAL_1 through SIGNAL_5: Generic states for indicator chain
    - IN_POSITION: Trade active
    """
    SCANNING = "scanning"           # Looking for first signal to start setup

    # Generic signal states (strategy defines what each means)
    SIGNAL_1 = "signal_1"           # First indicator detected
    SIGNAL_2 = "signal_2"           # Second indicator detected
    SIGNAL_3 = "signal_3"           # Third indicator detected
    SIGNAL_4 = "signal_4"           # Fourth indicator detected
    SIGNAL_5 = "signal_5"           # Fifth indicator detected

    # Trading states
    READY_TO_ENTER = "ready_to_enter"  # All required signals confirmed
    IN_POSITION = "in_position"        # Trade is active
    POSITION_CLOSED = "position_closed" # Trade closed


class ActionType(Enum):
    """
    Actions the state machine can take after processing each candle.
    """
    WAIT = "wait"                    # No detection, continue monitoring
    INDICATOR_DETECTED = "detected"  # An indicator triggered
    ENTER_TRADE = "enter_trade"      # Open a position
    EXIT_TRADE = "exit_trade"        # Close the position
    RESET = "reset"                  # Reset to scanning
    TIMEOUT = "timeout"              # Setup expired


class SignalDirection(Enum):
    """Direction of the trading signal."""
    LONG = "long"    # Buy signal
    SHORT = "short"  # Sell signal
    NONE = "none"    # No direction yet


@dataclass
class Detection:
    """
    Records any indicator detection.

    Simple binary detection - no strength scores.
    The indicator either detected something or it didn't.
    """
    indicator_name: str          # Name of indicator (e.g., "rsi_oversold", "ma_cross")
    timestamp: datetime          # When it detected
    candle_index: int           # Which candle number
    price: float                # Price at detection
    direction: SignalDirection   # Direction indicated by this signal

    # Optional indicator-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.indicator_name} at {self.price:.5f} ({self.direction.value})"


@dataclass
class StrategyConfig:
    """
    Defines the strategy's indicator sequence and rules.

    This makes the state machine completely flexible -
    any sequence of indicators can be defined here.
    """
    # Ordered list of indicators to check
    indicator_sequence: List[str]  # e.g., ["rsi_oversold", "support_bounce", "volume_spike"]

    # How many indicators must trigger to enter (might be less than total)
    required_confirmations: int = 3

    # Timeouts between each detection (in candles)
    timeouts: Dict[str, int] = field(default_factory=lambda: {
        'default': 50,  # Default timeout if not specified
    })

    # Risk management parameters
    stop_loss_pips: float = 20
    take_profit_pips: float = 40
    risk_percent: float = 0.01

    def get_timeout_after(self, indicator_name: str) -> int:
        """Get timeout to wait after specific indicator."""
        return self.timeouts.get(indicator_name, self.timeouts.get('default', 50))

    def get_next_indicator(self, current_detections: List[str]) -> Optional[str]:
        """
        Determine which indicator to look for next.

        Returns None if we have enough confirmations.
        """
        # Check if we have enough confirmations
        if len(current_detections) >= self.required_confirmations:
            return None

        # Find next indicator in sequence that hasn't been detected
        for indicator in self.indicator_sequence:
            if indicator not in current_detections:
                return indicator
        return None


@dataclass
class SetupContext:
    """
    Contains all data for the current active setup.

    Generic structure that works with any indicator sequence.
    """
    setup_id: int
    start_timestamp: datetime
    start_candle_index: int

    # Track which indicators have triggered (in order)
    detections: List[Detection] = field(default_factory=list)
    detected_indicators: List[str] = field(default_factory=list)

    # Overall direction (determined by consensus of signals)
    direction: SignalDirection = SignalDirection.NONE

    # Timing for timeouts
    last_detection_timestamp: Optional[datetime] = None
    last_detection_candle: Optional[int] = None

    def add_detection(self, detection: Detection):
        """Add a detection to this setup."""
        self.detections.append(detection)
        self.detected_indicators.append(detection.indicator_name)
        self.last_detection_timestamp = detection.timestamp
        self.last_detection_candle = detection.candle_index

        # Update direction based on detections
        self._update_direction()

    def _update_direction(self):
        """Determine overall direction from all detections."""
        if not self.detections:
            self.direction = SignalDirection.NONE
            return

        # Count votes
        long_votes = sum(1 for d in self.detections if d.direction == SignalDirection.LONG)
        short_votes = sum(1 for d in self.detections if d.direction == SignalDirection.SHORT)

        if long_votes > short_votes:
            self.direction = SignalDirection.LONG
        elif short_votes > long_votes:
            self.direction = SignalDirection.SHORT
        else:
            self.direction = SignalDirection.NONE

    def has_indicator(self, indicator_name: str) -> bool:
        """Check if specific indicator has been detected."""
        return indicator_name in self.detected_indicators

    def get_detection_count(self) -> int:
        """How many indicators have triggered."""
        return len(self.detections)

    def reset(self):
        """Clear all data for new setup."""
        self.detections.clear()
        self.detected_indicators.clear()
        self.direction = SignalDirection.NONE
        self.last_detection_timestamp = None
        self.last_detection_candle = None


@dataclass
class StateTransition:
    """Records a state change."""
    from_state: ExecutionState
    to_state: ExecutionState
    timestamp: datetime
    candle_index: int
    trigger: str  # What caused transition (indicator name or action)


@dataclass
class TradeExecution:
    """Represents an executed trade."""
    # Entry
    entry_timestamp: datetime
    entry_candle: int
    entry_price: float
    direction: SignalDirection

    # Risk management
    stop_loss: float
    take_profit: float
    position_size: float

    # What triggered this trade
    setup_detections: List[Detection]  # Copy of detections that led to trade

    # Exit (filled when closed)
    exit_timestamp: Optional[datetime] = None
    exit_candle: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Results
    pnl_pips: Optional[float] = None
    pnl_amount: Optional[float] = None

    def close(self, exit_timestamp: datetime, exit_candle: int,
              exit_price: float, reason: str):
        """Close the trade and calculate P&L."""
        self.exit_timestamp = exit_timestamp
        self.exit_candle = exit_candle
        self.exit_price = exit_price
        self.exit_reason = reason

        # Calculate P&L in pips
        if self.direction == SignalDirection.LONG:
            self.pnl_pips = (exit_price - self.entry_price) / 0.0001
        else:
            self.pnl_pips = (self.entry_price - exit_price) / 0.0001

        self.pnl_amount = self.pnl_pips * self.position_size


@dataclass
class BacktestState:
    """
    Complete state of the backtesting system.

    Flexible design that works with any strategy configuration.
    """
    # Strategy configuration
    strategy_config: StrategyConfig

    # Current state
    current_state: ExecutionState = ExecutionState.SCANNING
    current_state_index: int = 0  # Which signal number we're on

    # Single active setup
    active_setup: Optional[SetupContext] = None

    # Single active trade
    active_trade: Optional[TradeExecution] = None

    # History
    completed_trades: List[TradeExecution] = field(default_factory=list)
    state_history: List[StateTransition] = field(default_factory=list)

    # Statistics
    total_setups_started: int = 0
    total_setups_completed: int = 0
    total_setups_timeout: int = 0

    def can_start_new_setup(self) -> bool:
        """Check if we can start a new setup."""
        return (
            self.current_state == ExecutionState.SCANNING and
            self.active_setup is None and
            self.active_trade is None
        )

    def get_next_state(self) -> ExecutionState:
        """Get next state based on current progress."""
        state_progression = [
            ExecutionState.SCANNING,
            ExecutionState.SIGNAL_1,
            ExecutionState.SIGNAL_2,
            ExecutionState.SIGNAL_3,
            ExecutionState.SIGNAL_4,
            ExecutionState.SIGNAL_5,
            ExecutionState.READY_TO_ENTER
        ]

        try:
            current_idx = state_progression.index(self.current_state)
            if current_idx < len(state_progression) - 1:
                return state_progression[current_idx + 1]
        except ValueError:
            pass

        return self.current_state

    def check_timeout(self, current_candle: int) -> bool:
        """Check if current setup has timed out."""
        if not self.active_setup or self.active_setup.last_detection_candle is None:
            return False

        # Get timeout based on last detected indicator
        if self.active_setup.detected_indicators:
            last_indicator = self.active_setup.detected_indicators[-1]
            timeout = self.strategy_config.get_timeout_after(last_indicator)
        else:
            timeout = self.strategy_config.timeouts.get('default', 50)

        candles_elapsed = current_candle - self.active_setup.last_detection_candle
        return candles_elapsed > timeout

    def record_transition(self, from_state: ExecutionState, to_state: ExecutionState,
                         timestamp: datetime, candle_index: int, trigger: str):
        """Record a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            candle_index=candle_index,
            trigger=trigger
        )
        self.state_history.append(transition)
        self.current_state = to_state