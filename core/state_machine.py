"""
State machine for sequential trading strategy execution.

This module implements the core state machine that orchestrates the entire
trading flow, from signal detection through trade execution and position
management.
"""

import logging
from typing import Dict, Optional, List, Any, Tuple
import pandas as pd
from datetime import datetime

from .state_types import (
    ExecutionState,
    ActionType,
    SignalDirection,
    Detection,
    StrategyConfig,
    SetupContext,
    StateTransition,
    TradeExecution,
    BacktestState
)
from .data_loader import DataLoader


class StateMachine:
    """
    Main state machine for managing trading strategy execution.

    This class orchestrates the entire trading flow by:
    - Processing each candle through the appropriate state handler
    - Managing state transitions based on indicator detections
    - Executing trades when setup conditions are met
    - Managing open positions and exits
    - Tracking all state history and completed trades
    """

    def __init__(
        self,
        strategy_config: StrategyConfig,
        indicators: Dict[str, Any],  # Will be BaseIndicator instances
        data_loader: DataLoader
    ):
        """
        Initialize the state machine.

        Args:
            strategy_config: Strategy configuration defining rules and sequence
            indicators: Dictionary mapping indicator names to instances
            data_loader: DataLoader instance for accessing market data
        """
        self.strategy_config = strategy_config
        self.indicators = indicators
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

        # Initialize backtest state
        self.state = BacktestState(strategy_config=strategy_config)

        # Validate configuration
        self._validate_configuration()

        # Setup counter for unique IDs
        self._setup_counter = 0

        self.logger.info(
            f"State machine initialized with {len(indicators)} indicators"
        )

    def _validate_configuration(self) -> None:
        """Validate that all required indicators are available."""
        missing = []
        for indicator_name in self.strategy_config.indicator_sequence:
            if indicator_name not in self.indicators:
                missing.append(indicator_name)

        if missing:
            raise ValueError(
                f"Missing indicators in configuration: {missing}"
            )

    def process_candle(
        self,
        candle_data: pd.DataFrame,
        candle_index: int,
        current_candle: pd.Series
    ) -> ActionType:
        """
        Process a single candle through the state machine.

        Args:
            candle_data: Historical data including current candle
            candle_index: Index of current candle in backtest
            current_candle: Current candle data (last row of candle_data)

        Returns:
            ActionType indicating what action was taken
        """
        # Get current timestamp
        timestamp = current_candle['datetime']

        # Log current state
        self.logger.debug(
            f"Processing candle {candle_index} in state {self.state.current_state.value}"
        )

        # Route to appropriate handler based on current state
        action = ActionType.WAIT

        if self.state.current_state == ExecutionState.SCANNING:
            action = self._handle_scanning_state(candle_data, candle_index, timestamp)

        elif self.state.current_state in [
            ExecutionState.SIGNAL_1, ExecutionState.SIGNAL_2,
            ExecutionState.SIGNAL_3, ExecutionState.SIGNAL_4,
            ExecutionState.SIGNAL_5
        ]:
            action = self._handle_signal_state(candle_data, candle_index, timestamp)

        elif self.state.current_state == ExecutionState.READY_TO_ENTER:
            action = self._handle_ready_to_enter(current_candle, candle_index, timestamp)

        elif self.state.current_state == ExecutionState.IN_POSITION:
            action = self._handle_in_position(current_candle, candle_index, timestamp)

        elif self.state.current_state == ExecutionState.POSITION_CLOSED:
            action = self._handle_position_closed(candle_index, timestamp)

        return action

    def _handle_scanning_state(
        self,
        candle_data: pd.DataFrame,
        candle_index: int,
        timestamp: datetime
    ) -> ActionType:
        """
        Handle SCANNING state - look for any indicator to trigger.

        Args:
            candle_data: Historical data
            candle_index: Current candle index
            timestamp: Current timestamp

        Returns:
            INDICATOR_DETECTED if signal found, WAIT otherwise
        """
        # Check all indicators for a signal
        for indicator_name, indicator in self.indicators.items():
            detection = indicator.check(candle_data, candle_index)

            if detection:
                self.logger.info(
                    f"Signal detected by {indicator_name} at {detection.price:.5f}"
                )

                # Create new setup
                self._setup_counter += 1
                self.state.active_setup = SetupContext(
                    setup_id=self._setup_counter,
                    start_timestamp=timestamp,
                    start_candle_index=candle_index
                )

                # Add detection to setup
                self.state.active_setup.add_detection(detection)
                self.state.total_setups_started += 1

                # Transition to SIGNAL_1
                self._transition_state(
                    ExecutionState.SIGNAL_1,
                    timestamp,
                    candle_index,
                    f"detected_{indicator_name}"
                )

                return ActionType.INDICATOR_DETECTED

        return ActionType.WAIT

    def _handle_signal_state(
        self,
        candle_data: pd.DataFrame,
        candle_index: int,
        timestamp: datetime
    ) -> ActionType:
        """
        Handle SIGNAL_X states - look for next required indicator.

        Args:
            candle_data: Historical data
            candle_index: Current candle index
            timestamp: Current timestamp

        Returns:
            INDICATOR_DETECTED, TIMEOUT, or WAIT
        """
        if not self.state.active_setup:
            self.logger.error("No active setup in signal state")
            self._reset_to_scanning(timestamp, candle_index, "error_no_setup")
            return ActionType.RESET

        # Check for timeout
        if self.state.check_timeout(candle_index):
            self.logger.info("Setup timed out")
            self.state.total_setups_timeout += 1
            self._reset_to_scanning(timestamp, candle_index, "timeout")
            return ActionType.TIMEOUT

        # Get next indicator to check
        next_indicator = self.strategy_config.get_next_indicator(
            self.state.active_setup.detected_indicators
        )

        if not next_indicator:
            # We have enough confirmations, move to ready
            self.logger.info("All required confirmations received")
            self._transition_state(
                ExecutionState.READY_TO_ENTER,
                timestamp,
                candle_index,
                "confirmations_complete"
            )
            return ActionType.INDICATOR_DETECTED

        # Check only the next required indicator
        if next_indicator in self.indicators:
            indicator = self.indicators[next_indicator]
            detection = indicator.check(candle_data, candle_index)

            if detection:
                self.logger.info(
                    f"Signal detected by {next_indicator} at {detection.price:.5f}"
                )

                # Add to setup
                self.state.active_setup.add_detection(detection)

                # Determine next state
                next_state = self._get_next_signal_state()
                self._transition_state(
                    next_state,
                    timestamp,
                    candle_index,
                    f"detected_{next_indicator}"
                )

                # Check if we now have enough confirmations
                if self.strategy_config.get_next_indicator(
                    self.state.active_setup.detected_indicators
                ) is None:
                    self._transition_state(
                        ExecutionState.READY_TO_ENTER,
                        timestamp,
                        candle_index,
                        "confirmations_complete"
                    )

                return ActionType.INDICATOR_DETECTED

        return ActionType.WAIT

    def _handle_ready_to_enter(
        self,
        current_candle: pd.Series,
        candle_index: int,
        timestamp: datetime
    ) -> ActionType:
        """
        Handle READY_TO_ENTER state - execute trade.

        Args:
            current_candle: Current candle data
            candle_index: Current candle index
            timestamp: Current timestamp

        Returns:
            ENTER_TRADE
        """
        if not self.state.active_setup:
            self.logger.error("No active setup for trade entry")
            self._reset_to_scanning(timestamp, candle_index, "error_no_setup")
            return ActionType.RESET

        # Get trade direction from setup consensus
        direction = self.state.active_setup.direction
        if direction == SignalDirection.NONE:
            self.logger.warning("No clear direction from indicators")
            self._reset_to_scanning(timestamp, candle_index, "no_direction")
            return ActionType.RESET

        # Execute trade
        entry_price = current_candle['close']
        trade = self._execute_trade(
            entry_price,
            direction,
            timestamp,
            candle_index
        )

        self.state.active_trade = trade
        self.state.total_setups_completed += 1

        # Transition to IN_POSITION
        self._transition_state(
            ExecutionState.IN_POSITION,
            timestamp,
            candle_index,
            "trade_entered"
        )

        self.logger.info(
            f"Trade entered: {direction.value} at {entry_price:.5f}"
        )

        return ActionType.ENTER_TRADE

    def _handle_in_position(
        self,
        current_candle: pd.Series,
        candle_index: int,
        timestamp: datetime
    ) -> ActionType:
        """
        Handle IN_POSITION state - manage open trade.

        Args:
            current_candle: Current candle data
            candle_index: Current candle index
            timestamp: Current timestamp

        Returns:
            EXIT_TRADE or WAIT
        """
        if not self.state.active_trade:
            self.logger.error("No active trade in IN_POSITION state")
            self._reset_to_scanning(timestamp, candle_index, "error_no_trade")
            return ActionType.RESET

        trade = self.state.active_trade
        exit_reason = None
        exit_price = None

        # Check stop loss
        if trade.direction == SignalDirection.LONG:
            if current_candle['low'] <= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif current_candle['high'] >= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"
        else:  # SHORT
            if current_candle['high'] >= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif current_candle['low'] <= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"

        if exit_reason:
            # Close position
            self._close_position(
                exit_price,
                exit_reason,
                timestamp,
                candle_index
            )

            # Transition to POSITION_CLOSED
            self._transition_state(
                ExecutionState.POSITION_CLOSED,
                timestamp,
                candle_index,
                exit_reason
            )

            return ActionType.EXIT_TRADE

        return ActionType.WAIT

    def _handle_position_closed(
        self,
        candle_index: int,
        timestamp: datetime
    ) -> ActionType:
        """
        Handle POSITION_CLOSED state - immediate transition to SCANNING.

        Args:
            candle_index: Current candle index
            timestamp: Current timestamp

        Returns:
            RESET
        """
        # Immediately reset to scanning
        self._reset_to_scanning(timestamp, candle_index, "position_closed")
        return ActionType.RESET

    def _transition_state(
        self,
        new_state: ExecutionState,
        timestamp: datetime,
        candle_index: int,
        trigger: str
    ) -> None:
        """
        Transition to a new state and record the transition.

        Args:
            new_state: State to transition to
            timestamp: Current timestamp
            candle_index: Current candle index
            trigger: What triggered the transition
        """
        old_state = self.state.current_state
        self.state.record_transition(
            old_state,
            new_state,
            timestamp,
            candle_index,
            trigger
        )

        self.logger.debug(
            f"State transition: {old_state.value} -> {new_state.value} ({trigger})"
        )

    def _reset_to_scanning(
        self,
        timestamp: datetime,
        candle_index: int,
        reason: str
    ) -> None:
        """
        Reset state machine to SCANNING state.

        Args:
            timestamp: Current timestamp
            candle_index: Current candle index
            reason: Why reset occurred
        """
        # Clear active setup
        if self.state.active_setup:
            self.state.active_setup.reset()
            self.state.active_setup = None

        # Clear active trade (should not happen, but safety check)
        if self.state.active_trade:
            self.logger.warning("Active trade cleared during reset")
            self.state.active_trade = None

        # Reset all indicators
        for indicator in self.indicators.values():
            indicator.reset()

        # Transition to SCANNING
        self._transition_state(
            ExecutionState.SCANNING,
            timestamp,
            candle_index,
            reason
        )


    def _get_next_signal_state(self) -> ExecutionState:
        """
        Get the next SIGNAL_X state based on current detections.

        Returns:
            Next signal state
        """
        if not self.state.active_setup:
            return ExecutionState.SIGNAL_1

        num_detections = len(self.state.active_setup.detections)

        state_map = {
            1: ExecutionState.SIGNAL_1,
            2: ExecutionState.SIGNAL_2,
            3: ExecutionState.SIGNAL_3,
            4: ExecutionState.SIGNAL_4,
            5: ExecutionState.SIGNAL_5,
        }

        return state_map.get(num_detections, ExecutionState.SIGNAL_5)

    def _execute_trade(
        self,
        entry_price: float,
        direction: SignalDirection,
        timestamp: datetime,
        candle_index: int
    ) -> TradeExecution:
        """
        Execute a trade with proper risk management.

        Args:
            entry_price: Price to enter at
            direction: Trade direction (LONG/SHORT)
            timestamp: Entry timestamp
            candle_index: Entry candle index

        Returns:
            TradeExecution object
        """
        # Calculate stop loss and take profit
        pip_value = 0.0001

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (self.strategy_config.stop_loss_pips * pip_value)
            take_profit = entry_price + (self.strategy_config.take_profit_pips * pip_value)
        else:  # SHORT
            stop_loss = entry_price + (self.strategy_config.stop_loss_pips * pip_value)
            take_profit = entry_price - (self.strategy_config.take_profit_pips * pip_value)

        # Calculate position size (simplified - could be enhanced)
        # Using fixed lot size for now, could implement percentage risk later
        position_size = self.strategy_config.risk_percent * 100000  # Standard lot

        # Create trade execution
        trade = TradeExecution(
            entry_timestamp=timestamp,
            entry_candle=candle_index,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            setup_detections=list(self.state.active_setup.detections)
        )

        return trade

    def _close_position(
        self,
        exit_price: float,
        exit_reason: str,
        timestamp: datetime,
        candle_index: int
    ) -> None:
        """
        Close the active position.

        Args:
            exit_price: Price to exit at
            exit_reason: Reason for exit
            timestamp: Exit timestamp
            candle_index: Exit candle index
        """
        if not self.state.active_trade:
            return

        # Close the trade
        self.state.active_trade.close(
            exit_timestamp=timestamp,
            exit_candle=candle_index,
            exit_price=exit_price,
            reason=exit_reason
        )

        # Add to completed trades
        self.state.completed_trades.append(self.state.active_trade)

        self.logger.info(
            f"Trade closed: {exit_reason} at {exit_price:.5f}, "
            f"P&L: {self.state.active_trade.pnl_pips:.1f} pips"
        )

        # Clear active trade
        self.state.active_trade = None

    def get_state(self) -> BacktestState:
        """
        Get the current backtest state.

        Returns:
            Current BacktestState object
        """
        return self.state

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary of statistics
        """
        total_trades = len(self.state.completed_trades)
        winning_trades = sum(
            1 for t in self.state.completed_trades
            if t.pnl_pips and t.pnl_pips > 0
        )
        losing_trades = sum(
            1 for t in self.state.completed_trades
            if t.pnl_pips and t.pnl_pips < 0
        )

        total_pips = sum(
            t.pnl_pips for t in self.state.completed_trades
            if t.pnl_pips
        )

        return {
            'current_state': self.state.current_state.value,
            'total_setups_started': self.state.total_setups_started,
            'total_setups_completed': self.state.total_setups_completed,
            'total_setups_timeout': self.state.total_setups_timeout,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pips': total_pips,
            'active_setup': self.state.active_setup is not None,
            'active_trade': self.state.active_trade is not None
        }