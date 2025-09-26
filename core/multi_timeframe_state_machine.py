"""
Multi-timeframe state machine for cross-timeframe strategy execution.

This module implements the advanced state machine that coordinates trading
across multiple timeframes, handling complex synchronization, cross-timeframe
confirmation, and sophisticated trade execution logic.
"""

import logging
from typing import Dict, Optional, List, Any, Tuple, Set
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .state_machine import StateMachine
from .state_types import (
    ExecutionState,
    ActionType,
    SignalDirection,
    Detection,
    StrategyConfig
)
from .multi_timeframe_state import (
    MultiTimeframeExecutionState,
    TimeframeSyncStatus,
    TimeframeDetection,
    TimeframeState,
    MultiTimeframeSetupContext,
    MultiTimeframeBacktestState,
    MultiTimeframeStateTransition,
    create_multi_tf_setup,
    convert_legacy_detection_to_timeframe
)
from .timeframe_sync import (
    TimeframeSyncCoordinator,
    SyncConfiguration,
    SyncPolicy,
    SyncMode
)
from .multi_timeframe_detection import (
    MultiTimeframeDetectionProcessor,
    DetectionConfidence,
    ConfirmationStatus,
    create_detection_processor
)
from .json_converter import MultiTimeframeStrategyConfig
from .data_loader import DataLoader, TimeframeConverter


logger = logging.getLogger(__name__)


class MultiTimeframeActionType(Enum):
    """Extended action types for multi-timeframe execution."""
    WAIT = "wait"
    INDICATOR_DETECTED = "detected"
    TIMEFRAME_SYNC_REQUIRED = "sync_required"
    CROSS_TIMEFRAME_CONFIRMATION = "cross_tf_confirmation"
    ENTER_TRADE = "enter_trade"
    EXIT_TRADE = "exit_trade"
    RESET = "reset"
    TIMEOUT = "timeout"
    SYNC_TIMEOUT = "sync_timeout"
    CONFLICTING_SIGNALS = "conflicting_signals"


@dataclass
class TimeframeDataContext:
    """Context for managing data across multiple timeframes."""
    timeframe: str
    data: pd.DataFrame
    current_index: int
    last_processed_index: int = -1

    # Sync information
    sync_target: Optional[int] = None
    sync_quality: float = 1.0
    processing_lag: float = 0.0

    def is_ready_to_process(self) -> bool:
        """Check if this timeframe is ready for processing."""
        return self.current_index > self.last_processed_index

    def mark_processed(self):
        """Mark current index as processed."""
        self.last_processed_index = self.current_index


class MultiTimeframeStateMachine:
    """
    Advanced state machine for multi-timeframe strategy execution.

    Coordinates trading across multiple timeframes by:
    - Managing timeframe synchronization and alignment
    - Processing indicators across different timeframes
    - Handling cross-timeframe confirmations
    - Executing sophisticated trade entry/exit logic
    - Providing comprehensive state tracking and analytics
    """

    def __init__(
        self,
        multi_tf_config: MultiTimeframeStrategyConfig,
        indicators: Dict[str, Any],  # indicator_name -> BaseIndicator instance
        data_loader: DataLoader,
        sync_policy: SyncPolicy = SyncPolicy.MODERATE
    ):
        """
        Initialize the multi-timeframe state machine.

        Args:
            multi_tf_config: Multi-timeframe strategy configuration
            indicators: Dictionary mapping indicator names to instances
            data_loader: DataLoader instance for accessing market data
            sync_policy: Synchronization policy to use
        """
        self.multi_tf_config = multi_tf_config
        self.indicators = indicators
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

        # Create legacy config for parent compatibility
        legacy_config = multi_tf_config.to_legacy_config()

        # Initialize multi-timeframe state
        self.state = MultiTimeframeBacktestState(strategy_config=legacy_config)
        self.state.initialize_timeframes(multi_tf_config)

        # Initialize coordination components
        sync_config = SyncConfiguration(policy=sync_policy, mode=SyncMode.REAL_TIME)
        self.sync_coordinator = TimeframeSyncCoordinator(sync_config)
        self.detection_processor = create_detection_processor(multi_tf_config)

        # Timeframe data management
        self.timeframe_data: Dict[str, TimeframeDataContext] = {}
        self.timeframe_converter = TimeframeConverter()

        # Processing state
        self._setup_counter = 0
        self.active_timeframes = multi_tf_config.get_all_timeframes()
        self.primary_timeframe = multi_tf_config.primary_timeframe

        # Performance tracking
        self.processing_stats = {
            'total_candles_processed': 0,
            'cross_tf_confirmations': 0,
            'sync_attempts': 0,
            'successful_syncs': 0,
            'timeframe_conflicts': 0
        }

        # Validate configuration
        self._validate_multi_tf_configuration()

        self.logger.info(
            f"Multi-timeframe state machine initialized with {len(indicators)} indicators "
            f"across {len(self.active_timeframes)} timeframes"
        )

    def _validate_multi_tf_configuration(self) -> None:
        """Validate multi-timeframe configuration and indicators."""
        # Check that all required indicators are available
        missing_indicators = []
        for indicator_name in self.multi_tf_config.indicator_sequence:
            if indicator_name not in self.indicators:
                missing_indicators.append(indicator_name)

        if missing_indicators:
            raise ValueError(f"Missing indicators: {missing_indicators}")

        # Validate timeframe configuration
        if self.primary_timeframe not in self.active_timeframes:
            raise ValueError(f"Primary timeframe {self.primary_timeframe} not in active timeframes")

        # Validate indicator-timeframe mapping
        for indicator_name in self.multi_tf_config.indicator_sequence:
            expected_timeframes = self.multi_tf_config.get_indicators_for_timeframe
            # Additional validation logic can be added here

    def initialize_timeframe_data(self, start_date: str, end_date: str) -> None:
        """
        Initialize data contexts for all timeframes.

        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        self.logger.info(f"Initializing data for timeframes: {self.active_timeframes}")

        # Load data for each timeframe
        multi_tf_data = self.data_loader.load_multi_timeframe_data(
            timeframes=self.active_timeframes
        )

        # Create data contexts
        for timeframe in self.active_timeframes:
            if timeframe in multi_tf_data:
                data = multi_tf_data[timeframe]

                # Filter data by date range if specified
                if start_date and end_date:
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    data = data[mask]

                self.timeframe_data[timeframe] = TimeframeDataContext(
                    timeframe=timeframe,
                    data=data,
                    current_index=0
                )

                self.logger.debug(f"Loaded {len(data)} candles for {timeframe}")
            else:
                self.logger.warning(f"No data available for timeframe {timeframe}")

    def process_multi_timeframe_candle(self, primary_candle_index: int) -> MultiTimeframeActionType:
        """
        Process a candle across all timeframes with coordination.

        Args:
            primary_candle_index: Index of current candle in primary timeframe

        Returns:
            MultiTimeframeActionType indicating what action was taken
        """
        self.processing_stats['total_candles_processed'] += 1

        # Get primary timeframe data
        primary_context = self.timeframe_data.get(self.primary_timeframe)
        if not primary_context or primary_candle_index >= len(primary_context.data):
            return MultiTimeframeActionType.WAIT

        primary_context.current_index = primary_candle_index
        current_candle = primary_context.data.iloc[primary_candle_index]
        timestamp = current_candle.name if hasattr(current_candle.name, 'to_pydatetime') else datetime.now()

        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        self.logger.debug(
            f"Processing multi-TF candle {primary_candle_index} in state {self.state.current_tf_state.value}"
        )

        # Update equivalent indices for other timeframes
        self._sync_timeframe_indices(primary_candle_index, timestamp)

        # Route to appropriate multi-timeframe handler
        action = MultiTimeframeActionType.WAIT

        if self.state.current_tf_state == MultiTimeframeExecutionState.SCANNING:
            action = self._handle_multi_tf_scanning(primary_candle_index, timestamp)

        elif self.state.current_tf_state in [
            MultiTimeframeExecutionState.SIGNAL_1_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_2_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_3_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_4_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_5_TIMEFRAME_SYNC
        ]:
            action = self._handle_multi_tf_signal_state(primary_candle_index, timestamp)

        elif self.state.current_tf_state == MultiTimeframeExecutionState.TIMEFRAME_ALIGNMENT_CHECK:
            action = self._handle_timeframe_alignment_check(primary_candle_index, timestamp)

        elif self.state.current_tf_state == MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION:
            action = self._handle_cross_timeframe_confirmation(primary_candle_index, timestamp)

        elif self.state.current_tf_state == MultiTimeframeExecutionState.READY_TO_ENTER:
            action = self._handle_multi_tf_ready_to_enter(current_candle, primary_candle_index, timestamp)

        elif self.state.current_tf_state == MultiTimeframeExecutionState.IN_POSITION:
            action = self._handle_multi_tf_in_position(current_candle, primary_candle_index, timestamp)

        elif self.state.current_tf_state == MultiTimeframeExecutionState.POSITION_CLOSED:
            action = self._handle_multi_tf_position_closed(primary_candle_index, timestamp)

        # Mark primary timeframe as processed
        primary_context.mark_processed()

        return action

    def _sync_timeframe_indices(self, primary_index: int, timestamp: datetime) -> None:
        """Synchronize indices across all timeframes based on primary timeframe."""
        for timeframe, context in self.timeframe_data.items():
            if timeframe == self.primary_timeframe:
                continue

            try:
                # Convert primary index to equivalent index in this timeframe
                equivalent_index = self.timeframe_converter.convert_timeframe_index(
                    primary_index, self.primary_timeframe, timeframe
                )

                # Update sync target
                context.sync_target = equivalent_index

                # Update current index if within bounds
                if equivalent_index < len(context.data):
                    context.current_index = equivalent_index
                else:
                    # Handle case where equivalent index is beyond available data
                    context.current_index = len(context.data) - 1
                    context.processing_lag = equivalent_index - context.current_index

                self.logger.debug(
                    f"Synced {timeframe}: primary[{primary_index}] -> {timeframe}[{context.current_index}]"
                )

            except Exception as e:
                self.logger.warning(f"Failed to sync {timeframe} indices: {e}")

    def _handle_multi_tf_scanning(self, primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """
        Handle multi-timeframe scanning state.

        Args:
            primary_index: Current index in primary timeframe
            timestamp: Current timestamp

        Returns:
            Action type based on detection results
        """
        # Check indicators across all timeframes
        for timeframe, context in self.timeframe_data.items():
            if not context.is_ready_to_process():
                continue

            # Get current candle data for this timeframe
            candle_data = context.data.iloc[:context.current_index + 1]

            # Check indicators configured for this timeframe
            timeframe_indicators = self.multi_tf_config.get_indicators_for_timeframe(timeframe)

            for indicator_name in timeframe_indicators:
                if indicator_name not in self.indicators:
                    continue

                indicator = self.indicators[indicator_name]
                legacy_detection = indicator.check(candle_data, context.current_index)

                if legacy_detection:
                    # Convert to timeframe detection
                    tf_detection = convert_legacy_detection_to_timeframe(
                        legacy_detection, timeframe
                    )
                    tf_detection.timeframe_candle_index = context.current_index

                    self.logger.info(
                        f"Signal detected by {indicator_name} on {timeframe} at {tf_detection.price:.5f}"
                    )

                    # Process detection through multi-timeframe processor
                    processing_result = self._process_multi_tf_detection(tf_detection, timestamp, primary_index)

                    if processing_result == MultiTimeframeActionType.INDICATOR_DETECTED:
                        return processing_result

        return MultiTimeframeActionType.WAIT

    def _process_multi_tf_detection(self, detection: TimeframeDetection,
                                   timestamp: datetime, primary_index: int) -> MultiTimeframeActionType:
        """
        Process a multi-timeframe detection and create setup if needed.

        Args:
            detection: Timeframe detection to process
            timestamp: Current timestamp
            primary_index: Current primary timeframe index

        Returns:
            Action type based on processing result
        """
        # Create new setup if none exists
        if not self.state.multi_tf_setup:
            self._setup_counter += 1
            self.state.multi_tf_setup = create_multi_tf_setup(
                setup_id=self._setup_counter,
                timestamp=timestamp,
                candle_index=primary_index,
                config=self.multi_tf_config
            )
            self.state.total_setups_started += 1

        # Process detection through detection processor
        result = self.detection_processor.process_detection(detection, self.state.multi_tf_setup)

        if result['accepted']:
            # Add to setup context
            self.state.multi_tf_setup.add_timeframe_detection(detection)

            # Check if synchronization is required
            if result.get('confirmation_required', False):
                self._transition_multi_tf_state(
                    MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION,
                    timestamp, primary_index, f"confirmation_required_{detection.indicator_name}",
                    detection.timeframe
                )
                return MultiTimeframeActionType.CROSS_TIMEFRAME_CONFIRMATION
            else:
                # Direct progression to next signal state
                next_state = self._get_next_multi_tf_signal_state()
                self._transition_multi_tf_state(
                    next_state, timestamp, primary_index, f"detected_{detection.indicator_name}",
                    detection.timeframe
                )
                return MultiTimeframeActionType.INDICATOR_DETECTED

        elif result['recommendation'] == 'wait_confirmation':
            # Need to wait for cross-timeframe confirmation
            self._transition_multi_tf_state(
                MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION,
                timestamp, primary_index, f"awaiting_confirmation_{detection.indicator_name}",
                detection.timeframe
            )
            return MultiTimeframeActionType.CROSS_TIMEFRAME_CONFIRMATION

        return MultiTimeframeActionType.WAIT

    def _handle_multi_tf_signal_state(self, primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle multi-timeframe signal states."""
        if not self.state.multi_tf_setup:
            self.logger.error("No active setup in signal state")
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "error_no_setup")
            return MultiTimeframeActionType.RESET

        # Check for timeout
        if self.state.check_multi_tf_timeout(primary_index):
            self.logger.info("Multi-timeframe setup timed out")
            self.state.total_setups_timeout += 1
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "timeout")
            return MultiTimeframeActionType.TIMEOUT

        # Process pending confirmations
        self.detection_processor.process_timeouts()

        # Check if we have enough detections to proceed
        required_confirmations = self.multi_tf_config.required_confirmations
        current_detections = len(self.state.multi_tf_setup.detections)

        if current_detections >= required_confirmations:
            # Check sync quality
            if self.state.multi_tf_setup.sync_status == TimeframeSyncStatus.SYNCED:
                self._transition_multi_tf_state(
                    MultiTimeframeExecutionState.READY_TO_ENTER,
                    timestamp, primary_index, "confirmations_complete"
                )
                return MultiTimeframeActionType.INDICATOR_DETECTED
            else:
                # Need better synchronization
                self._transition_multi_tf_state(
                    MultiTimeframeExecutionState.TIMEFRAME_ALIGNMENT_CHECK,
                    timestamp, primary_index, "sync_required"
                )
                return MultiTimeframeActionType.TIMEFRAME_SYNC_REQUIRED

        # Continue looking for more detections
        return self._handle_multi_tf_scanning(primary_index, timestamp)

    def _handle_timeframe_alignment_check(self, primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle timeframe alignment checking."""
        if not self.state.multi_tf_setup:
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "no_setup")
            return MultiTimeframeActionType.RESET

        # Get latest detection to use as sync anchor
        latest_detection = None
        for detections in self.state.multi_tf_setup.detections_by_timeframe.values():
            if detections:
                detection = max(detections, key=lambda d: d.timestamp)
                if not latest_detection or detection.timestamp > latest_detection.timestamp:
                    latest_detection = detection

        if latest_detection:
            # Attempt synchronization
            self.processing_stats['sync_attempts'] += 1
            sync_success = self.sync_coordinator.synchronize_timeframes(
                self.state.multi_tf_setup, latest_detection.timeframe, latest_detection
            )

            if sync_success:
                self.processing_stats['successful_syncs'] += 1
                self._transition_multi_tf_state(
                    MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION,
                    timestamp, primary_index, "sync_successful"
                )
                return MultiTimeframeActionType.INDICATOR_DETECTED
            else:
                # Sync failed, check if we can proceed anyway
                if self.state.multi_tf_setup.sync_quality_score >= 0.6:
                    self.logger.warning("Proceeding with imperfect sync")
                    self._transition_multi_tf_state(
                        MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION,
                        timestamp, primary_index, "sync_degraded"
                    )
                    return MultiTimeframeActionType.CROSS_TIMEFRAME_CONFIRMATION
                else:
                    # Sync quality too poor, reset
                    self._reset_to_multi_tf_scanning(timestamp, primary_index, "sync_failed")
                    return MultiTimeframeActionType.SYNC_TIMEOUT

        return MultiTimeframeActionType.WAIT

    def _handle_cross_timeframe_confirmation(self, primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle cross-timeframe confirmation state."""
        if not self.state.multi_tf_setup:
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "no_setup")
            return MultiTimeframeActionType.RESET

        self.processing_stats['cross_tf_confirmations'] += 1

        # Check confirmation requests
        confirmed_requests = 0
        total_requests = len(self.detection_processor.confirmation_requests)

        for request in self.detection_processor.confirmation_requests.values():
            if request.status == ConfirmationStatus.CONFIRMED:
                confirmed_requests += 1
            elif request.status == ConfirmationStatus.TIMEOUT:
                # Handle timeout
                self._reset_to_multi_tf_scanning(timestamp, primary_index, "confirmation_timeout")
                return MultiTimeframeActionType.TIMEOUT

        # Check if we have enough confirmations
        confirmation_ratio = confirmed_requests / total_requests if total_requests > 0 else 1.0

        if confirmation_ratio >= 0.7:  # 70% confirmation rate required
            self._transition_multi_tf_state(
                MultiTimeframeExecutionState.READY_TO_ENTER,
                timestamp, primary_index, "cross_tf_confirmed"
            )
            return MultiTimeframeActionType.INDICATOR_DETECTED
        elif confirmation_ratio < 0.3:  # Too many rejections
            self.processing_stats['timeframe_conflicts'] += 1
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "conflicting_signals")
            return MultiTimeframeActionType.CONFLICTING_SIGNALS

        return MultiTimeframeActionType.WAIT

    def _handle_multi_tf_ready_to_enter(self, current_candle: pd.Series,
                                       primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle multi-timeframe ready to enter state."""
        if not self.state.multi_tf_setup:
            self.logger.error("No active setup for trade entry")
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "error_no_setup")
            return MultiTimeframeActionType.RESET

        # Get consensus direction from multi-timeframe setup
        direction = self.state.multi_tf_setup.direction
        if direction == SignalDirection.NONE:
            self.logger.warning("No clear direction from multi-timeframe indicators")
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "no_direction")
            return MultiTimeframeActionType.RESET

        # Execute trade with multi-timeframe context
        entry_price = current_candle['close']
        trade = self._execute_multi_tf_trade(
            entry_price, direction, timestamp, primary_index
        )

        self.state.active_trade = trade
        self.state.total_setups_completed += 1

        # Transition to IN_POSITION
        self._transition_multi_tf_state(
            MultiTimeframeExecutionState.IN_POSITION,
            timestamp, primary_index, "trade_entered"
        )

        self.logger.info(
            f"Multi-TF trade entered: {direction.value} at {entry_price:.5f} "
            f"(quality: {self.state.multi_tf_setup.sync_quality_score:.2f})"
        )

        return MultiTimeframeActionType.ENTER_TRADE

    def _handle_multi_tf_in_position(self, current_candle: pd.Series,
                                    primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle multi-timeframe in position state."""
        # Use base class logic but with multi-timeframe enhancements
        if not self.state.active_trade:
            self.logger.error("No active trade in IN_POSITION state")
            self._reset_to_multi_tf_scanning(timestamp, primary_index, "error_no_trade")
            return MultiTimeframeActionType.RESET

        trade = self.state.active_trade
        exit_reason = None
        exit_price = None

        # Enhanced exit logic with multi-timeframe considerations
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

        # Check for multi-timeframe exit signals
        if not exit_reason:
            exit_signal = self._check_multi_tf_exit_signals(primary_index)
            if exit_signal:
                exit_price = current_candle['close']
                exit_reason = f"multi_tf_exit_{exit_signal}"

        if exit_reason:
            # Close trade
            trade.close(timestamp, primary_index, exit_price, exit_reason)
            self.state.completed_trades.append(trade)
            self.state.active_trade = None

            # Transition to POSITION_CLOSED
            self._transition_multi_tf_state(
                MultiTimeframeExecutionState.POSITION_CLOSED,
                timestamp, primary_index, exit_reason
            )

            self.logger.info(
                f"Multi-TF trade closed: {exit_reason} at {exit_price:.5f} "
                f"(P&L: {trade.pnl_pips:.1f} pips)"
            )

            return MultiTimeframeActionType.EXIT_TRADE

        return MultiTimeframeActionType.WAIT

    def _handle_multi_tf_position_closed(self, primary_index: int, timestamp: datetime) -> MultiTimeframeActionType:
        """Handle multi-timeframe position closed state."""
        # Clean up and reset to scanning
        self._reset_to_multi_tf_scanning(timestamp, primary_index, "position_closed")
        return MultiTimeframeActionType.RESET

    def _check_multi_tf_exit_signals(self, primary_index: int) -> Optional[str]:
        """Check for exit signals across timeframes."""
        # Implement multi-timeframe exit logic
        # This could include checking for conflicting signals, momentum shifts, etc.

        # For now, return None (no exit signal)
        # This can be extended with sophisticated exit logic
        return None

    def _execute_multi_tf_trade(self, entry_price: float, direction: SignalDirection,
                               timestamp: datetime, candle_index: int):
        """Execute trade with multi-timeframe context."""
        from .state_types import TradeExecution

        # Calculate position size based on risk
        position_size = self.multi_tf_config.risk_percent * 10000  # Convert to units

        # Calculate stop loss and take profit
        stop_loss_pips = self.multi_tf_config.stop_loss_pips
        take_profit_pips = self.multi_tf_config.take_profit_pips

        pip_size = 0.0001  # Standard for most pairs

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (stop_loss_pips * pip_size)
            take_profit = entry_price + (take_profit_pips * pip_size)
        else:
            stop_loss = entry_price + (stop_loss_pips * pip_size)
            take_profit = entry_price - (take_profit_pips * pip_size)

        # Create trade execution with multi-timeframe setup context
        trade = TradeExecution(
            entry_timestamp=timestamp,
            entry_candle=candle_index,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            setup_detections=list(self.state.multi_tf_setup.detections)
        )

        return trade

    def _get_next_multi_tf_signal_state(self) -> MultiTimeframeExecutionState:
        """Get next multi-timeframe signal state based on current detections."""
        if not self.state.multi_tf_setup:
            return MultiTimeframeExecutionState.SCANNING

        detection_count = len(self.state.multi_tf_setup.detections)

        state_progression = [
            MultiTimeframeExecutionState.SCANNING,
            MultiTimeframeExecutionState.SIGNAL_1_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_2_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_3_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_4_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_5_TIMEFRAME_SYNC
        ]

        target_index = min(detection_count + 1, len(state_progression) - 1)
        return state_progression[target_index]

    def _transition_multi_tf_state(self, new_state: MultiTimeframeExecutionState,
                                  timestamp: datetime, candle_index: int,
                                  trigger: str, timeframe: Optional[str] = None):
        """Transition to new multi-timeframe state with logging."""
        old_state = self.state.current_tf_state
        self.state.record_multi_tf_transition(
            old_state, new_state, timestamp, candle_index, trigger, timeframe
        )

        self.logger.debug(
            f"State transition: {old_state.value} -> {new_state.value} "
            f"(trigger: {trigger}, timeframe: {timeframe})"
        )

    def _reset_to_multi_tf_scanning(self, timestamp: datetime, candle_index: int, reason: str):
        """Reset state machine to multi-timeframe scanning."""
        # Clean up current setup
        if self.state.multi_tf_setup:
            self.state.multi_tf_setup.reset()
            self.state.multi_tf_setup = None

        # Reset detection processor
        self.detection_processor.reset()

        # Transition to scanning
        self._transition_multi_tf_state(
            MultiTimeframeExecutionState.SCANNING,
            timestamp, candle_index, reason
        )

        self.logger.info(f"Reset to scanning: {reason}")

    def get_multi_tf_statistics(self) -> Dict[str, Any]:
        """Get comprehensive multi-timeframe statistics."""
        base_stats = self.state.get_sync_statistics()
        detection_stats = self.detection_processor.get_detection_statistics()
        sync_stats = self.sync_coordinator.get_sync_statistics()

        return {
            **base_stats,
            **detection_stats,
            **sync_stats,
            'processing_stats': self.processing_stats,
            'active_timeframes': self.active_timeframes,
            'primary_timeframe': self.primary_timeframe,
            'current_state': self.state.current_tf_state.value,
            'sync_quality': self.state.multi_tf_setup.sync_quality_score if self.state.multi_tf_setup else 0.0
        }

    def get_timeframe_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of timeframe coordination."""
        analysis = {
            'timeframe_data_status': {},
            'sync_quality_by_timeframe': {},
            'processing_lag_by_timeframe': {},
            'detection_count_by_timeframe': {}
        }

        for timeframe, context in self.timeframe_data.items():
            analysis['timeframe_data_status'][timeframe] = {
                'current_index': context.current_index,
                'last_processed': context.last_processed_index,
                'total_candles': len(context.data),
                'processing_progress': context.last_processed_index / len(context.data) if len(context.data) > 0 else 0.0
            }

            analysis['sync_quality_by_timeframe'][timeframe] = context.sync_quality
            analysis['processing_lag_by_timeframe'][timeframe] = context.processing_lag

        if self.state.multi_tf_setup:
            for timeframe, detections in self.state.multi_tf_setup.detections_by_timeframe.items():
                analysis['detection_count_by_timeframe'][timeframe] = len(detections)

        return analysis


# Utility functions for multi-timeframe state machine
def create_multi_tf_state_machine(
    config: MultiTimeframeStrategyConfig,
    indicators: Dict[str, Any],
    data_loader: DataLoader,
    sync_policy: SyncPolicy = SyncPolicy.MODERATE
) -> MultiTimeframeStateMachine:
    """
    Factory function to create a multi-timeframe state machine.

    Args:
        config: Multi-timeframe strategy configuration
        indicators: Dictionary of indicator instances
        data_loader: Data loader instance
        sync_policy: Synchronization policy

    Returns:
        Configured MultiTimeframeStateMachine instance
    """
    return MultiTimeframeStateMachine(config, indicators, data_loader, sync_policy)