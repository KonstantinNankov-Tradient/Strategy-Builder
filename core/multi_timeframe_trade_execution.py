"""
Multi-timeframe trade execution system.

This module handles trade execution across multiple timeframes, coordinating
entry and exit decisions based on cross-timeframe analysis and risk management.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from .multi_timeframe_state import (
    TimeframeDetection,
    MultiTimeframeSetupContext,
    MultiTimeframeBacktestState,
    TimeframeSyncStatus
)
from .multi_timeframe_detection import DetectionCluster
from .json_converter import MultiTimeframeStrategyConfig
from .data_loader import TimeframeConverter


logger = logging.getLogger(__name__)


class TradeDecisionType(Enum):
    """Types of trade decisions."""
    NO_ACTION = "no_action"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_POSITION = "exit_position"
    MODIFY_POSITION = "modify_position"
    WAIT_FOR_CONFIRMATION = "wait_for_confirmation"


class ExecutionTimingMode(Enum):
    """Trade execution timing modes."""
    IMMEDIATE = "immediate"         # Execute immediately when conditions are met
    NEXT_CANDLE = "next_candle"     # Wait for next candle open
    BREAKOUT = "breakout"           # Execute on price breakout
    PULLBACK = "pullback"           # Execute on pullback to entry zone


@dataclass
class TradeExecutionContext:
    """Context for trade execution decisions."""
    primary_timeframe: str
    execution_timeframe: str  # The timeframe to execute trades on

    # Detection information
    active_detections: List[TimeframeDetection] = field(default_factory=list)
    confirming_detections: List[TimeframeDetection] = field(default_factory=list)
    conflicting_detections: List[TimeframeDetection] = field(default_factory=list)

    # Risk parameters
    risk_percentage: float = 2.0
    max_position_size: float = 1.0
    stop_loss_buffer: float = 0.0005
    take_profit_ratio: float = 2.0

    # Execution settings
    timing_mode: ExecutionTimingMode = ExecutionTimingMode.IMMEDIATE
    require_all_timeframes: bool = False
    min_confirmation_count: int = 1

    # Current state
    current_position_size: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class TradeDecision:
    """Represents a trade execution decision."""
    decision_type: TradeDecisionType
    timeframe: str
    price: float
    size: Optional[float] = None

    # Trade parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Supporting information
    primary_detection: Optional[TimeframeDetection] = None
    supporting_detections: List[TimeframeDetection] = field(default_factory=list)
    confidence_score: float = 0.0

    # Execution details
    execution_timestamp: datetime = field(default_factory=datetime.now)
    execution_reason: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


class MultiTimeframeTradeExecutor:
    """
    Executes trades based on multi-timeframe analysis.

    Coordinates trade decisions across multiple timeframes, handling entry/exit
    logic with proper risk management and timing considerations.
    """

    def __init__(self, strategy_config: MultiTimeframeStrategyConfig):
        """
        Initialize the trade executor.

        Args:
            strategy_config: Multi-timeframe strategy configuration
        """
        self.strategy_config = strategy_config
        self.timeframe_converter = TimeframeConverter()
        self.logger = logging.getLogger(__name__)

        # Execution context
        self.execution_context = TradeExecutionContext(
            primary_timeframe=strategy_config.primary_timeframe,
            execution_timeframe=strategy_config.primary_timeframe  # Default to primary
        )

        # Performance tracking
        self.execution_stats = {
            'total_decisions': 0,
            'trades_executed': 0,
            'trades_skipped': 0,
            'avg_confidence': 0.0,
            'decision_accuracy': 0.0
        }

        # Risk management settings
        self.max_concurrent_trades = 3
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        self.last_trade_date = None

    def evaluate_trade_opportunity(
        self,
        setup_context: MultiTimeframeSetupContext,
        timeframe_data: Dict[str, Any],
        current_state: MultiTimeframeBacktestState
    ) -> TradeDecision:
        """
        Evaluate whether to execute a trade based on multi-timeframe setup.

        Args:
            setup_context: Current multi-timeframe setup
            timeframe_data: Data contexts for all timeframes
            current_state: Current backtesting state

        Returns:
            Trade decision with execution parameters
        """
        start_time = datetime.now()
        self.execution_stats['total_decisions'] += 1

        try:
            # Update execution context with current setup
            self._update_execution_context(setup_context, timeframe_data, current_state)

            # Analyze cross-timeframe alignment
            alignment_analysis = self._analyze_timeframe_alignment(setup_context)

            # Evaluate trade conditions
            trade_decision = self._evaluate_trade_conditions(
                setup_context, timeframe_data, alignment_analysis
            )

            # Apply risk management filters
            trade_decision = self._apply_risk_management(trade_decision, current_state)

            # Calculate confidence score
            trade_decision.confidence_score = self._calculate_confidence_score(
                trade_decision, alignment_analysis
            )

            # Log decision
            self.logger.info(
                f"Trade decision: {trade_decision.decision_type.value} "
                f"@ {trade_decision.price:.5f} (confidence: {trade_decision.confidence_score:.2f})"
            )

            if trade_decision.decision_type in [TradeDecisionType.ENTER_LONG, TradeDecisionType.ENTER_SHORT]:
                self.execution_stats['trades_executed'] += 1
            else:
                self.execution_stats['trades_skipped'] += 1

            return trade_decision

        except Exception as e:
            self.logger.error(f"Trade evaluation failed: {e}")
            return TradeDecision(
                decision_type=TradeDecisionType.NO_ACTION,
                timeframe=self.execution_context.primary_timeframe,
                price=0.0,
                execution_reason=f"Error: {str(e)}"
            )

    def _update_execution_context(
        self,
        setup_context: MultiTimeframeSetupContext,
        timeframe_data: Dict[str, Any],
        current_state: MultiTimeframeBacktestState
    ):
        """Update execution context with current information."""
        # Clear previous detections
        self.execution_context.active_detections.clear()
        self.execution_context.confirming_detections.clear()
        self.execution_context.conflicting_detections.clear()

        # Categorize detections by alignment
        for detection in setup_context.all_detections:
            if detection.timeframe == self.execution_context.primary_timeframe:
                self.execution_context.active_detections.append(detection)
            elif self._is_detection_confirming(detection, setup_context):
                self.execution_context.confirming_detections.append(detection)
            else:
                self.execution_context.conflicting_detections.append(detection)

        # Update position information
        if current_state.current_position:
            self.execution_context.current_position_size = current_state.current_position.size
            self.execution_context.entry_price = current_state.current_position.entry_price
            self.execution_context.stop_loss = current_state.current_position.stop_loss
            self.execution_context.take_profit = current_state.current_position.take_profit
        else:
            self.execution_context.current_position_size = 0.0
            self.execution_context.entry_price = None
            self.execution_context.stop_loss = None
            self.execution_context.take_profit = None

    def _is_detection_confirming(
        self,
        detection: TimeframeDetection,
        setup_context: MultiTimeframeSetupContext
    ) -> bool:
        """Check if a detection confirms the primary setup."""
        if not setup_context.all_detections:
            return False

        primary_detections = [
            d for d in setup_context.all_detections
            if d.timeframe == self.execution_context.primary_timeframe
        ]

        if not primary_detections:
            return False

        primary_direction = primary_detections[0].direction
        return detection.direction == primary_direction

    def _analyze_timeframe_alignment(
        self,
        setup_context: MultiTimeframeSetupContext
    ) -> Dict[str, Any]:
        """Analyze alignment across timeframes."""
        alignment_analysis = {
            'alignment_score': 0.0,
            'confirming_timeframes': [],
            'conflicting_timeframes': [],
            'neutral_timeframes': [],
            'primary_direction': None,
            'strength_by_timeframe': {}
        }

        if not self.execution_context.active_detections:
            return alignment_analysis

        # Get primary direction
        primary_direction = self.execution_context.active_detections[0].direction
        alignment_analysis['primary_direction'] = primary_direction

        # Analyze each timeframe
        all_timeframes = set(d.timeframe for d in setup_context.all_detections)

        for timeframe in all_timeframes:
            tf_detections = [
                d for d in setup_context.all_detections
                if d.timeframe == timeframe
            ]

            if not tf_detections:
                alignment_analysis['neutral_timeframes'].append(timeframe)
                continue

            # Calculate timeframe strength
            strength = sum(d.strength for d in tf_detections) / len(tf_detections)
            alignment_analysis['strength_by_timeframe'][timeframe] = strength

            # Check alignment
            tf_direction = tf_detections[0].direction
            if tf_direction == primary_direction:
                alignment_analysis['confirming_timeframes'].append(timeframe)
            else:
                alignment_analysis['conflicting_timeframes'].append(timeframe)

        # Calculate overall alignment score
        total_timeframes = len(all_timeframes)
        confirming_count = len(alignment_analysis['confirming_timeframes'])

        if total_timeframes > 0:
            base_alignment = confirming_count / total_timeframes

            # Weight by strength
            confirming_strength = sum(
                alignment_analysis['strength_by_timeframe'].get(tf, 0.0)
                for tf in alignment_analysis['confirming_timeframes']
            )

            conflicting_strength = sum(
                alignment_analysis['strength_by_timeframe'].get(tf, 0.0)
                for tf in alignment_analysis['conflicting_timeframes']
            )

            if confirming_strength + conflicting_strength > 0:
                strength_ratio = confirming_strength / (confirming_strength + conflicting_strength)
                alignment_analysis['alignment_score'] = (base_alignment + strength_ratio) / 2
            else:
                alignment_analysis['alignment_score'] = base_alignment

        return alignment_analysis

    def _evaluate_trade_conditions(
        self,
        setup_context: MultiTimeframeSetupContext,
        timeframe_data: Dict[str, Any],
        alignment_analysis: Dict[str, Any]
    ) -> TradeDecision:
        """Evaluate if trade conditions are met."""
        # Check if we already have a position
        if self.execution_context.current_position_size != 0:
            return self._evaluate_exit_conditions(setup_context, timeframe_data)

        # Check entry conditions
        return self._evaluate_entry_conditions(setup_context, timeframe_data, alignment_analysis)

    def _evaluate_entry_conditions(
        self,
        setup_context: MultiTimeframeSetupContext,
        timeframe_data: Dict[str, Any],
        alignment_analysis: Dict[str, Any]
    ) -> TradeDecision:
        """Evaluate conditions for entering a new position."""
        # Check alignment requirements
        if self.execution_context.require_all_timeframes:
            if alignment_analysis['conflicting_timeframes']:
                return TradeDecision(
                    decision_type=TradeDecisionType.NO_ACTION,
                    timeframe=self.execution_context.primary_timeframe,
                    price=0.0,
                    execution_reason="Conflicting timeframes present"
                )

        # Check minimum confirmation requirement
        if len(self.execution_context.confirming_detections) < self.execution_context.min_confirmation_count:
            return TradeDecision(
                decision_type=TradeDecisionType.WAIT_FOR_CONFIRMATION,
                timeframe=self.execution_context.primary_timeframe,
                price=0.0,
                execution_reason="Insufficient confirmation"
            )

        # Check alignment score threshold
        if alignment_analysis['alignment_score'] < 0.6:
            return TradeDecision(
                decision_type=TradeDecisionType.NO_ACTION,
                timeframe=self.execution_context.primary_timeframe,
                price=0.0,
                execution_reason="Low alignment score"
            )

        # Determine entry direction and price
        if not self.execution_context.active_detections:
            return TradeDecision(
                decision_type=TradeDecisionType.NO_ACTION,
                timeframe=self.execution_context.primary_timeframe,
                price=0.0,
                execution_reason="No active detections"
            )

        primary_detection = self.execution_context.active_detections[0]
        direction = primary_detection.direction
        entry_price = primary_detection.price

        # Calculate position size
        position_size = self._calculate_position_size(primary_detection, timeframe_data)

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_take_levels(
            entry_price, direction, primary_detection
        )

        # Determine decision type
        if direction.value == 'BULLISH':
            decision_type = TradeDecisionType.ENTER_LONG
        elif direction.value == 'BEARISH':
            decision_type = TradeDecisionType.ENTER_SHORT
        else:
            return TradeDecision(
                decision_type=TradeDecisionType.NO_ACTION,
                timeframe=self.execution_context.primary_timeframe,
                price=entry_price,
                execution_reason="Neutral direction"
            )

        return TradeDecision(
            decision_type=decision_type,
            timeframe=self.execution_context.execution_timeframe,
            price=entry_price,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            primary_detection=primary_detection,
            supporting_detections=self.execution_context.confirming_detections,
            execution_reason="Entry conditions met"
        )

    def _evaluate_exit_conditions(
        self,
        setup_context: MultiTimeframeSetupContext,
        timeframe_data: Dict[str, Any]
    ) -> TradeDecision:
        """Evaluate conditions for exiting current position."""
        # Check for opposite signals
        if self.execution_context.active_detections:
            primary_detection = self.execution_context.active_detections[0]
            current_direction = primary_detection.direction

            # If we're long and get bearish signal, or vice versa
            position_is_long = self.execution_context.current_position_size > 0
            signal_is_bearish = current_direction.value == 'BEARISH'
            signal_is_bullish = current_direction.value == 'BULLISH'

            if (position_is_long and signal_is_bearish) or (not position_is_long and signal_is_bullish):
                return TradeDecision(
                    decision_type=TradeDecisionType.EXIT_POSITION,
                    timeframe=self.execution_context.execution_timeframe,
                    price=primary_detection.price,
                    primary_detection=primary_detection,
                    execution_reason="Opposite signal detected"
                )

        # Check for conflicting timeframes
        if len(self.execution_context.conflicting_detections) > len(self.execution_context.confirming_detections):
            avg_conflict_price = sum(d.price for d in self.execution_context.conflicting_detections) / len(self.execution_context.conflicting_detections)
            return TradeDecision(
                decision_type=TradeDecisionType.EXIT_POSITION,
                timeframe=self.execution_context.execution_timeframe,
                price=avg_conflict_price,
                supporting_detections=self.execution_context.conflicting_detections,
                execution_reason="Conflicting timeframes dominant"
            )

        # No exit conditions met
        return TradeDecision(
            decision_type=TradeDecisionType.NO_ACTION,
            timeframe=self.execution_context.execution_timeframe,
            price=0.0,
            execution_reason="No exit conditions"
        )

    def _calculate_position_size(
        self,
        primary_detection: TimeframeDetection,
        timeframe_data: Dict[str, Any]
    ) -> float:
        """Calculate appropriate position size based on risk parameters."""
        # Base position size on detection strength
        base_size = min(self.execution_context.max_position_size,
                       primary_detection.strength * self.execution_context.max_position_size)

        # Adjust for risk percentage
        risk_adjusted_size = base_size * (self.execution_context.risk_percentage / 100.0)

        # Ensure minimum viable size
        return max(0.01, risk_adjusted_size)

    def _calculate_stop_take_levels(
        self,
        entry_price: float,
        direction,
        primary_detection: TimeframeDetection
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if direction.value == 'BULLISH':
            # For long positions
            stop_loss = entry_price - (entry_price * self.execution_context.stop_loss_buffer)
            take_profit = entry_price + ((entry_price - stop_loss) * self.execution_context.take_profit_ratio)
        else:
            # For short positions
            stop_loss = entry_price + (entry_price * self.execution_context.stop_loss_buffer)
            take_profit = entry_price - ((stop_loss - entry_price) * self.execution_context.take_profit_ratio)

        return stop_loss, take_profit

    def _apply_risk_management(
        self,
        trade_decision: TradeDecision,
        current_state: MultiTimeframeBacktestState
    ) -> TradeDecision:
        """Apply risk management filters to trade decision."""
        # Check daily trade limit
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date

        if self.daily_trade_count >= self.max_daily_trades:
            trade_decision.decision_type = TradeDecisionType.NO_ACTION
            trade_decision.execution_reason = "Daily trade limit reached"
            return trade_decision

        # Check concurrent trade limit
        active_positions = sum(1 for pos in current_state.position_history if pos.exit_price is None)
        if active_positions >= self.max_concurrent_trades:
            if trade_decision.decision_type in [TradeDecisionType.ENTER_LONG, TradeDecisionType.ENTER_SHORT]:
                trade_decision.decision_type = TradeDecisionType.NO_ACTION
                trade_decision.execution_reason = "Concurrent trade limit reached"

        # Validate position size
        if trade_decision.size and trade_decision.size > self.execution_context.max_position_size:
            trade_decision.size = self.execution_context.max_position_size
            trade_decision.execution_reason += " (size capped)"

        return trade_decision

    def _calculate_confidence_score(
        self,
        trade_decision: TradeDecision,
        alignment_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for trade decision."""
        if trade_decision.decision_type == TradeDecisionType.NO_ACTION:
            return 0.0

        # Base confidence on alignment
        base_confidence = alignment_analysis['alignment_score']

        # Adjust for detection strength
        if trade_decision.primary_detection:
            strength_factor = trade_decision.primary_detection.strength
            base_confidence = (base_confidence + strength_factor) / 2

        # Adjust for supporting detections
        if trade_decision.supporting_detections:
            support_strength = sum(d.strength for d in trade_decision.supporting_detections) / len(trade_decision.supporting_detections)
            base_confidence = (base_confidence + support_strength) / 2

        return min(1.0, base_confidence)

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        return {
            'execution_stats': self.execution_stats.copy(),
            'execution_context': {
                'primary_timeframe': self.execution_context.primary_timeframe,
                'execution_timeframe': self.execution_context.execution_timeframe,
                'active_detections_count': len(self.execution_context.active_detections),
                'confirming_detections_count': len(self.execution_context.confirming_detections),
                'conflicting_detections_count': len(self.execution_context.conflicting_detections),
                'current_position_size': self.execution_context.current_position_size
            },
            'risk_settings': {
                'risk_percentage': self.execution_context.risk_percentage,
                'max_position_size': self.execution_context.max_position_size,
                'max_concurrent_trades': self.max_concurrent_trades,
                'max_daily_trades': self.max_daily_trades,
                'daily_trade_count': self.daily_trade_count
            }
        }

    def update_execution_settings(self, **settings):
        """Update execution settings dynamically."""
        for key, value in settings.items():
            if hasattr(self.execution_context, key):
                setattr(self.execution_context, key, value)
                self.logger.info(f"Updated execution setting: {key} = {value}")
            else:
                self.logger.warning(f"Unknown execution setting: {key}")


# Utility functions
def create_trade_executor(config: MultiTimeframeStrategyConfig) -> MultiTimeframeTradeExecutor:
    """
    Factory function to create a trade executor.

    Args:
        config: Multi-timeframe strategy configuration

    Returns:
        Configured MultiTimeframeTradeExecutor instance
    """
    return MultiTimeframeTradeExecutor(config)


def analyze_trade_decision_quality(
    decisions: List[TradeDecision],
    actual_outcomes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Analyze the quality of trade decisions.

    Args:
        decisions: List of trade decisions to analyze
        actual_outcomes: Optional list of actual trade outcomes

    Returns:
        Analysis of decision quality
    """
    analysis = {
        'total_decisions': len(decisions),
        'execution_decisions': 0,
        'no_action_decisions': 0,
        'wait_decisions': 0,
        'avg_confidence': 0.0,
        'confidence_distribution': {},
        'decision_types': {}
    }

    if not decisions:
        return analysis

    # Analyze decision types
    for decision in decisions:
        decision_type = decision.decision_type.value
        analysis['decision_types'][decision_type] = analysis['decision_types'].get(decision_type, 0) + 1

        if decision.decision_type in [TradeDecisionType.ENTER_LONG, TradeDecisionType.ENTER_SHORT]:
            analysis['execution_decisions'] += 1
        elif decision.decision_type == TradeDecisionType.NO_ACTION:
            analysis['no_action_decisions'] += 1
        elif decision.decision_type == TradeDecisionType.WAIT_FOR_CONFIRMATION:
            analysis['wait_decisions'] += 1

    # Calculate average confidence
    confidences = [d.confidence_score for d in decisions if d.confidence_score > 0]
    if confidences:
        analysis['avg_confidence'] = sum(confidences) / len(confidences)

        # Confidence distribution
        for conf in confidences:
            bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
            analysis['confidence_distribution'][bucket] = analysis['confidence_distribution'].get(bucket, 0) + 1

    return analysis