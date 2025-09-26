"""
Multi-timeframe state types for cross-timeframe strategy execution.

This module extends the existing state system to handle multi-timeframe
trading strategies where indicators can execute on different timeframes
and require coordination across those timeframes.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
import logging

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
from .json_converter import MultiTimeframeStrategyConfig


logger = logging.getLogger(__name__)


class MultiTimeframeExecutionState(Enum):
    """
    Execution states for multi-timeframe strategies.

    Extends the basic ExecutionState with multi-timeframe coordination.
    """
    SCANNING = "scanning"                        # Looking for first signal on any timeframe

    # Multi-timeframe signal states
    SIGNAL_1_TIMEFRAME_SYNC = "signal_1_tf_sync"  # First signal detected, syncing timeframes
    SIGNAL_2_TIMEFRAME_SYNC = "signal_2_tf_sync"  # Second signal detected, syncing timeframes
    SIGNAL_3_TIMEFRAME_SYNC = "signal_3_tf_sync"  # Third signal detected, syncing timeframes
    SIGNAL_4_TIMEFRAME_SYNC = "signal_4_tf_sync"  # Fourth signal detected, syncing timeframes
    SIGNAL_5_TIMEFRAME_SYNC = "signal_5_tf_sync"  # Fifth signal detected, syncing timeframes

    # Cross-timeframe confirmation states
    TIMEFRAME_ALIGNMENT_CHECK = "timeframe_alignment_check"  # Checking alignment across timeframes
    CROSS_TIMEFRAME_CONFIRMATION = "cross_tf_confirmation"   # Waiting for confirmations across TFs

    # Trading states
    READY_TO_ENTER = "ready_to_enter"           # All timeframes aligned and confirmed
    IN_POSITION = "in_position"                 # Trade is active
    POSITION_CLOSED = "position_closed"         # Trade closed


class TimeframeSyncStatus(Enum):
    """Status of timeframe synchronization."""
    SYNCED = "synced"           # All timeframes are aligned
    SYNCING = "syncing"         # In process of synchronization
    DIVERGED = "diverged"       # Timeframes have conflicting signals
    TIMEOUT = "timeout"         # Sync timeout occurred


@dataclass
class TimeframeDetection(Detection):
    """
    Enhanced detection with timeframe information.

    Extends the base Detection class to include timeframe-specific data.
    """
    # Override parent fields with defaults
    metadata: Dict[str, Any] = field(default_factory=dict)

    # New timeframe-specific fields
    timeframe: str = ""                         # Timeframe this detection occurred on
    timeframe_candle_index: int = 0             # Index within the specific timeframe
    equivalent_primary_index: Optional[int] = None  # Equivalent index on primary timeframe
    confidence_score: float = 1.0              # Confidence in this detection (0-1)

    # Cross-timeframe correlation data
    correlated_detections: Dict[str, int] = field(default_factory=dict)  # timeframe -> candle_index
    sync_quality: float = 1.0                  # Quality of sync with other timeframes (0-1)

    def __str__(self):
        return f"{self.indicator_name}@{self.timeframe} at {self.price:.5f} ({self.direction.value})"


@dataclass
class TimeframeState:
    """
    State information for a specific timeframe.

    Tracks the state and progress of indicators on individual timeframes.
    """
    timeframe: str
    current_candle_index: int = 0
    last_processed_index: int = -1

    # Indicators active on this timeframe
    active_indicators: List[str] = field(default_factory=list)
    completed_indicators: List[str] = field(default_factory=list)

    # Detection history for this timeframe
    detections: List[TimeframeDetection] = field(default_factory=list)

    # Synchronization status with other timeframes
    sync_status: TimeframeSyncStatus = TimeframeSyncStatus.SYNCED
    sync_target_index: Optional[int] = None     # Target index for synchronization
    sync_attempts: int = 0

    # Performance tracking
    processing_lag: float = 0.0                 # Lag behind primary timeframe
    detection_latency: float = 0.0              # Average time to detect signals

    def add_detection(self, detection: TimeframeDetection):
        """Add a detection to this timeframe."""
        detection.timeframe = self.timeframe
        detection.timeframe_candle_index = self.current_candle_index
        self.detections.append(detection)

        # Update completed indicators
        if detection.indicator_name not in self.completed_indicators:
            self.completed_indicators.append(detection.indicator_name)

    def get_latest_detection(self, indicator_name: Optional[str] = None) -> Optional[TimeframeDetection]:
        """Get the most recent detection, optionally filtered by indicator."""
        filtered_detections = self.detections
        if indicator_name:
            filtered_detections = [d for d in self.detections if d.indicator_name == indicator_name]

        return filtered_detections[-1] if filtered_detections else None

    def is_synced_with_primary(self, primary_index: int, tolerance: int = 2) -> bool:
        """Check if this timeframe is synchronized with primary timeframe."""
        if self.sync_target_index is None:
            return True

        return abs(self.current_candle_index - self.sync_target_index) <= tolerance


@dataclass
class MultiTimeframeSetupContext(SetupContext):
    """
    Enhanced setup context for multi-timeframe strategies.

    Extends SetupContext to manage detections across multiple timeframes.
    """
    # Timeframe-specific data
    timeframe_states: Dict[str, TimeframeState] = field(default_factory=dict)
    primary_timeframe: str = "H1"

    # Cross-timeframe synchronization
    sync_status: TimeframeSyncStatus = TimeframeSyncStatus.SYNCED
    sync_quality_score: float = 1.0             # Overall sync quality (0-1)

    # Multi-timeframe detection tracking
    detections_by_timeframe: Dict[str, List[TimeframeDetection]] = field(default_factory=dict)
    required_timeframes: Set[str] = field(default_factory=set)
    confirmed_timeframes: Set[str] = field(default_factory=set)

    # Timing for cross-timeframe coordination
    timeframe_sync_start: Optional[datetime] = None
    last_sync_attempt: Optional[datetime] = None

    def initialize_timeframes(self, timeframes: List[str], primary_timeframe: str):
        """Initialize tracking for all required timeframes."""
        self.primary_timeframe = primary_timeframe
        self.required_timeframes = set(timeframes)

        for timeframe in timeframes:
            self.timeframe_states[timeframe] = TimeframeState(timeframe=timeframe)
            self.detections_by_timeframe[timeframe] = []

    def add_timeframe_detection(self, detection: TimeframeDetection):
        """Add a detection with timeframe coordination."""
        timeframe = detection.timeframe

        # Add to base detection list
        super().add_detection(detection)

        # Add to timeframe-specific tracking
        if timeframe not in self.detections_by_timeframe:
            self.detections_by_timeframe[timeframe] = []

        self.detections_by_timeframe[timeframe].append(detection)

        # Update timeframe state
        if timeframe in self.timeframe_states:
            self.timeframe_states[timeframe].add_detection(detection)

        # Update synchronization status
        self._update_sync_status()

    def _update_sync_status(self):
        """Update cross-timeframe synchronization status."""
        if len(self.timeframe_states) <= 1:
            self.sync_status = TimeframeSyncStatus.SYNCED
            self.sync_quality_score = 1.0
            return

        # Check alignment across timeframes
        primary_state = self.timeframe_states.get(self.primary_timeframe)
        if not primary_state:
            self.sync_status = TimeframeSyncStatus.DIVERGED
            return

        # Calculate sync quality based on timeframe alignment
        sync_scores = []
        for tf_name, tf_state in self.timeframe_states.items():
            if tf_name == self.primary_timeframe:
                continue

            if tf_state.is_synced_with_primary(primary_state.current_candle_index):
                sync_scores.append(1.0)
            else:
                # Calculate partial sync score based on deviation
                deviation = abs(tf_state.current_candle_index - primary_state.current_candle_index)
                sync_score = max(0.0, 1.0 - (deviation / 10.0))  # Penalize deviation
                sync_scores.append(sync_score)

        self.sync_quality_score = sum(sync_scores) / len(sync_scores) if sync_scores else 1.0

        # Update status based on quality
        if self.sync_quality_score >= 0.8:
            self.sync_status = TimeframeSyncStatus.SYNCED
        elif self.sync_quality_score >= 0.5:
            self.sync_status = TimeframeSyncStatus.SYNCING
        else:
            self.sync_status = TimeframeSyncStatus.DIVERGED

    def get_timeframe_progress(self) -> Dict[str, float]:
        """Get completion progress for each timeframe (0.0 to 1.0)."""
        progress = {}

        for tf_name, tf_state in self.timeframe_states.items():
            total_indicators = len(tf_state.active_indicators)
            completed_indicators = len(tf_state.completed_indicators)

            if total_indicators == 0:
                progress[tf_name] = 1.0
            else:
                progress[tf_name] = completed_indicators / total_indicators

        return progress

    def is_timeframe_ready(self, timeframe: str) -> bool:
        """Check if a specific timeframe has completed its requirements."""
        return timeframe in self.confirmed_timeframes

    def get_cross_timeframe_correlation(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlation between detections on different timeframes."""
        correlations = {}

        timeframes = list(self.timeframe_states.keys())
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i+1:]:
                correlation = self._calculate_timeframe_correlation(tf1, tf2)
                correlations[(tf1, tf2)] = correlation

        return correlations

    def _calculate_timeframe_correlation(self, tf1: str, tf2: str) -> float:
        """Calculate correlation between two timeframes."""
        detections1 = self.detections_by_timeframe.get(tf1, [])
        detections2 = self.detections_by_timeframe.get(tf2, [])

        if not detections1 or not detections2:
            return 0.0

        # Simple correlation based on detection timing alignment
        # More sophisticated correlation algorithms can be added here

        correlation_sum = 0.0
        comparison_count = 0

        for d1 in detections1:
            for d2 in detections2:
                # Check if detections are for similar indicators or directions
                if d1.direction == d2.direction:
                    # Calculate time-based correlation
                    time_diff = abs((d1.timestamp - d2.timestamp).total_seconds())
                    time_correlation = max(0.0, 1.0 - (time_diff / 3600.0))  # 1 hour window
                    correlation_sum += time_correlation
                    comparison_count += 1

        return correlation_sum / comparison_count if comparison_count > 0 else 0.0

    def reset(self):
        """Reset all multi-timeframe state."""
        super().reset()

        # Reset timeframe states
        for tf_state in self.timeframe_states.values():
            tf_state.detections.clear()
            tf_state.completed_indicators.clear()
            tf_state.sync_status = TimeframeSyncStatus.SYNCED
            tf_state.sync_attempts = 0

        # Reset multi-timeframe tracking
        self.detections_by_timeframe.clear()
        self.confirmed_timeframes.clear()
        self.sync_status = TimeframeSyncStatus.SYNCED
        self.sync_quality_score = 1.0
        self.timeframe_sync_start = None
        self.last_sync_attempt = None


@dataclass
class MultiTimeframeStateTransition(StateTransition):
    """
    Enhanced state transition with multi-timeframe information.
    """
    timeframe: Optional[str] = None             # Timeframe that triggered transition
    sync_status: TimeframeSyncStatus = TimeframeSyncStatus.SYNCED
    affected_timeframes: List[str] = field(default_factory=list)
    correlation_score: float = 1.0             # Cross-timeframe correlation at transition


@dataclass
class MultiTimeframeBacktestState(BacktestState):
    """
    Enhanced backtest state for multi-timeframe strategies.

    Extends BacktestState to manage execution across multiple timeframes.
    """
    # Multi-timeframe configuration
    multi_tf_config: Optional[MultiTimeframeStrategyConfig] = None

    # Current state (extends base state)
    current_tf_state: MultiTimeframeExecutionState = MultiTimeframeExecutionState.SCANNING

    # Timeframe coordination
    timeframe_states: Dict[str, TimeframeState] = field(default_factory=dict)
    primary_timeframe: str = "H1"
    active_timeframes: List[str] = field(default_factory=list)

    # Enhanced setup context
    multi_tf_setup: Optional[MultiTimeframeSetupContext] = None

    # Cross-timeframe synchronization
    sync_coordinator: Optional['TimeframeSyncCoordinator'] = None

    # Multi-timeframe statistics
    timeframe_sync_successes: int = 0
    timeframe_sync_failures: int = 0
    cross_tf_confirmations: int = 0

    # Performance metrics per timeframe
    timeframe_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def initialize_timeframes(self, config: MultiTimeframeStrategyConfig):
        """Initialize multi-timeframe state from configuration."""
        self.multi_tf_config = config
        self.primary_timeframe = config.primary_timeframe
        self.active_timeframes = config.get_all_timeframes()

        # Initialize timeframe states
        for timeframe in self.active_timeframes:
            self.timeframe_states[timeframe] = TimeframeState(timeframe=timeframe)

            # Set active indicators for this timeframe
            indicators = config.get_indicators_for_timeframe(timeframe)
            self.timeframe_states[timeframe].active_indicators = indicators

            # Initialize performance tracking
            self.timeframe_performance[timeframe] = {
                'detections': 0,
                'sync_successes': 0,
                'sync_failures': 0,
                'avg_latency': 0.0
            }

    def can_start_new_setup(self) -> bool:
        """Check if we can start a new multi-timeframe setup."""
        base_ready = super().can_start_new_setup()
        if not base_ready:
            return False

        # Additional multi-timeframe checks
        return (
            self.current_tf_state == MultiTimeframeExecutionState.SCANNING and
            self.multi_tf_setup is None and
            all(tf_state.sync_status != TimeframeSyncStatus.TIMEOUT
                for tf_state in self.timeframe_states.values())
        )

    def get_next_multi_tf_state(self) -> MultiTimeframeExecutionState:
        """Get next multi-timeframe state based on current progress."""
        if not self.multi_tf_setup:
            return MultiTimeframeExecutionState.SCANNING

        # Determine progression based on detections and sync status
        detection_count = len(self.multi_tf_setup.detections)
        sync_status = self.multi_tf_setup.sync_status

        state_progression = [
            MultiTimeframeExecutionState.SCANNING,
            MultiTimeframeExecutionState.SIGNAL_1_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_2_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_3_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_4_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.SIGNAL_5_TIMEFRAME_SYNC,
            MultiTimeframeExecutionState.TIMEFRAME_ALIGNMENT_CHECK,
            MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION,
            MultiTimeframeExecutionState.READY_TO_ENTER
        ]

        # Handle sync-related states
        if sync_status == TimeframeSyncStatus.SYNCING:
            return MultiTimeframeExecutionState.TIMEFRAME_ALIGNMENT_CHECK
        elif sync_status == TimeframeSyncStatus.DIVERGED:
            return MultiTimeframeExecutionState.CROSS_TIMEFRAME_CONFIRMATION

        # Normal progression based on detections
        try:
            current_idx = state_progression.index(self.current_tf_state)
            target_idx = min(detection_count + 1, len(state_progression) - 1)

            if target_idx > current_idx:
                return state_progression[target_idx]
        except ValueError:
            pass

        return self.current_tf_state

    def record_multi_tf_transition(self, from_state: MultiTimeframeExecutionState,
                                 to_state: MultiTimeframeExecutionState,
                                 timestamp: datetime, candle_index: int,
                                 trigger: str, timeframe: Optional[str] = None):
        """Record a multi-timeframe state transition."""
        # Create enhanced transition record
        transition = MultiTimeframeStateTransition(
            from_state=ExecutionState.SCANNING,  # Map to base enum for compatibility
            to_state=ExecutionState.SCANNING,    # Will be overridden in practice
            timestamp=timestamp,
            candle_index=candle_index,
            trigger=trigger,
            timeframe=timeframe,
            sync_status=self.multi_tf_setup.sync_status if self.multi_tf_setup else TimeframeSyncStatus.SYNCED,
            affected_timeframes=list(self.active_timeframes)
        )

        self.state_history.append(transition)
        self.current_tf_state = to_state

    def update_timeframe_performance(self, timeframe: str, metric: str, value: Any):
        """Update performance metrics for a specific timeframe."""
        if timeframe not in self.timeframe_performance:
            self.timeframe_performance[timeframe] = {}

        self.timeframe_performance[timeframe][metric] = value

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics across timeframes."""
        total_attempts = self.timeframe_sync_successes + self.timeframe_sync_failures
        sync_rate = (self.timeframe_sync_successes / total_attempts) if total_attempts > 0 else 0.0

        return {
            'sync_success_rate': sync_rate,
            'total_sync_attempts': total_attempts,
            'successful_syncs': self.timeframe_sync_successes,
            'failed_syncs': self.timeframe_sync_failures,
            'cross_tf_confirmations': self.cross_tf_confirmations,
            'active_timeframes_count': len(self.active_timeframes),
            'primary_timeframe': self.primary_timeframe
        }

    def check_multi_tf_timeout(self, current_candle: int) -> bool:
        """Check for timeout across multiple timeframes."""
        # Check base timeout
        base_timeout = super().check_timeout(current_candle)
        if base_timeout:
            return True

        # Check timeframe-specific timeouts
        if not self.multi_tf_setup:
            return False

        # Check if any timeframe has timed out on synchronization
        for tf_name, tf_state in self.timeframe_states.items():
            if tf_state.sync_status == TimeframeSyncStatus.TIMEOUT:
                logger.warning(f"Timeframe {tf_name} sync timeout detected")
                return True

        # Check cross-timeframe sync timeout
        if (self.multi_tf_setup.timeframe_sync_start and
            self.multi_tf_setup.last_sync_attempt):

            sync_duration = (datetime.now() - self.multi_tf_setup.timeframe_sync_start).total_seconds()
            if sync_duration > 300:  # 5 minute sync timeout
                return True

        return False


# Utility functions for multi-timeframe state management
def create_multi_tf_setup(setup_id: int, timestamp: datetime, candle_index: int,
                         config: MultiTimeframeStrategyConfig) -> MultiTimeframeSetupContext:
    """Create a new multi-timeframe setup context."""
    setup = MultiTimeframeSetupContext(
        setup_id=setup_id,
        start_timestamp=timestamp,
        start_candle_index=candle_index
    )

    # Initialize timeframes
    timeframes = config.get_all_timeframes()
    setup.initialize_timeframes(timeframes, config.primary_timeframe)

    return setup


def convert_legacy_detection_to_timeframe(detection: Detection, timeframe: str) -> TimeframeDetection:
    """Convert a legacy Detection to TimeframeDetection."""
    return TimeframeDetection(
        indicator_name=detection.indicator_name,
        timestamp=detection.timestamp,
        candle_index=detection.candle_index,
        price=detection.price,
        direction=detection.direction,
        metadata=detection.metadata,
        timeframe=timeframe,
        timeframe_candle_index=detection.candle_index,
        confidence_score=1.0
    )