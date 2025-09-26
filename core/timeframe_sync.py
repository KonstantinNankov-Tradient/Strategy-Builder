"""
Cross-timeframe state synchronization coordinator.

This module handles the coordination and synchronization of states and
detections across multiple timeframes, ensuring proper alignment and
timing for multi-timeframe trading strategies.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from .multi_timeframe_state import (
    TimeframeState,
    TimeframeDetection,
    MultiTimeframeSetupContext,
    TimeframeSyncStatus,
    MultiTimeframeExecutionState
)
from .data_loader import TimeframeConverter
from .json_converter import MultiTimeframeStrategyConfig


logger = logging.getLogger(__name__)


class SyncPolicy(Enum):
    """Policies for cross-timeframe synchronization."""
    STRICT = "strict"           # All timeframes must be perfectly aligned
    MODERATE = "moderate"       # Allow reasonable timing differences
    RELAXED = "relaxed"         # Accept larger timing differences
    ADAPTIVE = "adaptive"       # Adjust policy based on market conditions


class SyncMode(Enum):
    """Synchronization execution modes."""
    REAL_TIME = "real_time"     # Sync as data comes in
    BATCH = "batch"             # Sync at regular intervals
    ON_DEMAND = "on_demand"     # Sync only when requested


@dataclass
class SyncConfiguration:
    """Configuration for timeframe synchronization."""
    policy: SyncPolicy = SyncPolicy.MODERATE
    mode: SyncMode = SyncMode.REAL_TIME

    # Timing tolerances (in seconds)
    max_time_drift: float = 300.0           # 5 minutes max drift
    sync_window: float = 60.0               # 1 minute sync window
    timeout_threshold: float = 900.0        # 15 minute timeout

    # Alignment tolerances
    max_candle_offset: int = 3              # Max candles between timeframes
    min_correlation_score: float = 0.6      # Min required correlation

    # Performance settings
    max_sync_attempts: int = 5
    sync_retry_delay: float = 10.0          # 10 seconds between retries
    enable_predictive_sync: bool = True     # Predict future alignment


@dataclass
class SyncAttempt:
    """Records a synchronization attempt."""
    timestamp: datetime
    attempt_number: int
    source_timeframe: str
    target_timeframes: List[str]
    success: bool
    error_message: Optional[str] = None
    sync_quality: float = 0.0
    duration: float = 0.0                   # Sync duration in seconds


class TimeframeSyncCoordinator:
    """
    Coordinates synchronization across multiple timeframes.

    Manages the timing and alignment of detections and state changes
    across different timeframes to ensure coherent multi-timeframe
    strategy execution.
    """

    def __init__(self, config: SyncConfiguration = None):
        """
        Initialize the sync coordinator.

        Args:
            config: Synchronization configuration
        """
        self.config = config or SyncConfiguration()
        self.timeframe_converter = TimeframeConverter()

        # Synchronization state
        self.sync_attempts: List[SyncAttempt] = []
        self.last_sync_time: Optional[datetime] = None
        self.sync_in_progress: bool = False

        # Timeframe relationships
        self.timeframe_hierarchy: Dict[str, int] = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }

        # Performance tracking
        self.sync_performance: Dict[str, Any] = {
            'total_attempts': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'avg_sync_time': 0.0,
            'best_sync_quality': 0.0
        }

    def synchronize_timeframes(self, setup: MultiTimeframeSetupContext,
                             trigger_timeframe: str,
                             detection: TimeframeDetection) -> bool:
        """
        Synchronize all timeframes based on a new detection.

        Args:
            setup: Multi-timeframe setup context
            trigger_timeframe: Timeframe that triggered the sync
            detection: Detection that triggered the sync

        Returns:
            True if synchronization successful, False otherwise
        """
        logger.debug(f"Starting timeframe sync triggered by {trigger_timeframe}")

        sync_start = datetime.now()
        attempt = SyncAttempt(
            timestamp=sync_start,
            attempt_number=len(self.sync_attempts) + 1,
            source_timeframe=trigger_timeframe,
            target_timeframes=list(setup.timeframe_states.keys()),
            success=False
        )

        try:
            self.sync_in_progress = True

            # Update equivalent indices across timeframes
            self._update_equivalent_indices(setup, trigger_timeframe, detection)

            # Perform synchronization based on policy
            sync_success = self._execute_sync_policy(setup, trigger_timeframe, detection)

            # Update sync status
            if sync_success:
                setup.sync_status = TimeframeSyncStatus.SYNCED
                self._update_sync_quality(setup)
            else:
                setup.sync_status = TimeframeSyncStatus.DIVERGED

            attempt.success = sync_success
            attempt.sync_quality = setup.sync_quality_score
            attempt.duration = (datetime.now() - sync_start).total_seconds()

            # Update performance metrics
            self._update_performance_metrics(attempt)

            logger.info(f"Timeframe sync completed: success={sync_success}, "
                       f"quality={setup.sync_quality_score:.3f}")

            return sync_success

        except Exception as e:
            attempt.error_message = str(e)
            logger.error(f"Timeframe sync failed: {e}")
            return False

        finally:
            self.sync_attempts.append(attempt)
            self.sync_in_progress = False
            self.last_sync_time = datetime.now()

    def _update_equivalent_indices(self, setup: MultiTimeframeSetupContext,
                                 source_timeframe: str,
                                 detection: TimeframeDetection):
        """Update equivalent candle indices across all timeframes."""
        source_index = detection.timeframe_candle_index

        # Calculate equivalent indices for all other timeframes
        for target_timeframe in setup.timeframe_states.keys():
            if target_timeframe == source_timeframe:
                continue

            try:
                # Convert index from source to target timeframe
                equivalent_index = self.timeframe_converter.convert_timeframe_index(
                    source_index, source_timeframe, target_timeframe
                )

                # Update the timeframe state
                target_state = setup.timeframe_states[target_timeframe]
                target_state.sync_target_index = equivalent_index

                # Update detection correlation
                detection.correlated_detections[target_timeframe] = equivalent_index

                logger.debug(f"Mapped {source_timeframe}[{source_index}] -> "
                           f"{target_timeframe}[{equivalent_index}]")

            except Exception as e:
                logger.warning(f"Failed to map indices {source_timeframe}->{target_timeframe}: {e}")

    def _execute_sync_policy(self, setup: MultiTimeframeSetupContext,
                           source_timeframe: str,
                           detection: TimeframeDetection) -> bool:
        """Execute synchronization based on the configured policy."""
        if self.config.policy == SyncPolicy.STRICT:
            return self._strict_sync(setup, source_timeframe, detection)
        elif self.config.policy == SyncPolicy.MODERATE:
            return self._moderate_sync(setup, source_timeframe, detection)
        elif self.config.policy == SyncPolicy.RELAXED:
            return self._relaxed_sync(setup, source_timeframe, detection)
        elif self.config.policy == SyncPolicy.ADAPTIVE:
            return self._adaptive_sync(setup, source_timeframe, detection)
        else:
            return self._moderate_sync(setup, source_timeframe, detection)

    def _strict_sync(self, setup: MultiTimeframeSetupContext,
                   source_timeframe: str,
                   detection: TimeframeDetection) -> bool:
        """Strict synchronization - all timeframes must align perfectly."""
        tolerance = 1  # 1 candle tolerance

        for tf_name, tf_state in setup.timeframe_states.items():
            if tf_name == source_timeframe:
                continue

            if not tf_state.is_synced_with_primary(detection.timeframe_candle_index, tolerance):
                logger.debug(f"Strict sync failed: {tf_name} not aligned")
                return False

        return True

    def _moderate_sync(self, setup: MultiTimeframeSetupContext,
                     source_timeframe: str,
                     detection: TimeframeDetection) -> bool:
        """Moderate synchronization - allow reasonable timing differences."""
        tolerance = self.config.max_candle_offset

        aligned_count = 0
        total_timeframes = len(setup.timeframe_states) - 1  # Exclude source

        for tf_name, tf_state in setup.timeframe_states.items():
            if tf_name == source_timeframe:
                continue

            if tf_state.is_synced_with_primary(detection.timeframe_candle_index, tolerance):
                aligned_count += 1

        # Require at least 70% alignment
        alignment_ratio = aligned_count / total_timeframes if total_timeframes > 0 else 1.0
        return alignment_ratio >= 0.7

    def _relaxed_sync(self, setup: MultiTimeframeSetupContext,
                    source_timeframe: str,
                    detection: TimeframeDetection) -> bool:
        """Relaxed synchronization - accept larger timing differences."""
        tolerance = self.config.max_candle_offset * 2

        aligned_count = 0
        total_timeframes = len(setup.timeframe_states) - 1

        for tf_name, tf_state in setup.timeframe_states.items():
            if tf_name == source_timeframe:
                continue

            if tf_state.is_synced_with_primary(detection.timeframe_candle_index, tolerance):
                aligned_count += 1

        # Require at least 50% alignment
        alignment_ratio = aligned_count / total_timeframes if total_timeframes > 0 else 1.0
        return alignment_ratio >= 0.5

    def _adaptive_sync(self, setup: MultiTimeframeSetupContext,
                     source_timeframe: str,
                     detection: TimeframeDetection) -> bool:
        """Adaptive synchronization - adjust based on conditions."""
        # Analyze recent sync performance to adapt policy
        recent_attempts = self.sync_attempts[-10:] if len(self.sync_attempts) >= 10 else self.sync_attempts

        if recent_attempts:
            success_rate = sum(1 for attempt in recent_attempts if attempt.success) / len(recent_attempts)
            avg_quality = sum(attempt.sync_quality for attempt in recent_attempts) / len(recent_attempts)

            # Adjust tolerance based on recent performance
            if success_rate < 0.5:
                # Poor recent performance - use relaxed policy
                return self._relaxed_sync(setup, source_timeframe, detection)
            elif avg_quality > 0.8:
                # High quality - use strict policy
                return self._strict_sync(setup, source_timeframe, detection)
            else:
                # Normal performance - use moderate policy
                return self._moderate_sync(setup, source_timeframe, detection)
        else:
            # No history - default to moderate
            return self._moderate_sync(setup, source_timeframe, detection)

    def _update_sync_quality(self, setup: MultiTimeframeSetupContext):
        """Update the overall synchronization quality score."""
        if len(setup.timeframe_states) <= 1:
            setup.sync_quality_score = 1.0
            return

        quality_scores = []

        # Calculate alignment quality
        primary_tf = setup.primary_timeframe
        primary_state = setup.timeframe_states.get(primary_tf)

        if primary_state:
            for tf_name, tf_state in setup.timeframe_states.items():
                if tf_name == primary_tf:
                    continue

                # Calculate temporal alignment
                if tf_state.sync_target_index and primary_state.current_candle_index:
                    deviation = abs(tf_state.current_candle_index - tf_state.sync_target_index)
                    alignment_score = max(0.0, 1.0 - (deviation / 10.0))
                    quality_scores.append(alignment_score)

        # Calculate detection correlation quality
        correlations = setup.get_cross_timeframe_correlation()
        if correlations:
            avg_correlation = sum(correlations.values()) / len(correlations)
            quality_scores.append(avg_correlation)

        # Update overall quality
        if quality_scores:
            setup.sync_quality_score = sum(quality_scores) / len(quality_scores)
        else:
            setup.sync_quality_score = 0.5  # Neutral score when no data

    def check_sync_timeout(self, setup: MultiTimeframeSetupContext) -> bool:
        """Check if synchronization has timed out."""
        if not setup.timeframe_sync_start:
            return False

        elapsed = (datetime.now() - setup.timeframe_sync_start).total_seconds()
        return elapsed > self.config.timeout_threshold

    def force_resync(self, setup: MultiTimeframeSetupContext) -> bool:
        """Force a resynchronization of all timeframes."""
        logger.info("Forcing timeframe resynchronization")

        # Reset sync status
        setup.sync_status = TimeframeSyncStatus.SYNCING
        setup.timeframe_sync_start = datetime.now()

        # Reset all timeframe sync states
        for tf_state in setup.timeframe_states.values():
            tf_state.sync_status = TimeframeSyncStatus.SYNCING
            tf_state.sync_attempts += 1

        # Find the most recent detection to use as sync anchor
        latest_detection = None
        latest_timeframe = None

        for tf_name, detections in setup.detections_by_timeframe.items():
            if detections:
                tf_latest = max(detections, key=lambda d: d.timestamp)
                if not latest_detection or tf_latest.timestamp > latest_detection.timestamp:
                    latest_detection = tf_latest
                    latest_timeframe = tf_name

        if latest_detection and latest_timeframe:
            return self.synchronize_timeframes(setup, latest_timeframe, latest_detection)

        return False

    def predict_next_sync_point(self, setup: MultiTimeframeSetupContext,
                              timeframe: str) -> Optional[int]:
        """Predict the next synchronization point for a timeframe."""
        if not self.config.enable_predictive_sync:
            return None

        tf_state = setup.timeframe_states.get(timeframe)
        if not tf_state or not tf_state.detections:
            return None

        # Analyze detection patterns
        recent_detections = tf_state.detections[-5:]  # Last 5 detections
        if len(recent_detections) < 2:
            return None

        # Calculate average interval between detections
        intervals = []
        for i in range(1, len(recent_detections)):
            interval = (recent_detections[i].timeframe_candle_index -
                       recent_detections[i-1].timeframe_candle_index)
            intervals.append(interval)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            last_detection_index = recent_detections[-1].timeframe_candle_index
            predicted_index = int(last_detection_index + avg_interval)

            logger.debug(f"Predicted next sync point for {timeframe}: {predicted_index}")
            return predicted_index

        return None

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization statistics."""
        total_attempts = len(self.sync_attempts)
        successful_attempts = sum(1 for attempt in self.sync_attempts if attempt.success)

        stats = {
            'total_sync_attempts': total_attempts,
            'successful_syncs': successful_attempts,
            'failed_syncs': total_attempts - successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            'avg_sync_duration': 0.0,
            'avg_sync_quality': 0.0,
            'policy': self.config.policy.value,
            'mode': self.config.mode.value,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None
        }

        if self.sync_attempts:
            durations = [attempt.duration for attempt in self.sync_attempts if attempt.duration > 0]
            if durations:
                stats['avg_sync_duration'] = sum(durations) / len(durations)

            qualities = [attempt.sync_quality for attempt in self.sync_attempts if attempt.sync_quality > 0]
            if qualities:
                stats['avg_sync_quality'] = sum(qualities) / len(qualities)

            # Recent performance (last 10 attempts)
            recent = self.sync_attempts[-10:]
            recent_success = sum(1 for attempt in recent if attempt.success)
            stats['recent_success_rate'] = recent_success / len(recent) if recent else 0.0

        return stats

    def update_configuration(self, new_config: SyncConfiguration):
        """Update synchronization configuration."""
        old_policy = self.config.policy
        self.config = new_config

        logger.info(f"Updated sync configuration: {old_policy.value} -> {new_config.policy.value}")

    def _update_performance_metrics(self, attempt: SyncAttempt):
        """Update internal performance tracking metrics."""
        self.sync_performance['total_attempts'] += 1

        if attempt.success:
            self.sync_performance['successful_syncs'] += 1
        else:
            self.sync_performance['failed_syncs'] += 1

        # Update average sync time
        if attempt.duration > 0:
            current_avg = self.sync_performance['avg_sync_time']
            total_attempts = self.sync_performance['total_attempts']
            self.sync_performance['avg_sync_time'] = (
                (current_avg * (total_attempts - 1) + attempt.duration) / total_attempts
            )

        # Update best sync quality
        if attempt.sync_quality > self.sync_performance['best_sync_quality']:
            self.sync_performance['best_sync_quality'] = attempt.sync_quality


# Factory function for creating sync coordinators
def create_sync_coordinator(policy: SyncPolicy = SyncPolicy.MODERATE,
                          mode: SyncMode = SyncMode.REAL_TIME) -> TimeframeSyncCoordinator:
    """
    Create a timeframe sync coordinator with specified settings.

    Args:
        policy: Synchronization policy to use
        mode: Synchronization mode to use

    Returns:
        Configured TimeframeSyncCoordinator instance
    """
    config = SyncConfiguration(policy=policy, mode=mode)
    return TimeframeSyncCoordinator(config)


# Utility functions for sync management
def calculate_timeframe_lag(tf1: str, tf2: str,
                          tf1_index: int, tf2_index: int) -> float:
    """
    Calculate lag between two timeframes.

    Args:
        tf1: First timeframe
        tf2: Second timeframe
        tf1_index: Index in first timeframe
        tf2_index: Index in second timeframe

    Returns:
        Lag in minutes (positive if tf1 is ahead)
    """
    converter = TimeframeConverter()

    try:
        # Convert both indices to minutes from start
        tf1_minutes = converter.TIMEFRAME_MINUTES[tf1] * tf1_index
        tf2_minutes = converter.TIMEFRAME_MINUTES[tf2] * tf2_index

        return tf1_minutes - tf2_minutes
    except KeyError:
        return 0.0


def estimate_sync_quality(detections_by_timeframe: Dict[str, List[TimeframeDetection]]) -> float:
    """
    Estimate sync quality based on detection patterns.

    Args:
        detections_by_timeframe: Detections grouped by timeframe

    Returns:
        Quality score between 0.0 and 1.0
    """
    if len(detections_by_timeframe) <= 1:
        return 1.0

    timeframes = list(detections_by_timeframe.keys())
    quality_scores = []

    for i, tf1 in enumerate(timeframes):
        for tf2 in timeframes[i+1:]:
            detections1 = detections_by_timeframe[tf1]
            detections2 = detections_by_timeframe[tf2]

            if not detections1 or not detections2:
                continue

            # Calculate temporal alignment
            time_diffs = []
            for d1 in detections1:
                for d2 in detections2:
                    if d1.direction == d2.direction:
                        time_diff = abs((d1.timestamp - d2.timestamp).total_seconds())
                        time_diffs.append(time_diff)

            if time_diffs:
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                # Convert to quality score (lower time diff = higher quality)
                quality = max(0.0, 1.0 - (avg_time_diff / 1800.0))  # 30 min window
                quality_scores.append(quality)

    return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5