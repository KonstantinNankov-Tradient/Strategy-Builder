"""
Timeframe-aware detection handling for multi-timeframe strategies.

This module provides enhanced detection processing that handles indicators
across multiple timeframes, including detection correlation, validation,
and cross-timeframe confirmation logic.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import numpy as np

from .multi_timeframe_state import (
    TimeframeDetection,
    MultiTimeframeSetupContext,
    TimeframeSyncStatus
)
from .data_loader import TimeframeConverter
from .json_converter import MultiTimeframeStrategyConfig
from .state_types import SignalDirection, Detection


logger = logging.getLogger(__name__)


class DetectionConfidence(Enum):
    """Confidence levels for detections."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


class ConfirmationStatus(Enum):
    """Status of cross-timeframe confirmation."""
    PENDING = "pending"         # Waiting for confirmation
    CONFIRMED = "confirmed"     # Confirmed across timeframes
    REJECTED = "rejected"       # Rejected due to conflicts
    TIMEOUT = "timeout"         # Confirmation timeout


@dataclass
class DetectionCluster:
    """
    Groups related detections across timeframes.

    A cluster represents detections that are likely related to the same
    market event but occurred on different timeframes.
    """
    cluster_id: str
    primary_detection: TimeframeDetection
    related_detections: List[TimeframeDetection] = field(default_factory=list)

    # Cluster properties
    timeframes_involved: Set[str] = field(default_factory=set)
    consensus_direction: SignalDirection = SignalDirection.NONE
    confidence_score: float = 0.0
    correlation_strength: float = 0.0

    # Timing
    earliest_detection: Optional[datetime] = None
    latest_detection: Optional[datetime] = None
    time_span: Optional[timedelta] = None

    def add_detection(self, detection: TimeframeDetection):
        """Add a detection to this cluster."""
        self.related_detections.append(detection)
        self.timeframes_involved.add(detection.timeframe)

        # Update timing
        if not self.earliest_detection or detection.timestamp < self.earliest_detection:
            self.earliest_detection = detection.timestamp
        if not self.latest_detection or detection.timestamp > self.latest_detection:
            self.latest_detection = detection.timestamp

        if self.earliest_detection and self.latest_detection:
            self.time_span = self.latest_detection - self.earliest_detection

        # Update consensus
        self._update_consensus()

    def _update_consensus(self):
        """Update consensus direction and confidence."""
        all_detections = [self.primary_detection] + self.related_detections

        # Count direction votes
        long_votes = sum(1 for d in all_detections if d.direction == SignalDirection.LONG)
        short_votes = sum(1 for d in all_detections if d.direction == SignalDirection.SHORT)
        total_votes = long_votes + short_votes

        if total_votes == 0:
            self.consensus_direction = SignalDirection.NONE
            self.confidence_score = 0.0
        elif long_votes > short_votes:
            self.consensus_direction = SignalDirection.LONG
            self.confidence_score = long_votes / total_votes
        elif short_votes > long_votes:
            self.consensus_direction = SignalDirection.SHORT
            self.confidence_score = short_votes / total_votes
        else:
            self.consensus_direction = SignalDirection.NONE
            self.confidence_score = 0.5  # Tie

        # Adjust confidence based on timeframe diversity
        tf_count = len(self.timeframes_involved)
        if tf_count > 1:
            # Bonus for cross-timeframe agreement
            self.confidence_score = min(1.0, self.confidence_score * (1.0 + tf_count * 0.1))


@dataclass
class ConfirmationRequest:
    """Request for cross-timeframe confirmation of a detection."""
    request_id: str
    detection: TimeframeDetection
    target_timeframes: List[str]
    required_confirmations: int
    timeout_seconds: float = 300.0  # 5 minutes default

    # Request status
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    received_confirmations: List[TimeframeDetection] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        """Set expiration time after initialization."""
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(seconds=self.timeout_seconds)

    def add_confirmation(self, confirmation: TimeframeDetection) -> bool:
        """
        Add a confirmation detection.

        Returns:
            True if confirmation was accepted, False if rejected
        """
        if self.status != ConfirmationStatus.PENDING:
            return False

        # Validate confirmation
        if not self._validate_confirmation(confirmation):
            self.rejection_reasons.append(f"Invalid confirmation from {confirmation.timeframe}")
            return False

        self.received_confirmations.append(confirmation)

        # Check if we have enough confirmations
        if len(self.received_confirmations) >= self.required_confirmations:
            self.status = ConfirmationStatus.CONFIRMED

        return True

    def _validate_confirmation(self, confirmation: TimeframeDetection) -> bool:
        """Validate a confirmation detection."""
        # Check direction agreement
        if confirmation.direction != self.detection.direction:
            return False

        # Check timing (within reasonable window)
        time_diff = abs((confirmation.timestamp - self.detection.timestamp).total_seconds())
        if time_diff > 1800:  # 30 minutes max
            return False

        return True

    def is_expired(self) -> bool:
        """Check if the confirmation request has expired."""
        return datetime.now() > self.expires_at

    def check_timeout(self):
        """Check and update status if timeout occurred."""
        if self.is_expired() and self.status == ConfirmationStatus.PENDING:
            self.status = ConfirmationStatus.TIMEOUT


class MultiTimeframeDetectionProcessor:
    """
    Processes detections across multiple timeframes.

    Handles detection correlation, clustering, validation, and cross-timeframe
    confirmation for multi-timeframe trading strategies.
    """

    def __init__(self, config: MultiTimeframeStrategyConfig):
        """
        Initialize the detection processor.

        Args:
            config: Multi-timeframe strategy configuration
        """
        self.config = config
        self.timeframe_converter = TimeframeConverter()

        # Detection processing state
        self.detection_clusters: List[DetectionCluster] = []
        self.confirmation_requests: Dict[str, ConfirmationRequest] = {}
        self.processed_detections: List[TimeframeDetection] = []

        # Correlation settings
        self.correlation_window = 900.0  # 15 minutes correlation window
        self.min_correlation_strength = 0.6
        self.max_cluster_size = 10

        # Performance tracking
        self.processing_stats = {
            'total_detections': 0,
            'clustered_detections': 0,
            'confirmed_detections': 0,
            'rejected_detections': 0,
            'timeout_detections': 0
        }

    def process_detection(self, detection: TimeframeDetection,
                         setup: MultiTimeframeSetupContext) -> Dict[str, Any]:
        """
        Process a new detection in the context of multi-timeframe strategy.

        Args:
            detection: New detection to process
            setup: Current multi-timeframe setup context

        Returns:
            Processing result with recommendations and actions
        """
        logger.debug(f"Processing detection: {detection}")

        self.processing_stats['total_detections'] += 1
        result = {
            'detection_id': f"{detection.timeframe}_{detection.candle_index}_{detection.indicator_name}",
            'accepted': False,
            'cluster_id': None,
            'confirmation_required': False,
            'recommendation': 'wait',
            'confidence': detection.confidence_score,
            'actions': []
        }

        try:
            # Step 1: Enhance detection with correlation data
            enhanced_detection = self._enhance_detection(detection, setup)

            # Step 2: Find or create detection cluster
            cluster = self._find_or_create_cluster(enhanced_detection, setup)
            if cluster:
                result['cluster_id'] = cluster.cluster_id
                result['confidence'] = cluster.confidence_score
                self.processing_stats['clustered_detections'] += 1

            # Step 3: Validate detection against strategy rules
            validation_result = self._validate_detection(enhanced_detection, setup)
            if not validation_result['valid']:
                result['recommendation'] = 'reject'
                result['actions'].append(f"Rejected: {validation_result['reason']}")
                self.processing_stats['rejected_detections'] += 1
                return result

            # Step 4: Check if cross-timeframe confirmation is needed
            if self._requires_confirmation(enhanced_detection, setup):
                confirmation_request = self._create_confirmation_request(enhanced_detection, setup)
                self.confirmation_requests[confirmation_request.request_id] = confirmation_request
                result['confirmation_required'] = True
                result['recommendation'] = 'wait_confirmation'
                result['actions'].append("Waiting for cross-timeframe confirmation")
            else:
                result['accepted'] = True
                result['recommendation'] = 'accept'
                result['actions'].append("Detection accepted")
                self.processing_stats['confirmed_detections'] += 1

            # Step 5: Update setup context
            setup.add_timeframe_detection(enhanced_detection)
            self.processed_detections.append(enhanced_detection)

            # Step 6: Check pending confirmations
            self._check_pending_confirmations(enhanced_detection)

            return result

        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            result['recommendation'] = 'error'
            result['actions'].append(f"Processing error: {str(e)}")
            return result

    def _enhance_detection(self, detection: TimeframeDetection,
                          setup: MultiTimeframeSetupContext) -> TimeframeDetection:
        """Enhance detection with additional correlation and context data."""
        enhanced = detection

        # Calculate equivalent indices on other timeframes
        for timeframe in setup.timeframe_states.keys():
            if timeframe == detection.timeframe:
                continue

            try:
                equivalent_index = self.timeframe_converter.convert_timeframe_index(
                    detection.timeframe_candle_index,
                    detection.timeframe,
                    timeframe
                )
                enhanced.correlated_detections[timeframe] = equivalent_index
            except Exception as e:
                logger.warning(f"Failed to calculate equivalent index for {timeframe}: {e}")

        # Calculate sync quality with other timeframes
        sync_scores = []
        for tf_name, tf_state in setup.timeframe_states.items():
            if tf_name == detection.timeframe:
                continue

            # Check how well this detection aligns with timeframe state
            if tf_state.sync_target_index:
                deviation = abs(detection.timeframe_candle_index - tf_state.current_candle_index)
                sync_score = max(0.0, 1.0 - (deviation / 5.0))
                sync_scores.append(sync_score)

        enhanced.sync_quality = sum(sync_scores) / len(sync_scores) if sync_scores else 1.0

        return enhanced

    def _find_or_create_cluster(self, detection: TimeframeDetection,
                               setup: MultiTimeframeSetupContext) -> Optional[DetectionCluster]:
        """Find existing cluster for detection or create new one."""
        # Look for existing clusters within correlation window
        for cluster in self.detection_clusters:
            if self._should_join_cluster(detection, cluster):
                cluster.add_detection(detection)
                return cluster

        # Create new cluster
        cluster = DetectionCluster(
            cluster_id=f"cluster_{len(self.detection_clusters)}_{detection.timeframe}",
            primary_detection=detection
        )
        cluster.timeframes_involved.add(detection.timeframe)
        cluster.earliest_detection = detection.timestamp
        cluster.latest_detection = detection.timestamp

        self.detection_clusters.append(cluster)
        return cluster

    def _should_join_cluster(self, detection: TimeframeDetection,
                           cluster: DetectionCluster) -> bool:
        """Determine if detection should join existing cluster."""
        # Check time window
        if cluster.latest_detection:
            time_diff = abs((detection.timestamp - cluster.latest_detection).total_seconds())
            if time_diff > self.correlation_window:
                return False

        # Check direction compatibility
        if (cluster.consensus_direction != SignalDirection.NONE and
            detection.direction != SignalDirection.NONE and
            cluster.consensus_direction != detection.direction):
            return False

        # Check indicator compatibility (same indicator family)
        primary_indicator = cluster.primary_detection.indicator_name
        if self._are_indicators_related(primary_indicator, detection.indicator_name):
            return True

        # Check spatial correlation (price levels)
        price_diff = abs(detection.price - cluster.primary_detection.price)
        if price_diff / cluster.primary_detection.price < 0.01:  # Within 1%
            return True

        return False

    def _are_indicators_related(self, indicator1: str, indicator2: str) -> bool:
        """Check if two indicators are related/compatible."""
        # Define indicator families
        indicator_families = {
            'trend': ['choch_detector', 'bos_detector'],
            'liquidity': ['liquidity_grab_detector', 'order_block_detector'],
            'structure': ['fvg_detector', 'order_block_detector']
        }

        for family, indicators in indicator_families.items():
            if indicator1 in indicators and indicator2 in indicators:
                return True

        return indicator1 == indicator2

    def _validate_detection(self, detection: TimeframeDetection,
                           setup: MultiTimeframeSetupContext) -> Dict[str, Any]:
        """Validate detection against strategy rules and constraints."""
        result = {'valid': True, 'reason': None, 'confidence_adjustment': 0.0}

        # Check if detection is on expected timeframe
        expected_timeframes = self.config.get_all_timeframes()
        if detection.timeframe not in expected_timeframes:
            result['valid'] = False
            result['reason'] = f"Unexpected timeframe: {detection.timeframe}"
            return result

        # Check minimum confidence threshold
        if detection.confidence_score < 0.3:
            result['valid'] = False
            result['reason'] = "Confidence below minimum threshold"
            return result

        # Check for conflicting signals on same timeframe
        timeframe_state = setup.timeframe_states.get(detection.timeframe)
        if timeframe_state:
            recent_detections = [d for d in timeframe_state.detections
                               if (detection.timestamp - d.timestamp).total_seconds() < 300]  # 5 min window

            conflicting = [d for d in recent_detections
                          if d.direction != detection.direction and d.direction != SignalDirection.NONE]

            if conflicting:
                result['confidence_adjustment'] = -0.2  # Reduce confidence
                result['reason'] = f"Conflicting signals detected on {detection.timeframe}"

        # Check sync quality
        if detection.sync_quality < 0.5:
            result['confidence_adjustment'] -= 0.1

        # Adjust detection confidence
        detection.confidence_score = max(0.1, detection.confidence_score + result['confidence_adjustment'])

        return result

    def _requires_confirmation(self, detection: TimeframeDetection,
                              setup: MultiTimeframeSetupContext) -> bool:
        """Determine if detection requires cross-timeframe confirmation."""
        # Check strategy execution mode
        if self.config.execution_mode.value == 'parallel':
            return False  # Parallel mode doesn't require confirmation

        # Check if this is the primary timeframe
        if detection.timeframe == self.config.primary_timeframe:
            return len(self.config.get_all_timeframes()) > 1  # Only if multi-timeframe

        # Secondary timeframes always need confirmation
        return True

    def _create_confirmation_request(self, detection: TimeframeDetection,
                                   setup: MultiTimeframeSetupContext) -> ConfirmationRequest:
        """Create a confirmation request for cross-timeframe validation."""
        # Determine target timeframes
        all_timeframes = self.config.get_all_timeframes()
        target_timeframes = [tf for tf in all_timeframes if tf != detection.timeframe]

        # Calculate required confirmations based on strategy
        required_confirmations = max(1, self.config.required_confirmations - 1)

        request_id = f"conf_{detection.timeframe}_{detection.candle_index}_{detection.indicator_name}"

        return ConfirmationRequest(
            request_id=request_id,
            detection=detection,
            target_timeframes=target_timeframes,
            required_confirmations=required_confirmations
        )

    def _check_pending_confirmations(self, new_detection: TimeframeDetection):
        """Check if new detection provides confirmation for pending requests."""
        for request_id, request in list(self.confirmation_requests.items()):
            if request.status != ConfirmationStatus.PENDING:
                continue

            # Check if detection is from target timeframe
            if new_detection.timeframe not in request.target_timeframes:
                continue

            # Check if detection can confirm the request
            if self._can_confirm_detection(request.detection, new_detection):
                success = request.add_confirmation(new_detection)
                if success:
                    logger.debug(f"Added confirmation to request {request_id}")

                    if request.status == ConfirmationStatus.CONFIRMED:
                        logger.info(f"Confirmation request {request_id} completed")
                        self.processing_stats['confirmed_detections'] += 1

    def _can_confirm_detection(self, original: TimeframeDetection,
                              confirmation: TimeframeDetection) -> bool:
        """Check if confirmation detection can confirm the original."""
        # Check direction agreement
        if original.direction != confirmation.direction:
            return False

        # Check indicator compatibility
        if not self._are_indicators_related(original.indicator_name, confirmation.indicator_name):
            return False

        # Check timing window
        time_diff = abs((original.timestamp - confirmation.timestamp).total_seconds())
        if time_diff > 900:  # 15 minutes max
            return False

        return True

    def process_timeouts(self):
        """Process confirmation request timeouts."""
        current_time = datetime.now()

        for request_id, request in list(self.confirmation_requests.items()):
            if request.status == ConfirmationStatus.PENDING and request.is_expired():
                request.check_timeout()
                if request.status == ConfirmationStatus.TIMEOUT:
                    logger.warning(f"Confirmation request {request_id} timed out")
                    self.processing_stats['timeout_detections'] += 1

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection processing statistics."""
        total = self.processing_stats['total_detections']

        stats = {
            'total_detections_processed': total,
            'clustered_detections': self.processing_stats['clustered_detections'],
            'confirmed_detections': self.processing_stats['confirmed_detections'],
            'rejected_detections': self.processing_stats['rejected_detections'],
            'timeout_detections': self.processing_stats['timeout_detections'],
            'active_clusters': len(self.detection_clusters),
            'pending_confirmations': len([r for r in self.confirmation_requests.values()
                                         if r.status == ConfirmationStatus.PENDING]),
            'confirmation_rate': (self.processing_stats['confirmed_detections'] / total
                                 if total > 0 else 0.0),
            'rejection_rate': (self.processing_stats['rejected_detections'] / total
                              if total > 0 else 0.0)
        }

        return stats

    def get_cluster_analysis(self) -> List[Dict[str, Any]]:
        """Get analysis of detection clusters."""
        analysis = []

        for cluster in self.detection_clusters:
            cluster_data = {
                'cluster_id': cluster.cluster_id,
                'detection_count': len(cluster.related_detections) + 1,  # Include primary
                'timeframes_involved': list(cluster.timeframes_involved),
                'consensus_direction': cluster.consensus_direction.value,
                'confidence_score': cluster.confidence_score,
                'correlation_strength': cluster.correlation_strength,
                'time_span_seconds': cluster.time_span.total_seconds() if cluster.time_span else 0
            }
            analysis.append(cluster_data)

        return analysis

    def reset(self):
        """Reset all detection processing state."""
        self.detection_clusters.clear()
        self.confirmation_requests.clear()
        self.processed_detections.clear()

        # Reset statistics
        for key in self.processing_stats:
            self.processing_stats[key] = 0


# Utility functions for detection processing
def create_detection_processor(config: MultiTimeframeStrategyConfig) -> MultiTimeframeDetectionProcessor:
    """
    Create a multi-timeframe detection processor.

    Args:
        config: Multi-timeframe strategy configuration

    Returns:
        Configured MultiTimeframeDetectionProcessor instance
    """
    return MultiTimeframeDetectionProcessor(config)


def convert_legacy_detection(legacy_detection: Detection, timeframe: str) -> TimeframeDetection:
    """
    Convert a legacy Detection to TimeframeDetection.

    Args:
        legacy_detection: Legacy detection object
        timeframe: Timeframe to assign to the detection

    Returns:
        Enhanced TimeframeDetection object
    """
    return TimeframeDetection(
        indicator_name=legacy_detection.indicator_name,
        timestamp=legacy_detection.timestamp,
        candle_index=legacy_detection.candle_index,
        price=legacy_detection.price,
        direction=legacy_detection.direction,
        metadata=legacy_detection.metadata,
        timeframe=timeframe,
        timeframe_candle_index=legacy_detection.candle_index,
        confidence_score=0.8  # Default confidence for legacy detections
    )


def analyze_detection_patterns(detections: List[TimeframeDetection]) -> Dict[str, Any]:
    """
    Analyze patterns in detection data.

    Args:
        detections: List of detections to analyze

    Returns:
        Pattern analysis results
    """
    if not detections:
        return {'pattern_strength': 0.0, 'dominant_direction': SignalDirection.NONE}

    # Direction analysis
    long_count = sum(1 for d in detections if d.direction == SignalDirection.LONG)
    short_count = sum(1 for d in detections if d.direction == SignalDirection.SHORT)
    total_count = long_count + short_count

    if total_count == 0:
        dominant_direction = SignalDirection.NONE
        direction_strength = 0.0
    elif long_count > short_count:
        dominant_direction = SignalDirection.LONG
        direction_strength = long_count / total_count
    else:
        dominant_direction = SignalDirection.SHORT
        direction_strength = short_count / total_count

    # Timeframe analysis
    timeframes = list(set(d.timeframe for d in detections))
    timeframe_distribution = {tf: sum(1 for d in detections if d.timeframe == tf)
                            for tf in timeframes}

    # Confidence analysis
    avg_confidence = sum(d.confidence_score for d in detections) / len(detections)

    # Temporal analysis
    if len(detections) > 1:
        time_spans = []
        sorted_detections = sorted(detections, key=lambda d: d.timestamp)
        for i in range(1, len(sorted_detections)):
            span = (sorted_detections[i].timestamp - sorted_detections[i-1].timestamp).total_seconds()
            time_spans.append(span)
        avg_time_span = sum(time_spans) / len(time_spans)
    else:
        avg_time_span = 0.0

    return {
        'pattern_strength': direction_strength * avg_confidence,
        'dominant_direction': dominant_direction,
        'direction_strength': direction_strength,
        'avg_confidence': avg_confidence,
        'timeframe_count': len(timeframes),
        'timeframe_distribution': timeframe_distribution,
        'avg_time_between_detections': avg_time_span,
        'total_detections': len(detections)
    }