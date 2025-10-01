"""
Test suite for multi-timeframe state types and synchronization.

Tests the multi-timeframe state system, including state coordination,
synchronization, detection handling, and cross-timeframe validation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from core.multi_timeframe_state import (
    MultiTimeframeExecutionState,
    TimeframeSyncStatus,
    TimeframeDetection,
    TimeframeState,
    MultiTimeframeSetupContext,
    MultiTimeframeStateTransition,
    MultiTimeframeBacktestState,
    create_multi_tf_setup,
    convert_legacy_detection_to_timeframe
)

from core.timeframe_sync import (
    TimeframeSyncCoordinator,
    SyncConfiguration,
    SyncPolicy,
    SyncMode,
    SyncAttempt,
    create_sync_coordinator,
    calculate_timeframe_lag,
    estimate_sync_quality
)

from core.multi_timeframe_detection import (
    MultiTimeframeDetectionProcessor,
    DetectionCluster,
    ConfirmationRequest,
    ConfirmationStatus,
    DetectionConfidence,
    create_detection_processor,
    convert_legacy_detection,
    analyze_detection_patterns
)

from core.state_types import SignalDirection, Detection
from core.json_converter import MultiTimeframeStrategyConfig
from core.json_schemas import get_example_strategy


# Suppress logging during tests
logging.getLogger('core.multi_timeframe_state').setLevel(logging.CRITICAL)
logging.getLogger('core.timeframe_sync').setLevel(logging.CRITICAL)
logging.getLogger('core.multi_timeframe_detection').setLevel(logging.CRITICAL)


class TestTimeframeDetection(unittest.TestCase):
    """Test suite for TimeframeDetection class."""

    def test_timeframe_detection_creation(self):
        """Test creating timeframe detection objects."""
        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        self.assertEqual(detection.timeframe, "H1")
        self.assertEqual(detection.timeframe_candle_index, 25)
        self.assertEqual(detection.confidence_score, 1.0)
        self.assertEqual(detection.sync_quality, 1.0)

    def test_timeframe_detection_string_representation(self):
        """Test string representation of timeframe detection."""
        detection = TimeframeDetection(
            indicator_name="choch_detector",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.2345,
            direction=SignalDirection.SHORT,
            timeframe="H4",
            timeframe_candle_index=25
        )

        expected_str = "choch_detector@H4 at 1.23450 (short)"
        self.assertEqual(str(detection), expected_str)


class TestTimeframeState(unittest.TestCase):
    """Test suite for TimeframeState class."""

    def test_timeframe_state_initialization(self):
        """Test TimeframeState initialization."""
        state = TimeframeState(timeframe="H1")

        self.assertEqual(state.timeframe, "H1")
        self.assertEqual(state.current_candle_index, 0)
        self.assertEqual(state.last_processed_index, -1)
        self.assertEqual(state.sync_status, TimeframeSyncStatus.SYNCED)
        self.assertEqual(len(state.active_indicators), 0)

    def test_add_detection(self):
        """Test adding detection to timeframe state."""
        state = TimeframeState(timeframe="H1")

        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        state.add_detection(detection)

        self.assertEqual(len(state.detections), 1)
        self.assertIn("test_indicator", state.completed_indicators)
        self.assertEqual(state.detections[0].timeframe, "H1")

    def test_get_latest_detection(self):
        """Test getting latest detection from timeframe state."""
        state = TimeframeState(timeframe="H1")

        # Add multiple detections
        for i in range(3):
            detection = TimeframeDetection(
                indicator_name=f"indicator_{i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                candle_index=100 + i,
                price=1.1000 + i * 0.0001,
                direction=SignalDirection.LONG,
                timeframe="H1",
                timeframe_candle_index=25 + i
            )
            state.add_detection(detection)

        # Get latest overall
        latest = state.get_latest_detection()
        self.assertEqual(latest.indicator_name, "indicator_2")

        # Get latest for specific indicator
        latest_specific = state.get_latest_detection("indicator_1")
        self.assertEqual(latest_specific.indicator_name, "indicator_1")

    def test_sync_status_with_primary(self):
        """Test synchronization status with primary timeframe."""
        state = TimeframeState(timeframe="H4")
        state.current_candle_index = 100
        state.sync_target_index = 102

        # Within tolerance
        self.assertTrue(state.is_synced_with_primary(primary_index=400, tolerance=2))

        # Outside tolerance
        state.sync_target_index = 110
        self.assertFalse(state.is_synced_with_primary(primary_index=400, tolerance=2))


class TestMultiTimeframeSetupContext(unittest.TestCase):
    """Test suite for MultiTimeframeSetupContext class."""

    def test_multi_tf_setup_initialization(self):
        """Test MultiTimeframeSetupContext initialization."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        self.assertEqual(setup.setup_id, 1)
        self.assertEqual(setup.sync_status, TimeframeSyncStatus.SYNCED)
        self.assertEqual(setup.sync_quality_score, 1.0)
        self.assertEqual(len(setup.timeframe_states), 0)

    def test_initialize_timeframes(self):
        """Test timeframe initialization."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        timeframes = ["H1", "H4", "D1"]
        setup.initialize_timeframes(timeframes, "H1")

        self.assertEqual(setup.primary_timeframe, "H1")
        self.assertEqual(len(setup.timeframe_states), 3)
        self.assertEqual(len(setup.detections_by_timeframe), 3)
        self.assertIn("H1", setup.timeframe_states)
        self.assertIn("H4", setup.timeframe_states)
        self.assertIn("D1", setup.timeframe_states)

    def test_add_timeframe_detection(self):
        """Test adding timeframe-specific detections."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        setup.add_timeframe_detection(detection)

        self.assertEqual(len(setup.detections), 1)
        self.assertEqual(len(setup.detections_by_timeframe["H1"]), 1)
        self.assertEqual(len(setup.timeframe_states["H1"].detections), 1)

    def test_timeframe_progress_calculation(self):
        """Test calculation of timeframe completion progress."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        # Set up some active indicators
        setup.timeframe_states["H1"].active_indicators = ["ind1", "ind2", "ind3"]
        setup.timeframe_states["H4"].active_indicators = ["ind4", "ind5"]

        # Complete some indicators
        setup.timeframe_states["H1"].completed_indicators = ["ind1", "ind2"]
        setup.timeframe_states["H4"].completed_indicators = ["ind4"]

        progress = setup.get_timeframe_progress()

        self.assertAlmostEqual(progress["H1"], 2/3, places=2)  # 2 out of 3
        self.assertAlmostEqual(progress["H4"], 0.5, places=2)  # 1 out of 2

    def test_cross_timeframe_correlation(self):
        """Test cross-timeframe correlation calculation."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        # Add detections to both timeframes
        base_time = datetime.now()

        h1_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=base_time,
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        h4_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=base_time + timedelta(minutes=30),  # Close in time
            candle_index=25,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=25
        )

        setup.add_timeframe_detection(h1_detection)
        setup.add_timeframe_detection(h4_detection)

        correlations = setup.get_cross_timeframe_correlation()

        self.assertIsInstance(correlations, dict)
        self.assertIn(("H1", "H4"), correlations)
        self.assertGreater(correlations[("H1", "H4")], 0.0)


class TestTimeframeSyncCoordinator(unittest.TestCase):
    """Test suite for TimeframeSyncCoordinator class."""

    def test_sync_coordinator_initialization(self):
        """Test sync coordinator initialization."""
        coordinator = TimeframeSyncCoordinator()

        self.assertIsInstance(coordinator.config, SyncConfiguration)
        self.assertEqual(coordinator.config.policy, SyncPolicy.MODERATE)
        self.assertEqual(coordinator.config.mode, SyncMode.REAL_TIME)
        self.assertEqual(len(coordinator.sync_attempts), 0)

    def test_sync_coordinator_with_custom_config(self):
        """Test sync coordinator with custom configuration."""
        config = SyncConfiguration(
            policy=SyncPolicy.STRICT,
            mode=SyncMode.BATCH,
            max_time_drift=600.0
        )

        coordinator = TimeframeSyncCoordinator(config)

        self.assertEqual(coordinator.config.policy, SyncPolicy.STRICT)
        self.assertEqual(coordinator.config.mode, SyncMode.BATCH)
        self.assertEqual(coordinator.config.max_time_drift, 600.0)

    def test_synchronize_timeframes(self):
        """Test timeframe synchronization process."""
        coordinator = TimeframeSyncCoordinator()

        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        # Mock the timeframe converter to avoid dependency issues
        with patch.object(coordinator, 'timeframe_converter') as mock_converter:
            mock_converter.convert_timeframe_index.return_value = 6  # H1 index 25 -> H4 index 6

            result = coordinator.synchronize_timeframes(setup, "H1", detection)

            self.assertIsInstance(result, bool)
            self.assertGreater(len(coordinator.sync_attempts), 0)

    def test_get_sync_statistics(self):
        """Test sync statistics retrieval."""
        coordinator = TimeframeSyncCoordinator()

        stats = coordinator.get_sync_statistics()

        expected_keys = [
            'total_sync_attempts', 'successful_syncs', 'failed_syncs',
            'success_rate', 'policy', 'mode'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_create_sync_coordinator_factory(self):
        """Test sync coordinator factory function."""
        coordinator = create_sync_coordinator(SyncPolicy.RELAXED, SyncMode.BATCH)

        self.assertEqual(coordinator.config.policy, SyncPolicy.RELAXED)
        self.assertEqual(coordinator.config.mode, SyncMode.BATCH)


class TestMultiTimeframeDetectionProcessor(unittest.TestCase):
    """Test suite for MultiTimeframeDetectionProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock multi-timeframe config
        self.mock_config = Mock(spec=MultiTimeframeStrategyConfig)
        self.mock_config.get_all_timeframes.return_value = ["H1", "H4"]
        self.mock_config.primary_timeframe = "H1"
        self.mock_config.required_confirmations = 2
        self.mock_config.execution_mode = Mock()
        self.mock_config.execution_mode.value = 'sequential'

        self.processor = MultiTimeframeDetectionProcessor(self.mock_config)

    def test_processor_initialization(self):
        """Test detection processor initialization."""
        self.assertEqual(self.processor.config, self.mock_config)
        self.assertEqual(len(self.processor.detection_clusters), 0)
        self.assertEqual(len(self.processor.confirmation_requests), 0)

    def test_process_detection(self):
        """Test processing a single detection."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25,
            confidence_score=0.8
        )

        result = self.processor.process_detection(detection, setup)

        self.assertIsInstance(result, dict)
        self.assertIn('detection_id', result)
        self.assertIn('accepted', result)
        self.assertIn('recommendation', result)
        self.assertIn('confidence', result)

    def test_detection_clustering(self):
        """Test detection clustering functionality."""
        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        base_time = datetime.now()

        # Create related detections
        detection1 = TimeframeDetection(
            indicator_name="liquidity_grab_detector",
            timestamp=base_time,
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        detection2 = TimeframeDetection(
            indicator_name="liquidity_grab_detector",
            timestamp=base_time + timedelta(minutes=2),
            candle_index=101,
            price=1.1001,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=26
        )

        # Process both detections
        self.processor.process_detection(detection1, setup)
        self.processor.process_detection(detection2, setup)

        # Should have created clusters
        self.assertGreater(len(self.processor.detection_clusters), 0)

    def test_get_detection_statistics(self):
        """Test detection processing statistics."""
        stats = self.processor.get_detection_statistics()

        expected_keys = [
            'total_detections_processed', 'clustered_detections',
            'confirmed_detections', 'rejected_detections',
            'active_clusters', 'pending_confirmations'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_confirmation_request_creation(self):
        """Test creation of confirmation requests."""
        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H4",  # Non-primary timeframe
            timeframe_candle_index=25
        )

        setup = MultiTimeframeSetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=0
        )

        setup.initialize_timeframes(["H1", "H4"], "H1")

        request = self.processor._create_confirmation_request(detection, setup)

        self.assertIsInstance(request, ConfirmationRequest)
        self.assertEqual(request.detection, detection)
        self.assertEqual(request.status, ConfirmationStatus.PENDING)
        self.assertIn("H1", request.target_timeframes)


class TestDetectionCluster(unittest.TestCase):
    """Test suite for DetectionCluster class."""

    def test_cluster_creation(self):
        """Test detection cluster creation."""
        primary_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        cluster = DetectionCluster(
            cluster_id="test_cluster",
            primary_detection=primary_detection
        )

        self.assertEqual(cluster.cluster_id, "test_cluster")
        self.assertEqual(cluster.primary_detection, primary_detection)
        self.assertEqual(len(cluster.related_detections), 0)

    def test_add_detection_to_cluster(self):
        """Test adding detections to cluster."""
        primary_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=25
        )

        cluster = DetectionCluster(
            cluster_id="test_cluster",
            primary_detection=primary_detection
        )

        additional_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now() + timedelta(minutes=5),
            candle_index=105,
            price=1.1005,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=26
        )

        cluster.add_detection(additional_detection)

        self.assertEqual(len(cluster.related_detections), 1)
        self.assertIn("H4", cluster.timeframes_involved)
        self.assertEqual(cluster.consensus_direction, SignalDirection.LONG)


class TestConfirmationRequest(unittest.TestCase):
    """Test suite for ConfirmationRequest class."""

    def test_confirmation_request_creation(self):
        """Test confirmation request creation."""
        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=25
        )

        request = ConfirmationRequest(
            request_id="test_request",
            detection=detection,
            target_timeframes=["H1"],
            required_confirmations=1
        )

        self.assertEqual(request.request_id, "test_request")
        self.assertEqual(request.detection, detection)
        self.assertEqual(request.status, ConfirmationStatus.PENDING)
        self.assertIsNotNone(request.expires_at)

    def test_add_confirmation(self):
        """Test adding confirmation to request."""
        original_detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=25
        )

        request = ConfirmationRequest(
            request_id="test_request",
            detection=original_detection,
            target_timeframes=["H1"],
            required_confirmations=1
        )

        confirmation = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now() + timedelta(minutes=2),
            candle_index=102,
            price=1.1002,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=27
        )

        success = request.add_confirmation(confirmation)

        self.assertTrue(success)
        self.assertEqual(len(request.received_confirmations), 1)
        self.assertEqual(request.status, ConfirmationStatus.CONFIRMED)

    def test_confirmation_timeout(self):
        """Test confirmation request timeout."""
        detection = TimeframeDetection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=25
        )

        # Create request with very short timeout
        request = ConfirmationRequest(
            request_id="test_request",
            detection=detection,
            target_timeframes=["H1"],
            required_confirmations=1,
            timeout_seconds=0.1  # 100ms timeout
        )

        # Wait for timeout
        import time
        time.sleep(0.2)

        self.assertTrue(request.is_expired())

        request.check_timeout()
        self.assertEqual(request.status, ConfirmationStatus.TIMEOUT)


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def test_create_multi_tf_setup(self):
        """Test multi-timeframe setup creation utility."""
        mock_config = Mock(spec=MultiTimeframeStrategyConfig)
        mock_config.get_all_timeframes.return_value = ["H1", "H4"]
        mock_config.primary_timeframe = "H1"

        setup = create_multi_tf_setup(
            setup_id=1,
            timestamp=datetime.now(),
            candle_index=0,
            config=mock_config
        )

        self.assertIsInstance(setup, MultiTimeframeSetupContext)
        self.assertEqual(setup.setup_id, 1)
        self.assertEqual(len(setup.timeframe_states), 2)

    def test_convert_legacy_detection_to_timeframe(self):
        """Test conversion of legacy Detection to TimeframeDetection."""
        legacy_detection = Detection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG
        )

        timeframe_detection = convert_legacy_detection_to_timeframe(legacy_detection, "H1")

        self.assertIsInstance(timeframe_detection, TimeframeDetection)
        self.assertEqual(timeframe_detection.timeframe, "H1")
        self.assertEqual(timeframe_detection.indicator_name, legacy_detection.indicator_name)
        self.assertEqual(timeframe_detection.direction, legacy_detection.direction)

    def test_convert_legacy_detection_utility(self):
        """Test legacy detection conversion utility function."""
        legacy_detection = Detection(
            indicator_name="test_indicator",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.SHORT
        )

        timeframe_detection = convert_legacy_detection(legacy_detection, "H4")

        self.assertIsInstance(timeframe_detection, TimeframeDetection)
        self.assertEqual(timeframe_detection.timeframe, "H4")
        self.assertEqual(timeframe_detection.confidence_score, 0.8)

    def test_analyze_detection_patterns(self):
        """Test detection pattern analysis utility."""
        detections = []
        base_time = datetime.now()

        for i in range(5):
            detection = TimeframeDetection(
                indicator_name=f"indicator_{i}",
                timestamp=base_time + timedelta(minutes=i * 10),
                candle_index=100 + i,
                price=1.1000 + i * 0.0001,
                direction=SignalDirection.LONG if i % 2 == 0 else SignalDirection.SHORT,
                timeframe="H1",
                timeframe_candle_index=25 + i
            )
            detections.append(detection)

        analysis = analyze_detection_patterns(detections)

        self.assertIsInstance(analysis, dict)
        self.assertIn('pattern_strength', analysis)
        self.assertIn('dominant_direction', analysis)
        self.assertIn('avg_confidence', analysis)
        self.assertIn('timeframe_count', analysis)

    def test_calculate_timeframe_lag_utility(self):
        """Test timeframe lag calculation utility."""
        lag = calculate_timeframe_lag("H1", "H4", 100, 25)
        self.assertIsInstance(lag, (int, float))  # Accept both int and float
        self.assertNotEqual(lag, None)

    def test_estimate_sync_quality_utility(self):
        """Test sync quality estimation utility."""
        detections_by_tf = {
            "H1": [
                TimeframeDetection(
                    indicator_name="test",
                    timestamp=datetime.now(),
                    candle_index=100,
                    price=1.1000,
                    direction=SignalDirection.LONG,
                    timeframe="H1",
                    timeframe_candle_index=25
                )
            ],
            "H4": [
                TimeframeDetection(
                    indicator_name="test",
                    timestamp=datetime.now() + timedelta(minutes=5),
                    candle_index=25,
                    price=1.1000,
                    direction=SignalDirection.LONG,
                    timeframe="H4",
                    timeframe_candle_index=25
                )
            ]
        }

        quality = estimate_sync_quality(detections_by_tf)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-timeframe state system."""

    def setUp(self):
        """Set up integration test fixtures."""
        from core.json_converter import JSONToObjectConverter

        # Create a real multi-timeframe config
        example_json = get_example_strategy()
        converter = JSONToObjectConverter()
        self.config = converter.convert_json_to_strategy(example_json)

    def test_complete_multi_timeframe_workflow(self):
        """Test complete multi-timeframe detection and synchronization workflow."""
        # Initialize components
        processor = create_detection_processor(self.config)
        coordinator = create_sync_coordinator()

        # Create setup
        setup = create_multi_tf_setup(
            setup_id=1,
            timestamp=datetime.now(),
            candle_index=0,
            config=self.config
        )

        # Create detections on different timeframes
        base_time = datetime.now()

        h1_detection = TimeframeDetection(
            indicator_name="liquidity_grab_detector",
            timestamp=base_time,
            candle_index=100,
            price=1.1000,
            direction=SignalDirection.LONG,
            timeframe="H1",
            timeframe_candle_index=100
        )

        h4_detection = TimeframeDetection(
            indicator_name="choch_detector",
            timestamp=base_time + timedelta(minutes=30),
            candle_index=25,
            price=1.1005,
            direction=SignalDirection.LONG,
            timeframe="H4",
            timeframe_candle_index=25
        )

        # Process detections
        result1 = processor.process_detection(h1_detection, setup)
        result2 = processor.process_detection(h4_detection, setup)

        # Verify processing results
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)

        # Verify setup state
        self.assertGreater(len(setup.detections), 0)
        self.assertGreater(len(setup.detections_by_timeframe), 0)

        # Test synchronization
        with patch.object(coordinator, 'timeframe_converter') as mock_converter:
            mock_converter.convert_timeframe_index.return_value = 25

            sync_result = coordinator.synchronize_timeframes(setup, "H1", h1_detection)
            self.assertIsInstance(sync_result, bool)

    def test_multi_timeframe_state_transitions(self):
        """Test multi-timeframe state transitions."""
        # Create a mock legacy config for the parent class
        from core.state_types import StrategyConfig
        legacy_config = StrategyConfig(
            indicator_sequence=["test_indicator"],
            required_confirmations=1,
            timeouts={'default': 30}
        )

        backtest_state = MultiTimeframeBacktestState(strategy_config=legacy_config)
        backtest_state.initialize_timeframes(self.config)

        # Verify initialization
        self.assertEqual(backtest_state.primary_timeframe, self.config.primary_timeframe)
        self.assertGreater(len(backtest_state.active_timeframes), 1)
        self.assertTrue(backtest_state.can_start_new_setup())

        # Test state progression
        initial_state = backtest_state.current_tf_state
        next_state = backtest_state.get_next_multi_tf_state()

        # Should be able to progress states
        self.assertIsInstance(next_state, MultiTimeframeExecutionState)


if __name__ == '__main__':
    unittest.main()