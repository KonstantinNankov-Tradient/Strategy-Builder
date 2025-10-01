"""
Import tests for multi-timeframe system components.

Simple tests to verify all multi-timeframe components can be imported successfully.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest


class TestMultiTimeframeImports(unittest.TestCase):
    """Test that all multi-timeframe components can be imported."""

    def test_state_machine_import(self):
        """Test MultiTimeframeStateMachine import."""
        try:
            from core.multi_timeframe_state_machine import (
                MultiTimeframeStateMachine,
                MultiTimeframeActionType
            )
            self.assertIsNotNone(MultiTimeframeStateMachine)
            self.assertIsNotNone(MultiTimeframeActionType)
        except ImportError as e:
            self.fail(f"Failed to import state machine components: {e}")

    def test_coordinator_import(self):
        """Test MultiTimeframeIndicatorCoordinator import."""
        try:
            from core.multi_timeframe_coordinator import (
                MultiTimeframeIndicatorCoordinator,
                CoordinationStrategy,
                IndicatorPriority
            )
            self.assertIsNotNone(MultiTimeframeIndicatorCoordinator)
            self.assertIsNotNone(CoordinationStrategy)
            self.assertIsNotNone(IndicatorPriority)
        except ImportError as e:
            self.fail(f"Failed to import coordinator components: {e}")

    def test_trade_execution_import(self):
        """Test MultiTimeframeTradeExecutor import."""
        try:
            from core.multi_timeframe_trade_execution import (
                MultiTimeframeTradeExecutor,
                TradeDecisionType,
                ExecutionTimingMode
            )
            self.assertIsNotNone(MultiTimeframeTradeExecutor)
            self.assertIsNotNone(TradeDecisionType)
            self.assertIsNotNone(ExecutionTimingMode)
        except ImportError as e:
            self.fail(f"Failed to import trade execution components: {e}")

    def test_state_types_import(self):
        """Test multi-timeframe state types import."""
        try:
            from core.multi_timeframe_state import (
                TimeframeDetection,
                MultiTimeframeSetupContext,
                MultiTimeframeBacktestState
            )
            self.assertIsNotNone(TimeframeDetection)
            self.assertIsNotNone(MultiTimeframeSetupContext)
            self.assertIsNotNone(MultiTimeframeBacktestState)
        except ImportError as e:
            self.fail(f"Failed to import state types: {e}")

    def test_enums_have_expected_values(self):
        """Test that enums have expected values."""
        from core.multi_timeframe_state_machine import MultiTimeframeActionType
        from core.multi_timeframe_coordinator import CoordinationStrategy
        from core.multi_timeframe_trade_execution import TradeDecisionType

        # Test MultiTimeframeActionType
        self.assertTrue(hasattr(MultiTimeframeActionType, 'WAIT'))
        self.assertTrue(hasattr(MultiTimeframeActionType, 'INDICATOR_DETECTED'))

        # Test CoordinationStrategy
        self.assertTrue(hasattr(CoordinationStrategy, 'SEQUENTIAL'))
        self.assertTrue(hasattr(CoordinationStrategy, 'PARALLEL'))

        # Test TradeDecisionType
        self.assertTrue(hasattr(TradeDecisionType, 'NO_ACTION'))
        self.assertTrue(hasattr(TradeDecisionType, 'ENTER_LONG'))

    def test_factory_functions_import(self):
        """Test factory functions import."""
        try:
            from core.multi_timeframe_coordinator import create_indicator_coordinator
            from core.multi_timeframe_trade_execution import create_trade_executor
            self.assertIsNotNone(create_indicator_coordinator)
            self.assertIsNotNone(create_trade_executor)
        except ImportError as e:
            self.fail(f"Failed to import factory functions: {e}")

    def test_json_converter_integration(self):
        """Test JSON converter integration."""
        try:
            from core.json_converter import MultiTimeframeStrategyConfig
            self.assertIsNotNone(MultiTimeframeStrategyConfig)

            # Test basic configuration creation
            config = MultiTimeframeStrategyConfig(
                name="Test",
                version="1.0",
                primary_timeframe="H1"
            )
            self.assertEqual(config.name, "Test")
            self.assertEqual(config.primary_timeframe, "H1")
        except ImportError as e:
            self.fail(f"Failed to import or use JSON converter: {e}")

    def test_class_instantiation_basic(self):
        """Test basic class instantiation without complex dependencies."""
        try:
            from core.json_converter import MultiTimeframeStrategyConfig

            # Create minimal config
            config = MultiTimeframeStrategyConfig(
                name="Test Strategy",
                version="1.0",
                primary_timeframe="H1"
            )

            # Test that classes can be instantiated with config
            from core.multi_timeframe_coordinator import MultiTimeframeIndicatorCoordinator
            from core.multi_timeframe_trade_execution import MultiTimeframeTradeExecutor

            coordinator = MultiTimeframeIndicatorCoordinator(config)
            executor = MultiTimeframeTradeExecutor(config)

            self.assertIsNotNone(coordinator)
            self.assertIsNotNone(executor)
            self.assertEqual(coordinator.strategy_config, config)
            self.assertEqual(executor.strategy_config, config)

        except Exception as e:
            self.fail(f"Failed to instantiate classes: {e}")

    def test_integration_readiness(self):
        """Test that the system is ready for integration."""
        # This test verifies that all components exist and can be imported together
        try:
            from core.multi_timeframe_state_machine import MultiTimeframeStateMachine
            from core.multi_timeframe_coordinator import MultiTimeframeIndicatorCoordinator
            from core.multi_timeframe_trade_execution import MultiTimeframeTradeExecutor
            from core.multi_timeframe_state import (
                TimeframeDetection,
                MultiTimeframeSetupContext
            )
            from core.json_converter import MultiTimeframeStrategyConfig

            # All imports successful
            self.assertTrue(True)

        except ImportError as e:
            self.fail(f"System not ready for integration: {e}")


if __name__ == '__main__':
    unittest.main()