"""
Comprehensive tests for state_types module.

Tests all data structures, enums, and logic in the state system.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_types import (
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


class TestEnums(unittest.TestCase):
    """Test all enum definitions."""

    def test_execution_state_values(self):
        """Test ExecutionState enum values."""
        self.assertEqual(ExecutionState.SCANNING.value, "scanning")
        self.assertEqual(ExecutionState.SIGNAL_1.value, "signal_1")
        self.assertEqual(ExecutionState.IN_POSITION.value, "in_position")

        # Test all states exist
        states = [
            ExecutionState.SCANNING,
            ExecutionState.SIGNAL_1,
            ExecutionState.SIGNAL_2,
            ExecutionState.SIGNAL_3,
            ExecutionState.SIGNAL_4,
            ExecutionState.SIGNAL_5,
            ExecutionState.READY_TO_ENTER,
            ExecutionState.IN_POSITION,
            ExecutionState.POSITION_CLOSED
        ]
        self.assertEqual(len(states), 9)

    def test_action_type_values(self):
        """Test ActionType enum values."""
        self.assertEqual(ActionType.WAIT.value, "wait")
        self.assertEqual(ActionType.ENTER_TRADE.value, "enter_trade")
        self.assertEqual(ActionType.TIMEOUT.value, "timeout")

    def test_signal_direction_values(self):
        """Test SignalDirection enum values."""
        self.assertEqual(SignalDirection.LONG.value, "long")
        self.assertEqual(SignalDirection.SHORT.value, "short")
        self.assertEqual(SignalDirection.NONE.value, "none")


class TestDetection(unittest.TestCase):
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a detection."""
        now = datetime.now()
        detection = Detection(
            indicator_name="test_indicator",
            timestamp=now,
            candle_index=100,
            price=1.1050,
            direction=SignalDirection.LONG,
            metadata={"volume": 1000}
        )

        self.assertEqual(detection.indicator_name, "test_indicator")
        self.assertEqual(detection.timestamp, now)
        self.assertEqual(detection.candle_index, 100)
        self.assertEqual(detection.price, 1.1050)
        self.assertEqual(detection.direction, SignalDirection.LONG)
        self.assertEqual(detection.metadata["volume"], 1000)

    def test_detection_string_representation(self):
        """Test string representation of detection."""
        detection = Detection(
            indicator_name="rsi_oversold",
            timestamp=datetime.now(),
            candle_index=100,
            price=1.10505,
            direction=SignalDirection.LONG
        )

        str_repr = str(detection)
        self.assertIn("rsi_oversold", str_repr)
        self.assertIn("1.10505", str_repr)
        self.assertIn("long", str_repr)


class TestStrategyConfig(unittest.TestCase):
    """Test StrategyConfig dataclass."""

    def test_strategy_config_creation(self):
        """Test creating a strategy configuration."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2", "ind3"],
            required_confirmations=2,
            timeouts={"ind1": 30, "ind2": 20, "default": 50},
            stop_loss_pips=25,
            take_profit_pips=50,
            risk_percent=0.02
        )

        self.assertEqual(len(config.indicator_sequence), 3)
        self.assertEqual(config.required_confirmations, 2)
        self.assertEqual(config.stop_loss_pips, 25)
        self.assertEqual(config.risk_percent, 0.02)

    def test_get_timeout_after(self):
        """Test timeout retrieval for indicators."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2"],
            timeouts={"ind1": 30, "default": 50}
        )

        # Specific timeout
        self.assertEqual(config.get_timeout_after("ind1"), 30)

        # Default timeout
        self.assertEqual(config.get_timeout_after("ind2"), 50)
        self.assertEqual(config.get_timeout_after("unknown"), 50)

    def test_get_next_indicator(self):
        """Test next indicator logic."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2", "ind3"],
            required_confirmations=2
        )

        # No detections yet
        next_ind = config.get_next_indicator([])
        self.assertEqual(next_ind, "ind1")

        # One detection
        next_ind = config.get_next_indicator(["ind1"])
        self.assertEqual(next_ind, "ind2")

        # Two detections - enough confirmations
        next_ind = config.get_next_indicator(["ind1", "ind2"])
        self.assertIsNone(next_ind)

        # Skip detected indicators
        next_ind = config.get_next_indicator(["ind1", "ind3"])
        self.assertIsNone(next_ind)  # Already have 2 confirmations

    def test_get_next_indicator_all_required(self):
        """Test when all indicators are required."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2", "ind3"],
            required_confirmations=3  # All required
        )

        next_ind = config.get_next_indicator(["ind1", "ind2"])
        self.assertEqual(next_ind, "ind3")

        next_ind = config.get_next_indicator(["ind1", "ind2", "ind3"])
        self.assertIsNone(next_ind)


class TestSetupContext(unittest.TestCase):
    """Test SetupContext dataclass."""

    def test_setup_context_creation(self):
        """Test creating a setup context."""
        now = datetime.now()
        setup = SetupContext(
            setup_id=1,
            start_timestamp=now,
            start_candle_index=100
        )

        self.assertEqual(setup.setup_id, 1)
        self.assertEqual(setup.start_timestamp, now)
        self.assertEqual(setup.start_candle_index, 100)
        self.assertEqual(setup.direction, SignalDirection.NONE)
        self.assertEqual(len(setup.detections), 0)

    def test_add_detection(self):
        """Test adding detections to setup."""
        setup = SetupContext(
            setup_id=1,
            start_timestamp=datetime.now(),
            start_candle_index=100
        )

        # Add first detection
        detection1 = Detection(
            indicator_name="ind1",
            timestamp=datetime.now(),
            candle_index=101,
            price=1.1050,
            direction=SignalDirection.LONG
        )
        setup.add_detection(detection1)

        self.assertEqual(len(setup.detections), 1)
        self.assertIn("ind1", setup.detected_indicators)
        self.assertEqual(setup.direction, SignalDirection.LONG)
        self.assertEqual(setup.last_detection_candle, 101)

    def test_direction_consensus(self):
        """Test direction determination by consensus."""
        setup = SetupContext(1, datetime.now(), 100)

        # Add mixed signals
        setup.add_detection(Detection(
            "ind1", datetime.now(), 101, 1.1050, SignalDirection.LONG
        ))
        setup.add_detection(Detection(
            "ind2", datetime.now(), 102, 1.1051, SignalDirection.LONG
        ))
        setup.add_detection(Detection(
            "ind3", datetime.now(), 103, 1.1052, SignalDirection.SHORT
        ))

        # Long should win (2 vs 1)
        self.assertEqual(setup.direction, SignalDirection.LONG)

        # Add another short signal
        setup.add_detection(Detection(
            "ind4", datetime.now(), 104, 1.1053, SignalDirection.SHORT
        ))

        # Now it's tied (2 vs 2)
        setup._update_direction()
        self.assertEqual(setup.direction, SignalDirection.NONE)

    def test_has_indicator(self):
        """Test checking for specific indicators."""
        setup = SetupContext(1, datetime.now(), 100)

        self.assertFalse(setup.has_indicator("ind1"))

        setup.add_detection(Detection(
            "ind1", datetime.now(), 101, 1.1050, SignalDirection.LONG
        ))

        self.assertTrue(setup.has_indicator("ind1"))
        self.assertFalse(setup.has_indicator("ind2"))

    def test_reset(self):
        """Test resetting setup context."""
        setup = SetupContext(1, datetime.now(), 100)

        # Add some detections
        setup.add_detection(Detection(
            "ind1", datetime.now(), 101, 1.1050, SignalDirection.LONG
        ))
        setup.add_detection(Detection(
            "ind2", datetime.now(), 102, 1.1051, SignalDirection.LONG
        ))

        self.assertEqual(len(setup.detections), 2)
        self.assertEqual(setup.direction, SignalDirection.LONG)

        # Reset
        setup.reset()

        self.assertEqual(len(setup.detections), 0)
        self.assertEqual(len(setup.detected_indicators), 0)
        self.assertEqual(setup.direction, SignalDirection.NONE)
        self.assertIsNone(setup.last_detection_timestamp)
        self.assertIsNone(setup.last_detection_candle)


class TestTradeExecution(unittest.TestCase):
    """Test TradeExecution dataclass."""

    def test_trade_creation(self):
        """Test creating a trade execution."""
        now = datetime.now()
        detections = [
            Detection("ind1", now, 100, 1.1050, SignalDirection.LONG)
        ]

        trade = TradeExecution(
            entry_timestamp=now,
            entry_candle=100,
            entry_price=1.1000,
            direction=SignalDirection.LONG,
            stop_loss=1.0980,
            take_profit=1.1040,
            position_size=0.01,
            setup_detections=detections
        )

        self.assertEqual(trade.entry_price, 1.1000)
        self.assertEqual(trade.direction, SignalDirection.LONG)
        self.assertEqual(trade.stop_loss, 1.0980)
        self.assertEqual(trade.position_size, 0.01)
        self.assertIsNone(trade.exit_price)
        self.assertIsNone(trade.pnl_pips)

    def test_close_trade_with_profit(self):
        """Test closing a trade with profit."""
        trade = TradeExecution(
            entry_timestamp=datetime.now(),
            entry_candle=100,
            entry_price=1.1000,
            direction=SignalDirection.LONG,
            stop_loss=1.0980,
            take_profit=1.1040,
            position_size=0.01,
            setup_detections=[]
        )

        # Close with profit
        exit_time = datetime.now() + timedelta(hours=1)
        trade.close(
            exit_timestamp=exit_time,
            exit_candle=150,
            exit_price=1.1020,
            reason="take_profit"
        )

        self.assertEqual(trade.exit_price, 1.1020)
        self.assertEqual(trade.exit_reason, "take_profit")
        self.assertAlmostEqual(trade.pnl_pips, 20.0, places=5)
        self.assertAlmostEqual(trade.pnl_amount, 20.0 * 0.01, places=5)

    def test_close_trade_with_loss(self):
        """Test closing a trade with loss."""
        trade = TradeExecution(
            entry_timestamp=datetime.now(),
            entry_candle=100,
            entry_price=1.1000,
            direction=SignalDirection.LONG,
            stop_loss=1.0980,
            take_profit=1.1040,
            position_size=0.01,
            setup_detections=[]
        )

        # Close with loss
        trade.close(
            exit_timestamp=datetime.now(),
            exit_candle=150,
            exit_price=1.0980,
            reason="stop_loss"
        )

        self.assertAlmostEqual(trade.pnl_pips, -20.0, places=5)
        self.assertEqual(trade.exit_reason, "stop_loss")

    def test_short_trade_pnl(self):
        """Test P&L calculation for short trades."""
        trade = TradeExecution(
            entry_timestamp=datetime.now(),
            entry_candle=100,
            entry_price=1.1000,
            direction=SignalDirection.SHORT,
            stop_loss=1.1020,
            take_profit=1.0960,
            position_size=0.01,
            setup_detections=[]
        )

        # Close short with profit
        trade.close(
            exit_timestamp=datetime.now(),
            exit_candle=150,
            exit_price=1.0980,
            reason="take_profit"
        )

        self.assertAlmostEqual(trade.pnl_pips, 20.0, places=5)  # Profit on short


class TestBacktestState(unittest.TestCase):
    """Test BacktestState dataclass."""

    def test_backtest_state_creation(self):
        """Test creating backtest state."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2"],
            required_confirmations=2
        )

        state = BacktestState(strategy_config=config)

        self.assertEqual(state.current_state, ExecutionState.SCANNING)
        self.assertIsNone(state.active_setup)
        self.assertIsNone(state.active_trade)
        self.assertEqual(state.total_setups_started, 0)

    def test_can_start_new_setup(self):
        """Test checking if new setup can start."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2"],
            required_confirmations=2
        )
        state = BacktestState(strategy_config=config)

        # Initially can start
        self.assertTrue(state.can_start_new_setup())

        # Cannot start with active setup
        state.active_setup = SetupContext(1, datetime.now(), 100)
        self.assertFalse(state.can_start_new_setup())

        # Cannot start when not scanning
        state.active_setup = None
        state.current_state = ExecutionState.IN_POSITION
        self.assertFalse(state.can_start_new_setup())

        # Can start when back to scanning
        state.current_state = ExecutionState.SCANNING
        self.assertTrue(state.can_start_new_setup())

    def test_get_next_state(self):
        """Test state progression logic."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2"],
            required_confirmations=2
        )
        state = BacktestState(strategy_config=config)

        # Progress through states
        state.current_state = ExecutionState.SCANNING
        self.assertEqual(state.get_next_state(), ExecutionState.SIGNAL_1)

        state.current_state = ExecutionState.SIGNAL_1
        self.assertEqual(state.get_next_state(), ExecutionState.SIGNAL_2)

        state.current_state = ExecutionState.SIGNAL_2
        self.assertEqual(state.get_next_state(), ExecutionState.SIGNAL_3)

        state.current_state = ExecutionState.SIGNAL_5
        self.assertEqual(state.get_next_state(), ExecutionState.READY_TO_ENTER)

        # No progression from terminal states
        state.current_state = ExecutionState.READY_TO_ENTER
        self.assertEqual(state.get_next_state(), ExecutionState.READY_TO_ENTER)

    def test_check_timeout(self):
        """Test timeout checking."""
        config = StrategyConfig(
            indicator_sequence=["ind1", "ind2"],
            required_confirmations=2,
            timeouts={"ind1": 30, "default": 50}
        )
        state = BacktestState(strategy_config=config)

        # No timeout without setup
        self.assertFalse(state.check_timeout(100))

        # Create setup with detection
        state.active_setup = SetupContext(1, datetime.now(), 100)
        state.active_setup.add_detection(Detection(
            "ind1", datetime.now(), 100, 1.1050, SignalDirection.LONG
        ))

        # Within timeout
        self.assertFalse(state.check_timeout(120))  # 20 candles elapsed < 30

        # Exceeds timeout
        self.assertTrue(state.check_timeout(131))  # 31 candles elapsed > 30

    def test_record_transition(self):
        """Test recording state transitions."""
        config = StrategyConfig(
            indicator_sequence=["ind1"],
            required_confirmations=1
        )
        state = BacktestState(strategy_config=config)

        # Record a transition
        state.record_transition(
            from_state=ExecutionState.SCANNING,
            to_state=ExecutionState.SIGNAL_1,
            timestamp=datetime.now(),
            candle_index=100,
            trigger="ind1_detected"
        )

        self.assertEqual(len(state.state_history), 1)
        self.assertEqual(state.current_state, ExecutionState.SIGNAL_1)

        transition = state.state_history[0]
        self.assertEqual(transition.from_state, ExecutionState.SCANNING)
        self.assertEqual(transition.to_state, ExecutionState.SIGNAL_1)
        self.assertEqual(transition.trigger, "ind1_detected")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_complete_setup_flow(self):
        """Test a complete setup from detection to trade."""
        # Create strategy
        config = StrategyConfig(
            indicator_sequence=["momentum", "volume", "breakout"],
            required_confirmations=3,
            timeouts={"momentum": 30, "volume": 20, "default": 50}
        )

        # Initialize state
        state = BacktestState(strategy_config=config)
        now = datetime.now()

        # Start setup
        self.assertTrue(state.can_start_new_setup())
        state.active_setup = SetupContext(1, now, 100)
        state.total_setups_started += 1

        # Add first detection
        detection1 = Detection("momentum", now, 100, 1.1050, SignalDirection.LONG)
        state.active_setup.add_detection(detection1)
        state.current_state = ExecutionState.SIGNAL_1

        # Add second detection
        detection2 = Detection("volume", now, 105, 1.1055, SignalDirection.LONG)
        state.active_setup.add_detection(detection2)
        state.current_state = ExecutionState.SIGNAL_2

        # Add third detection
        detection3 = Detection("breakout", now, 110, 1.1060, SignalDirection.LONG)
        state.active_setup.add_detection(detection3)
        state.current_state = ExecutionState.SIGNAL_3

        # Check we have enough confirmations
        next_indicator = config.get_next_indicator(state.active_setup.detected_indicators)
        self.assertIsNone(next_indicator)

        # Ready to enter trade
        state.current_state = ExecutionState.READY_TO_ENTER
        self.assertEqual(state.active_setup.direction, SignalDirection.LONG)
        self.assertEqual(state.active_setup.get_detection_count(), 3)

        # Create trade
        trade = TradeExecution(
            entry_timestamp=now,
            entry_candle=111,
            entry_price=1.1065,
            direction=state.active_setup.direction,
            stop_loss=1.1045,
            take_profit=1.1105,
            position_size=0.01,
            setup_detections=list(state.active_setup.detections)
        )

        state.active_trade = trade
        state.current_state = ExecutionState.IN_POSITION
        state.active_setup = None  # Clear setup after trade entry

        # Close trade
        trade.close(now, 150, 1.1105, "take_profit")
        state.completed_trades.append(trade)
        state.active_trade = None
        state.current_state = ExecutionState.SCANNING

        # Verify final state
        self.assertTrue(state.can_start_new_setup())
        self.assertEqual(len(state.completed_trades), 1)
        self.assertAlmostEqual(state.completed_trades[0].pnl_pips, 40.0, places=5)


if __name__ == '__main__':
    unittest.main()