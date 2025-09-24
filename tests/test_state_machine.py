"""
Comprehensive tests for the state machine.

Tests all state transitions, indicator sequencing, trade execution,
and error handling scenarios.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_machine import StateMachine
from core.state_types import (
    ExecutionState,
    ActionType,
    SignalDirection,
    Detection,
    StrategyConfig,
    BacktestState
)
from core.data_loader import DataLoader
from indicators.base_indicator import BaseIndicator


class MockIndicator(BaseIndicator):
    """Mock indicator for testing."""

    def __init__(self, name: str, should_detect: bool = False,
                 detection_direction: SignalDirection = SignalDirection.LONG):
        super().__init__(name)
        self.should_detect = should_detect
        self.detection_direction = detection_direction
        self.check_count = 0

    def check(self, data: pd.DataFrame, candle_index: int) -> Optional[Detection]:
        """Mock check that can be controlled."""
        self.check_count += 1

        if self.should_detect:
            current = data.iloc[-1]
            return Detection(
                indicator_name=self.name,
                timestamp=current['datetime'],
                candle_index=candle_index,
                price=current['close'],
                direction=self.detection_direction,
                metadata={'mock': True}
            )
        return None

    def get_lookback_period(self) -> int:
        """Return minimal lookback."""
        return 10

    def reset(self) -> None:
        """Reset the indicator."""
        super().reset()
        self.check_count = 0


class TestStateMachine(unittest.TestCase):
    """Test the state machine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = self._create_sample_data(100)

        # Create data loader
        self.data_loader = DataLoader(cache_enabled=False)

        # Create strategy config
        self.strategy_config = StrategyConfig(
            indicator_sequence=['indicator1', 'indicator2', 'indicator3'],
            required_confirmations=2,
            timeouts={'default': 10},
            stop_loss_pips=20,
            take_profit_pips=40,
            risk_percent=0.01
        )

        # Create mock indicators
        self.indicators = {
            'indicator1': MockIndicator('indicator1'),
            'indicator2': MockIndicator('indicator2'),
            'indicator3': MockIndicator('indicator3')
        }

    def _create_sample_data(self, num_candles: int) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=num_candles, freq='1h')

        np.random.seed(42)
        close_prices = 1.1000 + np.random.randn(num_candles) * 0.001

        return pd.DataFrame({
            'datetime': dates,
            'open': close_prices + np.random.randn(num_candles) * 0.0001,
            'high': close_prices + abs(np.random.randn(num_candles) * 0.0003),
            'low': close_prices - abs(np.random.randn(num_candles) * 0.0003),
            'close': close_prices,
            'volume': np.random.randint(1000, 5000, num_candles)
        })

    def test_initialization(self):
        """Test state machine initialization."""
        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        self.assertEqual(sm.state.current_state, ExecutionState.SCANNING)
        self.assertIsNone(sm.state.active_setup)
        self.assertIsNone(sm.state.active_trade)
        self.assertEqual(len(sm.state.completed_trades), 0)

    def test_initialization_missing_indicators(self):
        """Test initialization with missing indicators."""
        bad_config = StrategyConfig(
            indicator_sequence=['indicator1', 'missing_indicator'],
            required_confirmations=2
        )

        with self.assertRaises(ValueError) as context:
            StateMachine(bad_config, self.indicators, self.data_loader)

        self.assertIn("Missing indicators", str(context.exception))

    def test_scanning_no_detection(self):
        """Test SCANNING state with no detections."""
        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Process candle - no indicators should detect
        action = sm.process_candle(
            self.sample_data.head(20),
            0,
            self.sample_data.iloc[19]
        )

        self.assertEqual(action, ActionType.WAIT)
        self.assertEqual(sm.state.current_state, ExecutionState.SCANNING)
        self.assertIsNone(sm.state.active_setup)

    def test_scanning_with_detection(self):
        """Test SCANNING state with detection."""
        # Make indicator1 detect
        self.indicators['indicator1'].should_detect = True

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        action = sm.process_candle(
            self.sample_data.head(20),
            0,
            self.sample_data.iloc[19]
        )

        self.assertEqual(action, ActionType.INDICATOR_DETECTED)
        self.assertEqual(sm.state.current_state, ExecutionState.SIGNAL_1)
        self.assertIsNotNone(sm.state.active_setup)
        self.assertEqual(len(sm.state.active_setup.detections), 1)
        self.assertEqual(sm.state.total_setups_started, 1)

    def test_signal_state_next_detection(self):
        """Test SIGNAL state detecting next indicator."""
        # Setup: indicator1 detects first, then indicator2
        self.indicators['indicator1'].should_detect = True

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # First detection (SCANNING -> SIGNAL_1)
        sm.process_candle(self.sample_data.head(20), 0, self.sample_data.iloc[19])

        # Now make indicator2 detect
        self.indicators['indicator2'].should_detect = True
        self.indicators['indicator1'].should_detect = False

        # Second detection (SIGNAL_1 -> SIGNAL_2)
        action = sm.process_candle(
            self.sample_data.head(21),
            1,
            self.sample_data.iloc[20]
        )

        self.assertEqual(action, ActionType.INDICATOR_DETECTED)
        # Should move to READY_TO_ENTER since we need 2 confirmations
        self.assertEqual(sm.state.current_state, ExecutionState.READY_TO_ENTER)
        self.assertEqual(len(sm.state.active_setup.detections), 2)

    def test_signal_state_timeout(self):
        """Test timeout in SIGNAL state."""
        self.indicators['indicator1'].should_detect = True

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # First detection
        sm.process_candle(self.sample_data.head(20), 0, self.sample_data.iloc[19])
        self.assertEqual(sm.state.current_state, ExecutionState.SIGNAL_1)

        # Process many candles without detection (exceed timeout of 10)
        for i in range(15):
            action = sm.process_candle(
                self.sample_data.head(21 + i),
                1 + i,
                self.sample_data.iloc[20 + i]
            )

            if i < 10:
                self.assertEqual(action, ActionType.WAIT)
            else:
                # Should timeout after 10 candles
                self.assertEqual(action, ActionType.TIMEOUT)
                self.assertEqual(sm.state.current_state, ExecutionState.SCANNING)
                self.assertIsNone(sm.state.active_setup)
                self.assertEqual(sm.state.total_setups_timeout, 1)
                break

    def test_ready_to_enter_trade_execution(self):
        """Test trade execution from READY_TO_ENTER state."""
        # Setup to get to READY_TO_ENTER
        self.indicators['indicator1'].should_detect = True
        self.indicators['indicator1'].detection_direction = SignalDirection.LONG

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # First detection
        sm.process_candle(self.sample_data.head(20), 0, self.sample_data.iloc[19])

        # Second detection
        self.indicators['indicator2'].should_detect = True
        self.indicators['indicator2'].detection_direction = SignalDirection.LONG
        sm.process_candle(self.sample_data.head(21), 1, self.sample_data.iloc[20])

        self.assertEqual(sm.state.current_state, ExecutionState.READY_TO_ENTER)

        # Process next candle - should enter trade
        action = sm.process_candle(
            self.sample_data.head(22),
            2,
            self.sample_data.iloc[21]
        )

        self.assertEqual(action, ActionType.ENTER_TRADE)
        self.assertEqual(sm.state.current_state, ExecutionState.IN_POSITION)
        self.assertIsNotNone(sm.state.active_trade)

        trade = sm.state.active_trade
        self.assertEqual(trade.direction, SignalDirection.LONG)
        self.assertAlmostEqual(
            trade.stop_loss,
            trade.entry_price - 0.0020,
            places=4
        )
        self.assertAlmostEqual(
            trade.take_profit,
            trade.entry_price + 0.0040,
            places=4
        )

    def test_in_position_stop_loss(self):
        """Test stop loss exit."""
        # Get to IN_POSITION state
        self._setup_to_in_position()

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Execute setup to trade
        for i in range(3):
            sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

        self.assertEqual(sm.state.current_state, ExecutionState.IN_POSITION)
        trade = sm.state.active_trade

        # Create candle that hits stop loss
        sl_candle = self.sample_data.iloc[22].copy()
        sl_candle['low'] = trade.stop_loss - 0.0001  # Below stop loss

        sl_data = self.sample_data.head(23).copy()
        sl_data.iloc[-1] = sl_candle

        action = sm.process_candle(sl_data, 3, sl_candle)

        self.assertEqual(action, ActionType.EXIT_TRADE)
        self.assertEqual(sm.state.current_state, ExecutionState.POSITION_CLOSED)
        self.assertIsNone(sm.state.active_trade)
        self.assertEqual(len(sm.state.completed_trades), 1)

        closed_trade = sm.state.completed_trades[0]
        self.assertEqual(closed_trade.exit_reason, "stop_loss")
        self.assertAlmostEqual(closed_trade.pnl_pips, -20, places=0)

    def test_in_position_take_profit(self):
        """Test take profit exit."""
        # Get to IN_POSITION state
        self._setup_to_in_position()

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Execute setup to trade
        for i in range(3):
            sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

        self.assertEqual(sm.state.current_state, ExecutionState.IN_POSITION)
        trade = sm.state.active_trade

        # Create candle that hits take profit
        tp_candle = self.sample_data.iloc[22].copy()
        tp_candle['high'] = trade.take_profit + 0.0001  # Above take profit

        tp_data = self.sample_data.head(23).copy()
        tp_data.iloc[-1] = tp_candle

        action = sm.process_candle(tp_data, 3, tp_candle)

        self.assertEqual(action, ActionType.EXIT_TRADE)
        self.assertEqual(sm.state.current_state, ExecutionState.POSITION_CLOSED)
        self.assertEqual(len(sm.state.completed_trades), 1)

        closed_trade = sm.state.completed_trades[0]
        self.assertEqual(closed_trade.exit_reason, "take_profit")
        self.assertAlmostEqual(closed_trade.pnl_pips, 40, places=0)

    def test_position_closed_to_scanning(self):
        """Test immediate transition from POSITION_CLOSED to SCANNING."""
        # Get to POSITION_CLOSED
        self._setup_to_in_position()

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Execute to trade and close
        for i in range(3):
            sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

        # Hit stop loss
        sl_candle = self.sample_data.iloc[22].copy()
        sl_candle['low'] = sm.state.active_trade.stop_loss - 0.0001

        sl_data = self.sample_data.head(23).copy()
        sl_data.iloc[-1] = sl_candle

        sm.process_candle(sl_data, 3, sl_candle)
        self.assertEqual(sm.state.current_state, ExecutionState.POSITION_CLOSED)

        # Next candle should reset to SCANNING
        action = sm.process_candle(
            self.sample_data.head(24),
            4,
            self.sample_data.iloc[23]
        )

        self.assertEqual(action, ActionType.RESET)
        self.assertEqual(sm.state.current_state, ExecutionState.SCANNING)

    def test_get_statistics(self):
        """Test statistics calculation."""
        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Run a few trades
        self._run_sample_trades(sm)

        stats = sm.get_statistics()

        self.assertIn('current_state', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('winning_trades', stats)
        self.assertIn('losing_trades', stats)
        self.assertIn('win_rate', stats)
        self.assertIn('total_pips', stats)

    def test_indicator_reset_on_timeout(self):
        """Test that indicators are reset when setup times out."""
        self.indicators['indicator1'].should_detect = True

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # First detection
        sm.process_candle(self.sample_data.head(20), 0, self.sample_data.iloc[19])
        self.assertEqual(sm.state.current_state, ExecutionState.SIGNAL_1)

        # Store initial internal state to verify reset
        self.indicators['indicator1'].internal_state['test_key'] = 'test_value'

        # Process candles until timeout (11th candle should trigger timeout since detection was at 0)
        timeout_triggered = False
        for i in range(1, 15):
            action = sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

            if action == ActionType.TIMEOUT:
                timeout_triggered = True
                # Verify we're back in SCANNING
                self.assertEqual(sm.state.current_state, ExecutionState.SCANNING)
                # Verify setup was cleared
                self.assertIsNone(sm.state.active_setup)
                # Verify indicator was reset (internal state cleared)
                self.assertEqual(self.indicators['indicator1'].internal_state, {})
                break

        self.assertTrue(timeout_triggered, "Timeout should have been triggered")

    def test_short_trade_execution(self):
        """Test SHORT trade execution and exit."""
        # Setup for SHORT trade
        self.indicators['indicator1'].should_detect = True
        self.indicators['indicator1'].detection_direction = SignalDirection.SHORT
        self.indicators['indicator2'].should_detect = True
        self.indicators['indicator2'].detection_direction = SignalDirection.SHORT

        sm = StateMachine(self.strategy_config, self.indicators, self.data_loader)

        # Get to IN_POSITION with SHORT
        for i in range(3):
            sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

        self.assertEqual(sm.state.current_state, ExecutionState.IN_POSITION)
        trade = sm.state.active_trade
        self.assertEqual(trade.direction, SignalDirection.SHORT)

        # For SHORT: stop loss is above entry, take profit is below
        self.assertGreater(trade.stop_loss, trade.entry_price)
        self.assertLess(trade.take_profit, trade.entry_price)

    def _setup_to_in_position(self):
        """Helper to set indicators for reaching IN_POSITION."""
        self.indicators['indicator1'].should_detect = True
        self.indicators['indicator1'].detection_direction = SignalDirection.LONG
        self.indicators['indicator2'].should_detect = True
        self.indicators['indicator2'].detection_direction = SignalDirection.LONG

    def _run_sample_trades(self, sm):
        """Helper to run some sample trades for statistics."""
        # Run a winning trade
        self._setup_to_in_position()

        for i in range(3):
            sm.process_candle(
                self.sample_data.head(20 + i),
                i,
                self.sample_data.iloc[19 + i]
            )

        # Exit with profit
        tp_candle = self.sample_data.iloc[22].copy()
        tp_candle['high'] = sm.state.active_trade.take_profit + 0.0001

        tp_data = self.sample_data.head(23).copy()
        tp_data.iloc[-1] = tp_candle

        sm.process_candle(tp_data, 3, tp_candle)

        # Reset and process next candle to return to SCANNING
        sm.process_candle(self.sample_data.head(24), 4, self.sample_data.iloc[23])


if __name__ == '__main__':
    unittest.main()