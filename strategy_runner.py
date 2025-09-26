"""
Strategy Runner - Bridge between state machine and indicators.

This module provides the main interface for running multi-indicator strategies
using the state machine framework. It handles data loading, indicator initialization,
and strategy execution with comprehensive logging and statistics.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from core.state_machine import StateMachine
from core.state_types import StrategyConfig, ActionType, ExecutionState
from core.data_loader import DataLoader
from indicators.liquidity_grab_detector import LiquidityGrabDetector
from indicators.choch_detector import ChochDetector
from indicators.bos_detector import BOSDetector


class StrategyRunner:
    """
    Main strategy execution engine.

    Orchestrates the entire trading simulation by:
    - Loading market data
    - Initializing indicators according to strategy config
    - Running the state machine on each candle
    - Collecting statistics and generating reports
    """

    def __init__(self, strategy_config: StrategyConfig, symbol: str = 'EURUSD'):
        """
        Initialize the strategy runner.

        Args:
            strategy_config: Strategy configuration defining indicator sequence
            symbol: Trading symbol to test on
        """
        self.strategy_config = strategy_config
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.indicators = {}
        self.state_machine = None
        self.data = None

        # Results tracking
        self.execution_log = []
        self.performance_stats = {}

        self.logger.info(f"Strategy Runner initialized for {symbol}")

    def setup_indicators(self, indicator_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Initialize indicators based on strategy configuration.

        Args:
            indicator_configs: Dictionary mapping indicator names to their configs
                Example: {
                    'liquidity_grab_detector': {'min_wick_extension_pips': 3.0},
                    'choch_detector': {'base_strength': 5, 'min_gap': 3}
                }
        """
        self.logger.info("Setting up indicators...")

        for indicator_name in self.strategy_config.indicator_sequence:
            config = indicator_configs.get(indicator_name, {})
            config['symbol'] = self.symbol  # Ensure symbol is set

            if indicator_name == 'liquidity_grab_detector':
                self.indicators[indicator_name] = LiquidityGrabDetector(
                    name=indicator_name,
                    config=config
                )
            elif indicator_name == 'choch_detector':
                self.indicators[indicator_name] = ChochDetector(
                    name=indicator_name,
                    config=config
                )
            elif indicator_name == 'bos_detector':
                self.indicators[indicator_name] = BOSDetector(
                    name=indicator_name,
                    config=config
                )
            else:
                raise ValueError(f"Unknown indicator: {indicator_name}")

            self.logger.info(f"Initialized {indicator_name} with config: {config}")

    def load_data(self, start_date: str, end_date: str, timeframe: str = 'H1') -> None:
        """
        Load market data for backtesting.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe to use (default: H1)
        """
        self.logger.info(f"Loading {self.symbol} data: {start_date} to {end_date} ({timeframe})")

        # Load data from demo_strategy_builder directory (using the same data)
        data_path = Path(__file__).parent.parent / 'demo_strategy_builder' / 'data' / f'{self.symbol}_20200101_20250809.csv'

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load and filter data
        self.data = self.data_loader.load_csv(data_path, symbol=self.symbol)
        self.data = self.data_loader.get_data(
            symbol=self.symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        # Reset index to have datetime as column for state machine
        self.data = self.data.reset_index()

        self.logger.info(f"Loaded {len(self.data)} candles")
        self.logger.info(f"Date range: {self.data['datetime'].iloc[0]} to {self.data['datetime'].iloc[-1]}")

    def initialize_state_machine(self) -> None:
        """Initialize the state machine with loaded indicators."""
        if not self.indicators:
            raise ValueError("No indicators initialized. Call setup_indicators() first.")

        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.state_machine = StateMachine(
            strategy_config=self.strategy_config,
            indicators=self.indicators,
            data_loader=self.data_loader
        )

        self.logger.info("State machine initialized")

    def run_strategy(self) -> Dict[str, Any]:
        """
        Execute the strategy on loaded data.

        Returns:
            Dictionary with execution results and statistics
        """
        if not self.state_machine:
            raise ValueError("State machine not initialized. Call initialize_state_machine() first.")

        self.logger.info("Starting strategy execution...")

        # Get maximum lookback period from all indicators
        max_lookback = max(
            indicator.get_lookback_period()
            for indicator in self.indicators.values()
        )

        total_candles = len(self.data)
        processed_candles = 0
        actions_taken = {action.value: 0 for action in ActionType}

        # Process each candle through the state machine
        for i in range(max_lookback, total_candles):
            # Get data window for current candle (includes lookback)
            window_data = self.data.iloc[max(0, i - max_lookback):i + 1].copy()
            current_candle = self.data.iloc[i]

            # Update BOS detector trend context if present
            if 'bos_detector' in self.indicators:
                trend_context = self._determine_trend_context(window_data)
                self.indicators['bos_detector'].set_trend_context(trend_context)

            # Process candle through state machine
            action = self.state_machine.process_candle(
                candle_data=window_data,
                candle_index=i,
                current_candle=current_candle
            )

            # Track actions
            actions_taken[action.value] += 1

            # Log significant actions
            if action != ActionType.WAIT:
                self.execution_log.append({
                    'candle_index': i,
                    'timestamp': current_candle['datetime'],
                    'action': action.value,
                    'state': self.state_machine.state.current_state.value,
                    'price': current_candle['close']
                })

                if action in [ActionType.INDICATOR_DETECTED, ActionType.ENTER_TRADE, ActionType.EXIT_TRADE]:
                    self.logger.info(
                        f"Candle {i}: {action.value} at {current_candle['close']:.5f} "
                        f"(State: {self.state_machine.state.current_state.value})"
                    )

            processed_candles += 1

            # Progress update every 100 candles
            if processed_candles % 100 == 0:
                self.logger.info(f"Processed {processed_candles}/{total_candles - max_lookback} candles")

        # Collect final statistics
        self.performance_stats = self.state_machine.get_statistics()
        self.performance_stats.update({
            'total_candles_processed': processed_candles,
            'actions_taken': actions_taken,
            'execution_log_entries': len(self.execution_log),
            'data_period': {
                'start': str(self.data['datetime'].iloc[0]),
                'end': str(self.data['datetime'].iloc[-1]),
                'symbol': self.symbol
            }
        })

        self.logger.info("Strategy execution completed")
        self._log_performance_summary()

        return self.performance_stats

    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive results from strategy execution.

        Returns:
            Dictionary containing all results, statistics, and execution logs
        """
        if not self.state_machine:
            return {}

        return {
            'performance_stats': self.performance_stats,
            'execution_log': self.execution_log,
            'completed_trades': [
                self._trade_to_dict(trade)
                for trade in self.state_machine.state.completed_trades
            ],
            'state_transitions': [
                self._transition_to_dict(transition)
                for transition in self.state_machine.state.state_history
            ],
            'final_state': {
                'current_state': self.state_machine.state.current_state.value,
                'active_setup': self.state_machine.state.active_setup is not None,
                'active_trade': self.state_machine.state.active_trade is not None
            }
        }

    def _trade_to_dict(self, trade) -> Dict[str, Any]:
        """Convert TradeExecution to dictionary."""
        return {
            'entry_timestamp': str(trade.entry_timestamp),
            'entry_candle': trade.entry_candle,
            'entry_price': trade.entry_price,
            'direction': trade.direction.value,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'position_size': trade.position_size,
            'exit_timestamp': str(trade.exit_timestamp) if trade.exit_timestamp else None,
            'exit_candle': trade.exit_candle,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'pnl_pips': trade.pnl_pips,
            'pnl_amount': trade.pnl_amount,
            'setup_indicators': [d.indicator_name for d in trade.setup_detections]
        }

    def _transition_to_dict(self, transition) -> Dict[str, Any]:
        """Convert StateTransition to dictionary."""
        return {
            'from_state': transition.from_state.value,
            'to_state': transition.to_state.value,
            'timestamp': str(transition.timestamp),
            'candle_index': transition.candle_index,
            'trigger': transition.trigger
        }

    def _log_performance_summary(self) -> None:
        """Log performance summary."""
        stats = self.performance_stats

        self.logger.info("=== STRATEGY PERFORMANCE SUMMARY ===")
        self.logger.info(f"Final State: {stats.get('current_state', 'UNKNOWN')}")
        self.logger.info(f"Total Setups Started: {stats.get('total_setups_started', 0)}")
        self.logger.info(f"Total Setups Completed: {stats.get('total_setups_completed', 0)}")
        self.logger.info(f"Total Setups Timeout: {stats.get('total_setups_timeout', 0)}")
        self.logger.info(f"Total Trades: {stats.get('total_trades', 0)}")
        self.logger.info(f"Winning Trades: {stats.get('winning_trades', 0)}")
        self.logger.info(f"Losing Trades: {stats.get('losing_trades', 0)}")
        self.logger.info(f"Win Rate: {stats.get('win_rate', 0):.1f}%")
        self.logger.info(f"Total Pips: {stats.get('total_pips', 0):.1f}")

        actions = stats.get('actions_taken', {})
        self.logger.info(f"Actions - Wait: {actions.get('wait', 0)}, "
                        f"Detected: {actions.get('detected', 0)}, "
                        f"Enter: {actions.get('enter_trade', 0)}, "
                        f"Exit: {actions.get('exit_trade', 0)}")

    def _determine_trend_context(self, data: pd.DataFrame) -> int:
        """
        Determine current trend context for BOS detector.

        Simple trend determination based on price movements and CHoCH signals.
        In production, this could be enhanced with additional trend indicators.

        Args:
            data: Recent price data window

        Returns:
            1 for bullish trend, -1 for bearish trend, 0 for no clear trend
        """
        if len(data) < 20:
            return 0  # Not enough data for trend determination

        # Method 1: Check if CHoCH has recently signaled a trend change
        if 'choch_detector' in self.indicators:
            # Use recent CHoCH signals to determine trend
            # Get data with datetime index for CHoCH analysis
            data_indexed = data.set_index('datetime') if 'datetime' in data.columns else data
            _, choch_signals = self.indicators['choch_detector'].get_all_swings_and_chochs(data_indexed)
            if choch_signals:
                # Get most recent CHoCH signal
                recent_choch = choch_signals[-1]
                trend_direction = recent_choch.get('trend_direction', 0)
                if trend_direction != 0:
                    return trend_direction

        # Method 2: Simple price trend analysis
        lookback_period = min(20, len(data) - 1)
        start_price = data.iloc[-lookback_period]['close']
        current_price = data.iloc[-1]['close']

        # Calculate percentage change
        price_change_pct = (current_price - start_price) / start_price

        # Threshold for trend determination (0.2% for EURUSD)
        trend_threshold = 0.002

        if price_change_pct > trend_threshold:
            return 1  # Bullish trend
        elif price_change_pct < -trend_threshold:
            return -1  # Bearish trend
        else:
            return 0  # No clear trend


def main():
    """Example usage of StrategyRunner."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Define strategy configuration
        strategy_config = StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector'],
            required_confirmations=2,  # Both indicators must confirm
            timeouts={'default': 50},  # 50 candles timeout
            stop_loss_pips=20,
            take_profit_pips=40,
            risk_percent=0.01
        )

        # Define indicator configurations
        indicator_configs = {
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 3.0,
                'detect_same_session': True
            },
            'choch_detector': {
                'base_strength': 5,
                'min_gap': 3
            }
        }

        # Initialize strategy runner
        runner = StrategyRunner(strategy_config, symbol='EURUSD')

        # Setup and run
        runner.setup_indicators(indicator_configs)
        runner.load_data('2024-01-01', '2024-02-01', 'H1')
        runner.initialize_state_machine()

        # Execute strategy
        print("🚀 Starting Multi-Indicator Strategy Test")
        print("=" * 60)

        performance = runner.run_strategy()
        results = runner.get_results()

        print("\n" + "=" * 60)
        print("📊 EXECUTION COMPLETED")

        # Display key results
        trades = results.get('completed_trades', [])
        if trades:
            print(f"\n🎯 Completed Trades ({len(trades)}):")
            for i, trade in enumerate(trades, 1):
                print(f"  {i}. {trade['direction']} @ {trade['entry_price']:.5f} "
                      f"→ {trade['exit_price']:.5f} = {trade['pnl_pips']:.1f} pips "
                      f"({trade['exit_reason']})")

        transitions = results.get('state_transitions', [])
        key_transitions = [t for t in transitions if 'detected' in t['trigger'] or 'trade' in t['trigger']]
        if key_transitions:
            print(f"\n🔄 Key State Transitions ({len(key_transitions)}):")
            for transition in key_transitions[-10:]:  # Last 10
                print(f"  {transition['from_state']} → {transition['to_state']} "
                      f"({transition['trigger']})")

        print(f"\n✅ Strategy test completed successfully!")
        print(f"Final Summary: {performance.get('total_trades', 0)} trades, "
              f"{performance.get('win_rate', 0):.1f}% win rate, "
              f"{performance.get('total_pips', 0):.1f} total pips")

    except Exception as e:
        print(f"\n❌ Error during strategy execution: {e}")
        raise


if __name__ == "__main__":
    main()