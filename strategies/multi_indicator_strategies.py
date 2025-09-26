"""
Pre-configured multi-indicator trading strategies.

This module contains various strategy configurations that combine
Liquidity Grab, CHoCH, and BOS indicators in different sequences and with
different confirmation requirements and risk parameters.
"""

from typing import Dict, Any
from core.state_types import StrategyConfig


class LiquidityGrabChochStrategy:
    """
    Primary strategy: Liquidity Grab followed by CHoCH confirmation.

    Logic: Look for liquidity grabs to identify potential reversal zones,
    then wait for CHoCH to confirm trend change before entering trade.

    Best for: Reversal trading at key levels
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector'],
            required_confirmations=2,  # Both must confirm
            timeouts={
                'liquidity_grab_detector': 30,  # 30 candles to find CHoCH after LG
                'choch_detector': 20,           # 20 candles for any additional confirmations
                'default': 25
            },
            stop_loss_pips=25,
            take_profit_pips=50,  # 1:2 R:R
            risk_percent=0.01
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
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


class ChochLiquidityGrabStrategy:
    """
    Alternative strategy: CHoCH followed by Liquidity Grab confirmation.

    Logic: Look for trend changes first via CHoCH, then wait for
    liquidity grabs to confirm the reversal is taking hold.

    Best for: Trend change confirmation trading
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['choch_detector', 'liquidity_grab_detector'],
            required_confirmations=2,  # Both must confirm
            timeouts={
                'choch_detector': 25,           # 25 candles to find LG after CHoCH
                'liquidity_grab_detector': 35, # 35 candles for LG confirmation
                'default': 30
            },
            stop_loss_pips=20,
            take_profit_pips=45,  # Slightly better R:R
            risk_percent=0.015    # Slightly higher risk for trend following
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'choch_detector': {
                'base_strength': 4,  # Slightly more sensitive for first signal
                'min_gap': 3
            },
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 2.5,  # Slightly less strict for confirmation
                'detect_same_session': True
            }
        }


class ConservativeStrategy:
    """
    Conservative approach with strict confirmations.

    Logic: Requires both indicators with tight parameters and
    lower risk per trade for steady returns.

    Best for: Risk-averse trading, smaller account sizes
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector'],
            required_confirmations=2,
            timeouts={
                'liquidity_grab_detector': 20,  # Shorter timeouts - less waiting
                'choch_detector': 15,
                'default': 18
            },
            stop_loss_pips=15,      # Tighter stops
            take_profit_pips=45,    # 1:3 R:R for better win rate requirement
            risk_percent=0.005      # Lower risk per trade
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 4.0,  # Stricter wick requirements
                'detect_same_session': True
            },
            'choch_detector': {
                'base_strength': 6,  # More conservative swing detection
                'min_gap': 4        # Larger gaps between swings
            }
        }


class AggressiveStrategy:
    """
    Aggressive approach with faster signals and higher risk.

    Logic: More sensitive indicators with wider stops and
    higher position sizing for maximum profit potential.

    Best for: Experienced traders, larger accounts
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['choch_detector', 'liquidity_grab_detector'],
            required_confirmations=2,
            timeouts={
                'choch_detector': 40,           # Longer timeouts - more patient
                'liquidity_grab_detector': 45,
                'default': 40
            },
            stop_loss_pips=35,      # Wider stops
            take_profit_pips=50,    # Lower R:R but faster signals
            risk_percent=0.02       # Higher risk per trade
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'choch_detector': {
                'base_strength': 3,  # More sensitive swing detection
                'min_gap': 2        # Smaller gaps for faster signals
            },
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 2.0,  # Less strict wick requirements
                'detect_same_session': True
            }
        }


class ScalpingStrategy:
    """
    High-frequency strategy for shorter timeframes.

    Logic: Fast signals with tight stops for quick profits.
    Best used on M15 or lower timeframes.

    Best for: Active trading, scalping opportunities
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector'],
            required_confirmations=2,
            timeouts={
                'liquidity_grab_detector': 8,   # Very short timeouts
                'choch_detector': 10,
                'default': 8
            },
            stop_loss_pips=8,       # Very tight stops
            take_profit_pips=16,    # 1:2 R:R
            risk_percent=0.005      # Lower risk for high frequency
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 1.5,  # Very sensitive for scalping
                'detect_same_session': True
            },
            'choch_detector': {
                'base_strength': 3,  # Fast swing detection
                'min_gap': 1        # Minimal gaps for speed
            }
        }


class SwingTradingStrategy:
    """
    Longer-term strategy for swing trading.

    Logic: Patient approach with wider stops and targets
    for capturing larger price moves.

    Best for: Part-time traders, swing trading on H4/D1
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['choch_detector', 'liquidity_grab_detector'],
            required_confirmations=2,
            timeouts={
                'choch_detector': 100,          # Very patient - wait for confirmation
                'liquidity_grab_detector': 80,
                'default': 90
            },
            stop_loss_pips=50,      # Wide stops for swing trading
            take_profit_pips=150,   # 1:3 R:R for larger moves
            risk_percent=0.015      # Moderate risk for longer holds
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'choch_detector': {
                'base_strength': 8,  # Very conservative for major swings
                'min_gap': 5        # Large gaps for significant structure
            },
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 5.0,  # Significant wicks only
                'detect_same_session': False     # Cross-session grabs more important
            }
        }


class BOSConfirmationStrategy:
    """
    Three-indicator strategy: Liquidity Grab → CHoCH → BOS confirmation.

    Logic: Start with liquidity grab, confirm trend change with CHoCH,
    then wait for BOS to confirm structural continuation in new trend.

    Best for: High-confidence trend continuation entries
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector', 'bos_detector'],
            required_confirmations=3,  # All three must confirm
            timeouts={
                'liquidity_grab_detector': 25,  # 25 candles to find CHoCH after LG
                'choch_detector': 20,           # 20 candles to find BOS after CHoCH
                'bos_detector': 30,             # 30 candles for BOS confirmation
                'default': 25
            },
            stop_loss_pips=20,
            take_profit_pips=60,  # 1:3 R:R for triple confirmation
            risk_percent=0.012    # Moderate risk for strong signals
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 3.0,
                'detect_same_session': True
            },
            'choch_detector': {
                'base_strength': 5,
                'min_gap': 3
            },
            'bos_detector': {
                'base_strength': 5,
                'min_gap': 3,
                'min_break_pips': 2.0,
                'direction_filter': 'both'
            }
        }


class TrendContinuationStrategy:
    """
    BOS-focused strategy: CHoCH → BOS → Liquidity Grab confirmation.

    Logic: Identify trend change with CHoCH, confirm continuation with BOS,
    then look for liquidity grabs in the new trend direction.

    Best for: Riding strong trend continuations
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['choch_detector', 'bos_detector', 'liquidity_grab_detector'],
            required_confirmations=3,
            timeouts={
                'choch_detector': 20,           # 20 candles to find BOS after CHoCH
                'bos_detector': 25,             # 25 candles to find LG after BOS
                'liquidity_grab_detector': 30, # 30 candles for final confirmation
                'default': 25
            },
            stop_loss_pips=30,
            take_profit_pips=75,  # 1:2.5 R:R for trend riding
            risk_percent=0.015
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'choch_detector': {
                'base_strength': 4,  # Slightly more sensitive for trend changes
                'min_gap': 3
            },
            'bos_detector': {
                'base_strength': 4,
                'min_gap': 3,
                'min_break_pips': 2.5,
                'direction_filter': 'both'
            },
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 2.5,
                'detect_same_session': True
            }
        }


class StructuralBreakStrategy:
    """
    BOS-primary strategy: BOS → CHoCH → Liquidity Grab sequence.

    Logic: Start with structural breaks (BOS), confirm with trend changes (CHoCH),
    finalize with liquidity validation.

    Best for: Structural breakout trading
    """

    @staticmethod
    def get_config() -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(
            indicator_sequence=['bos_detector', 'choch_detector', 'liquidity_grab_detector'],
            required_confirmations=3,
            timeouts={
                'bos_detector': 25,             # 25 candles to find CHoCH after BOS
                'choch_detector': 20,           # 20 candles to find LG after CHoCH
                'liquidity_grab_detector': 35, # 35 candles for validation
                'default': 25
            },
            stop_loss_pips=25,
            take_profit_pips=70,  # Strong R:R for structural plays
            risk_percent=0.01
        )

    @staticmethod
    def get_indicator_configs() -> Dict[str, Dict[str, Any]]:
        """Get indicator configurations for this strategy."""
        return {
            'bos_detector': {
                'base_strength': 5,
                'min_gap': 3,
                'min_break_pips': 3.0,
                'direction_filter': 'both'
            },
            'choch_detector': {
                'base_strength': 5,
                'min_gap': 3
            },
            'liquidity_grab_detector': {
                'enable_wick_extension_filter': True,
                'min_wick_extension_pips': 3.0,
                'detect_same_session': True
            }
        }


# Strategy registry for easy access
STRATEGIES = {
    'liquidity_grab_choch': LiquidityGrabChochStrategy,
    'choch_liquidity_grab': ChochLiquidityGrabStrategy,
    'conservative': ConservativeStrategy,
    'aggressive': AggressiveStrategy,
    'scalping': ScalpingStrategy,
    'swing_trading': SwingTradingStrategy,
    'bos_confirmation': BOSConfirmationStrategy,
    'trend_continuation': TrendContinuationStrategy,
    'structural_break': StructuralBreakStrategy
}


def get_strategy(strategy_name: str):
    """
    Get a strategy class by name.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")

    return STRATEGIES[strategy_name]


def list_strategies() -> Dict[str, str]:
    """
    Get a list of all available strategies with descriptions.

    Returns:
        Dictionary mapping strategy names to descriptions
    """
    return {
        'liquidity_grab_choch': 'Liquidity Grab → CHoCH confirmation (reversal trading)',
        'choch_liquidity_grab': 'CHoCH → Liquidity Grab confirmation (trend change)',
        'conservative': 'Strict confirmations, lower risk (risk-averse)',
        'aggressive': 'Faster signals, higher risk (experienced traders)',
        'scalping': 'High frequency, tight stops (M15 and below)',
        'swing_trading': 'Patient approach, wide stops (H4/D1 timeframes)',
        'bos_confirmation': 'LG → CHoCH → BOS triple confirmation (high confidence)',
        'trend_continuation': 'CHoCH → BOS → LG trend riding (strong continuations)',
        'structural_break': 'BOS → CHoCH → LG structural breakouts (breakout trading)'
    }