"""
Indicators module for the Strategy Builder system.

This module contains all indicator implementations used for detecting
trading signals and patterns.
"""

from .base_indicator import BaseIndicator
from .liquidity_grab_detector import LiquidityGrabDetector
from .choch_detector import ChochDetector

# Aliases for backward compatibility
LiquidityGrabIndicator = LiquidityGrabDetector
ChangeOfCharacterIndicator = ChochDetector

__all__ = [
    'BaseIndicator',
    'LiquidityGrabDetector',
    'ChochDetector',
    'LiquidityGrabIndicator',  # Backward compatibility alias
    'ChangeOfCharacterIndicator',  # Backward compatibility alias
]