"""
Core module for Strategy Builder system.

This module contains the fundamental components for data loading,
state management, and execution context.
"""

from .data_loader import DataLoader
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
from .state_machine import StateMachine

__all__ = [
    'DataLoader',
    'StateMachine',
    'ExecutionState',
    'ActionType',
    'SignalDirection',
    'Detection',
    'StrategyConfig',
    'SetupContext',
    'StateTransition',
    'TradeExecution',
    'BacktestState'
]