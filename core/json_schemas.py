"""
JSON schema definitions for multi-timeframe strategy configuration.

This module defines the JSON schemas used for configuring multi-timeframe
trading strategies, including indicator configurations, risk parameters,
and execution rules.
"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum


class TimeframeType(str, Enum):
    """Supported timeframe types."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


class IndicatorType(str, Enum):
    """Supported indicator types."""
    LIQUIDITY_GRAB = "liquidity_grab_detector"
    CHOCH = "choch_detector"
    BOS = "bos_detector"
    FVG = "fvg_detector"
    ORDER_BLOCK = "order_block_detector"


class SignalDirection(str, Enum):
    """Signal direction types."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class ExecutionMode(str, Enum):
    """Strategy execution modes."""
    SEQUENTIAL = "sequential"     # Indicators execute in strict sequence
    PARALLEL = "parallel"        # Indicators can execute simultaneously
    CONDITIONAL = "conditional"  # Indicators execute based on conditions


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator."""
    type: IndicatorType
    timeframe: TimeframeType
    params: Dict[str, Any]
    required: bool = True
    timeout: Optional[int] = None
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class RiskConfig:
    """Risk management configuration."""
    position_sizing: Dict[str, Any]  # {"method": "percent", "value": 0.01}
    stop_loss: Dict[str, Any]        # {"pips": 25, "method": "fixed"}
    take_profit: Dict[str, Any]      # {"pips": 50, "method": "fixed"}
    max_drawdown: Optional[float] = None
    max_concurrent_trades: int = 1


@dataclass
class TimingConfig:
    """Timing and timeout configuration."""
    setup_timeout: int = 30          # Max candles to wait for complete setup
    signal_timeout: int = 20         # Max candles to wait for next signal
    position_timeout: Optional[int] = None  # Max candles to hold position
    trading_sessions: Optional[List[Dict]] = None  # Active trading sessions


# JSON Schema Definitions
INDICATOR_SCHEMA = {
    "type": "object",
    "required": ["type", "timeframe", "params"],
    "properties": {
        "type": {
            "type": "string",
            "enum": [t.value for t in IndicatorType]
        },
        "timeframe": {
            "type": "string",
            "enum": [t.value for t in TimeframeType]
        },
        "params": {
            "type": "object",
            "additionalProperties": True
        },
        "required": {
            "type": "boolean",
            "default": True
        },
        "timeout": {
            "type": ["integer", "null"],
            "minimum": 1
        },
        "conditions": {
            "type": ["object", "null"],
            "additionalProperties": True
        }
    }
}

RISK_SCHEMA = {
    "type": "object",
    "required": ["position_sizing", "stop_loss", "take_profit"],
    "properties": {
        "position_sizing": {
            "type": "object",
            "required": ["method", "value"],
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["percent", "fixed", "kelly", "martingale"]
                },
                "value": {
                    "type": "number",
                    "minimum": 0
                }
            }
        },
        "stop_loss": {
            "type": "object",
            "required": ["pips", "method"],
            "properties": {
                "pips": {
                    "type": "number",
                    "minimum": 0
                },
                "method": {
                    "type": "string",
                    "enum": ["fixed", "atr", "swing", "percentage"]
                }
            }
        },
        "take_profit": {
            "type": "object",
            "required": ["pips", "method"],
            "properties": {
                "pips": {
                    "type": "number",
                    "minimum": 0
                },
                "method": {
                    "type": "string",
                    "enum": ["fixed", "atr", "swing", "percentage", "ratio"]
                }
            }
        },
        "max_drawdown": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1
        },
        "max_concurrent_trades": {
            "type": "integer",
            "minimum": 1,
            "default": 1
        }
    }
}

TIMING_SCHEMA = {
    "type": "object",
    "properties": {
        "setup_timeout": {
            "type": "integer",
            "minimum": 1,
            "default": 30
        },
        "signal_timeout": {
            "type": "integer",
            "minimum": 1,
            "default": 20
        },
        "position_timeout": {
            "type": ["integer", "null"],
            "minimum": 1
        },
        "trading_sessions": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["start", "end"],
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "timezone": {"type": "string", "default": "UTC"}
                }
            }
        }
    }
}

STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["name", "version", "indicators", "risk", "execution"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+$"
        },
        "description": {
            "type": "string"
        },
        "indicators": {
            "type": "array",
            "minItems": 1,
            "items": INDICATOR_SCHEMA
        },
        "risk": RISK_SCHEMA,
        "timing": TIMING_SCHEMA,
        "execution": {
            "type": "object",
            "required": ["mode", "primary_timeframe"],
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [m.value for m in ExecutionMode]
                },
                "primary_timeframe": {
                    "type": "string",
                    "enum": [t.value for t in TimeframeType]
                },
                "confirmation_required": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "direction_filter": {
                    "type": "string",
                    "enum": [d.value for d in SignalDirection],
                    "default": "both"
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "created_by": {"type": "string"},
                "created_at": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "backtest_results": {"type": "object"}
            }
        }
    }
}


def get_example_strategy() -> Dict[str, Any]:
    """
    Get an example multi-timeframe strategy configuration.

    Returns:
        Dictionary containing a complete example strategy
    """
    return {
        "name": "LiquidityGrab-CHoCH Multi-Timeframe",
        "version": "1.0.0",
        "description": "Multi-timeframe strategy using Liquidity Grab on H1 and CHoCH confirmation on H4",
        "indicators": [
            {
                "type": "liquidity_grab_detector",
                "timeframe": "H1",
                "params": {
                    "enable_wick_extension_filter": True,
                    "min_wick_extension_pips": 3.0,
                    "detect_same_session": True,
                    "lookback_period": 48
                },
                "required": True,
                "timeout": 30,
                "conditions": {
                    "must_be_first": True
                }
            },
            {
                "type": "choch_detector",
                "timeframe": "H4",
                "params": {
                    "base_strength": 5,
                    "min_gap": 3,
                    "lookback_period": 20
                },
                "required": True,
                "timeout": 10,
                "conditions": {
                    "requires": ["liquidity_grab_detector"]
                }
            }
        ],
        "risk": {
            "position_sizing": {
                "method": "percent",
                "value": 0.01
            },
            "stop_loss": {
                "pips": 25,
                "method": "fixed"
            },
            "take_profit": {
                "pips": 50,
                "method": "fixed"
            },
            "max_drawdown": 0.15,
            "max_concurrent_trades": 1
        },
        "timing": {
            "setup_timeout": 40,
            "signal_timeout": 25,
            "position_timeout": 100,
            "trading_sessions": [
                {
                    "start": "08:00",
                    "end": "17:00",
                    "timezone": "UTC"
                }
            ]
        },
        "execution": {
            "mode": "sequential",
            "primary_timeframe": "H1",
            "confirmation_required": 2,
            "direction_filter": "both"
        },
        "metadata": {
            "created_by": "Strategy Builder",
            "created_at": "2024-01-01T00:00:00Z",
            "tags": ["multi-timeframe", "liquidity-grab", "choch", "reversal"],
            "backtest_results": {}
        }
    }


def get_single_timeframe_example() -> Dict[str, Any]:
    """
    Get an example single-timeframe strategy for backward compatibility.

    Returns:
        Dictionary containing a single-timeframe strategy
    """
    return {
        "name": "Single Timeframe CHoCH Strategy",
        "version": "1.0.0",
        "description": "Simple CHoCH detection strategy on H1 timeframe",
        "indicators": [
            {
                "type": "choch_detector",
                "timeframe": "H1",
                "params": {
                    "base_strength": 4,
                    "min_gap": 2,
                    "lookback_period": 20
                },
                "required": True,
                "timeout": 20
            }
        ],
        "risk": {
            "position_sizing": {
                "method": "percent",
                "value": 0.005
            },
            "stop_loss": {
                "pips": 15,
                "method": "fixed"
            },
            "take_profit": {
                "pips": 30,
                "method": "fixed"
            },
            "max_concurrent_trades": 1
        },
        "timing": {
            "setup_timeout": 20,
            "signal_timeout": 15
        },
        "execution": {
            "mode": "sequential",
            "primary_timeframe": "H1",
            "confirmation_required": 1,
            "direction_filter": "both"
        },
        "metadata": {
            "created_by": "Strategy Builder",
            "created_at": "2024-01-01T00:00:00Z",
            "tags": ["single-timeframe", "choch", "simple"]
        }
    }