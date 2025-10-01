"""
Pydantic models for API request/response validation.

Defines the data models for JSON strategy configurations and API responses.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, List, Optional
from datetime import datetime


class IndicatorConfig(BaseModel):
    """Configuration for a single indicator."""
    type: str = Field(..., description="Indicator type (e.g., 'liquidity_grab_detector')")
    sequence: int = Field(..., description="Execution sequence within timeframe")
    config: Dict[str, Any] = Field(default_factory=dict, description="Indicator-specific configuration")


class TimeframeBlock(BaseModel):
    """Configuration for a single timeframe block."""
    timeframe: str = Field(..., description="Timeframe (e.g., 'H4', 'H1')")
    sequence: int = Field(..., description="Execution sequence among timeframes")
    indicators: List[IndicatorConfig] = Field(..., description="List of indicators for this timeframe")


class SymbolBlock(BaseModel):
    """Symbol selection configuration."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD')")


class RiskManagementBlock(BaseModel):
    """Risk management configuration."""
    stop_loss_pips: Optional[float] = Field(None, description="Stop loss in pips")
    take_profit_pips: Optional[float] = Field(None, description="Take profit in pips")
    position_size_percent: Optional[float] = Field(None, description="Position size as percentage of account")


class EntryBlock(BaseModel):
    """Entry conditions configuration."""
    conditions: str = Field(..., description="Entry conditions logic")


class StrategyConfiguration(BaseModel):
    """Complete strategy configuration."""
    name: str = Field(..., description="Strategy name")
    version: str = Field(default="1.0", description="Strategy version")
    symbol_block: SymbolBlock = Field(..., description="Symbol selection")
    timeframe_blocks: List[TimeframeBlock] = Field(..., description="Timeframe configurations")
    risk_management_block: Optional[RiskManagementBlock] = Field(None, description="Risk management")
    entry_block: EntryBlock = Field(..., description="Entry conditions")


class StrategyRequest(BaseModel):
    """Request model for strategy processing."""
    strategy: StrategyConfiguration = Field(..., description="Strategy configuration")


class PerformanceSummary(BaseModel):
    """Performance summary of strategy execution."""
    total_trades: int = Field(0, description="Total number of trades")
    winning_trades: int = Field(0, description="Number of winning trades")
    losing_trades: int = Field(0, description="Number of losing trades")
    win_rate: float = Field(0.0, description="Win rate percentage")
    total_return: float = Field(0.0, description="Total return percentage")
    max_drawdown: float = Field(0.0, description="Maximum drawdown percentage")


class StrategyResults(BaseModel):
    """Results of strategy execution."""
    total_trades: int = Field(..., description="Total number of trades executed")
    indicators_used: List[str] = Field(..., description="List of indicators used")
    performance_summary: PerformanceSummary = Field(..., description="Performance metrics")
    raw_results: Dict[str, Any] = Field(..., description="Raw execution results")


class StrategyCreateResponse(BaseModel):
    """Response model for strategy creation."""
    id: int = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: str = Field(default="saved", description="Strategy status")
    message: str = Field(default="Strategy saved successfully", description="Response message")


class StrategyBacktestRequest(BaseModel):
    """Request model for strategy backtesting."""
    start_date: Optional[str] = Field(None, description="Backtest start date")
    end_date: Optional[str] = Field(None, description="Backtest end date")
    initial_capital: Optional[float] = Field(10000.0, description="Initial capital for backtest")


class StrategyBacktestResponse(BaseModel):
    """Response model for strategy backtesting."""
    strategy_id: int = Field(..., description="Strategy ID")
    strategy_name: str = Field(..., description="Name of executed strategy")
    execution_time: datetime = Field(..., description="Execution timestamp")
    results: StrategyResults = Field(..., description="Execution results")


class StrategyResponse(BaseModel):
    """Response model for strategy processing."""
    status: str = Field(..., description="Execution status")
    strategy_name: str = Field(..., description="Name of executed strategy")
    execution_time: datetime = Field(..., description="Execution timestamp")
    results: StrategyResults = Field(..., description="Execution results")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    validation_errors: Optional[List[str]] = Field(None, description="Validation errors")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    components: Dict[str, str] = Field(..., description="Component status")
    timestamp: datetime = Field(..., description="Check timestamp")


class ExampleStrategyResponse(BaseModel):
    """Response model for example strategy."""
    strategy: StrategyConfiguration = Field(..., description="Example strategy configuration")


# Authentication Models

class UserRegisterRequest(BaseModel):
    """Request model for user registration."""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password")


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class UserResponse(BaseModel):
    """Response model for user information."""
    id: int = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    is_active: bool = Field(..., description="User active status")
    created_at: datetime = Field(..., description="Account creation timestamp")


class AuthResponse(BaseModel):
    """Response model for authentication (login/register)."""
    user: UserResponse = Field(..., description="User information")
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str = Field(..., description="Refresh token")


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str = Field(..., description="Response message")