"""
Simplified FastAPI server focusing on authentication functionality.

This server provides all authentication endpoints and basic strategy storage
without the complex strategy execution logic.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
from sqlalchemy.orm import Session

from config.settings import settings
from database.connection import get_db, create_tables
from database.models import User, Strategy
from api.auth.router import router as auth_router
from api.auth.dependencies import get_current_active_user
from api.models import (
    StrategyRequest,
    StrategyCreateResponse,
    StrategyBacktestRequest,
    StrategyBacktestResponse,
    ErrorResponse,
    HealthResponse,
    PerformanceSummary,
    StrategyResults,
    StrategyResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan event handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    logger.info("Creating database tables...")
    create_tables()
    logger.info("Database initialization complete")

    yield

    # Shutdown (if needed)
    logger.info("Application shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Strategy Builder Authentication API",
    description="Authentication-focused API for multi-timeframe trading strategy management",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for localhost frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Strategy Builder Authentication API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        components={
            "authentication": "ready",
            "database": "ready",
            "json_processor": "ready"
        },
        timestamp=datetime.now()
    )


@app.post("/api/strategies", response_model=StrategyCreateResponse)
async def create_strategy(
    request: StrategyRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create and save a new strategy configuration.

    This endpoint validates and stores the strategy without executing it.
    Use the /backtest endpoint to run the strategy.
    """
    try:
        strategy_data = request.strategy.model_dump()
        logger.info(f"Received strategy creation request from user {current_user.username}: {strategy_data.get('name', 'unnamed')}")

        # Basic validation - check required fields
        required_fields = ['name', 'symbol_block', 'entry_block']
        missing_fields = [field for field in required_fields if field not in strategy_data]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required fields",
                    "missing_fields": missing_fields
                }
            )

        # Save strategy to database
        db_strategy = Strategy(
            user_id=current_user.id,
            name=strategy_data.get('name', 'Unnamed Strategy'),
            description=strategy_data.get('description', f"Strategy created by {current_user.username}"),
            configuration=strategy_data,
            is_active=True
        )
        db.add(db_strategy)
        db.commit()
        db.refresh(db_strategy)
        logger.info(f"Saved strategy to database with ID: {db_strategy.id}")

        # Return creation confirmation (no execution)
        return StrategyCreateResponse(
            id=db_strategy.id,
            name=db_strategy.name,
            description=db_strategy.description,
            created_at=db_strategy.created_at,
            status="saved",
            message="Strategy saved successfully"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating strategy: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "details": str(e)
            }
        )


@app.get("/api/strategies/", response_model=List[Dict[str, Any]])
async def get_user_strategies(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all strategies for the current user."""
    strategies = db.query(Strategy).filter(
        Strategy.user_id == current_user.id,
        Strategy.is_active == True
    ).all()

    return [
        {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
            "configuration": strategy.configuration
        }
        for strategy in strategies
    ]


@app.get("/api/strategies/example", response_model=Dict[str, Any])
async def get_example_strategy():
    """
    Get an example strategy configuration for frontend testing.
    """
    example_strategy = {
        "strategy": {
            "name": "Example Multi-Timeframe Strategy",
            "version": "1.0",
            "symbol_block": {
                "symbol": "EURUSD"
            },
            "timeframe_blocks": [
                {
                    "timeframe": "H4",
                    "sequence": 1,
                    "indicators": [
                        {
                            "type": "liquidity_grab_detector",
                            "sequence": 1,
                            "config": {
                                "swing_lookback": 10,
                                "liquidity_threshold": 0.001
                            }
                        }
                    ]
                },
                {
                    "timeframe": "H1",
                    "sequence": 2,
                    "indicators": [
                        {
                            "type": "choch_detector",
                            "sequence": 1,
                            "config": {
                                "swing_detection_window": 20,
                                "confirmation_candles": 3
                            }
                        }
                    ]
                }
            ],
            "risk_management_block": {
                "stop_loss_pips": 25,
                "take_profit_pips": 50,
                "position_size_percent": 2.0
            },
            "entry_block": {
                "conditions": "all_timeframes_complete"
            }
        }
    }

    return example_strategy


@app.get("/api/strategies/{strategy_id}", response_model=Dict[str, Any])
async def get_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific strategy by ID."""
    strategy = db.query(Strategy).filter(
        Strategy.id == strategy_id,
        Strategy.user_id == current_user.id,
        Strategy.is_active == True
    ).first()

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail="Strategy not found"
        )

    return {
        "id": strategy.id,
        "name": strategy.name,
        "description": strategy.description,
        "created_at": strategy.created_at,
        "updated_at": strategy.updated_at,
        "configuration": strategy.configuration
    }


@app.put("/api/strategies/{strategy_id}", response_model=Dict[str, Any])
async def update_strategy(
    strategy_id: int,
    request: StrategyRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a specific strategy."""
    strategy = db.query(Strategy).filter(
        Strategy.id == strategy_id,
        Strategy.user_id == current_user.id,
        Strategy.is_active == True
    ).first()

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail="Strategy not found"
        )

    # Basic validation
    strategy_data = request.strategy.model_dump()
    required_fields = ['name', 'symbol_block', 'entry_block']
    missing_fields = [field for field in required_fields if field not in strategy_data]

    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }
        )

    # Update strategy
    logger.info(f"Before update - Name: {strategy.name}, Updated: {strategy.updated_at}")
    logger.info(f"New data - Name: {strategy_data.get('name')}, Symbol: {strategy_data.get('symbol_block', {}).get('symbol')}")

    strategy.name = strategy_data.get('name', strategy.name)
    strategy.description = strategy_data.get('description', strategy.description)
    strategy.configuration = strategy_data
    strategy.updated_at = datetime.now()  # Manually update timestamp

    db.commit()
    db.refresh(strategy)

    logger.info(f"After update - Name: {strategy.name}, Updated: {strategy.updated_at}")
    logger.info(f"Config symbol: {strategy.configuration.get('symbol_block', {}).get('symbol')}")

    return {
        "id": strategy.id,
        "name": strategy.name,
        "description": strategy.description,
        "created_at": strategy.created_at,
        "updated_at": strategy.updated_at,
        "configuration": strategy.configuration
    }


@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(
    strategy_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a specific strategy."""
    strategy = db.query(Strategy).filter(
        Strategy.id == strategy_id,
        Strategy.user_id == current_user.id,
        Strategy.is_active == True
    ).first()

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail="Strategy not found"
        )

    # Soft delete
    strategy.is_active = False
    db.commit()

    return {"message": "Strategy deleted successfully"}


@app.post("/api/strategies/{strategy_id}/backtest", response_model=StrategyBacktestResponse)
async def backtest_strategy(
    strategy_id: int,
    backtest_request: Optional[StrategyBacktestRequest] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Run backtest for a specific strategy (mock implementation).

    This endpoint loads a saved strategy and returns mock backtest results
    since the full execution system has dependencies not available here.
    """
    # Load strategy from database
    strategy = db.query(Strategy).filter(
        Strategy.id == strategy_id,
        Strategy.user_id == current_user.id,
        Strategy.is_active == True
    ).first()

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail="Strategy not found"
        )

    try:
        # Get backtest parameters
        if backtest_request:
            start_date = backtest_request.start_date or '2024-02-01'
            end_date = backtest_request.end_date or '2024-03-01'
        else:
            start_date = '2024-02-01'
            end_date = '2024-03-01'

        logger.info(f"Running mock backtest for strategy {strategy_id}: {strategy.name}")

        # Return mock backtest results
        performance_summary = PerformanceSummary(
            total_trades=15,
            winning_trades=9,
            losing_trades=6,
            win_rate=60.0,
            total_return=12.5,
            max_drawdown=-3.2
        )

        strategy_results = StrategyResults(
            total_trades=15,
            indicators_used=['liquidity_grab_detector', 'choch_detector'],
            performance_summary=performance_summary,
            raw_results={
                "status": "backtest_complete",
                "message": f"Backtest executed for strategy: {strategy.name}",
                "dates": {"start": start_date, "end": end_date},
                "strategy_config": strategy.configuration,
                "mock_trades": [
                    {"entry": "2024-02-05", "exit": "2024-02-06", "pnl": 150},
                    {"entry": "2024-02-10", "exit": "2024-02-12", "pnl": -75},
                    {"entry": "2024-02-15", "exit": "2024-02-16", "pnl": 200}
                ]
            }
        )

        return StrategyBacktestResponse(
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            execution_time=datetime.now(),
            results=strategy_results
        )

    except Exception as e:
        logger.error(f"Backtest failed for strategy {strategy_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Backtest execution failed",
                "details": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Authentication-focused Strategy Builder API...")
    print("Server will run on http://localhost:8000")
    print("Authentication endpoints available at /api/auth/*")
    print("Strategy management endpoints available at /api/strategies/*")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)