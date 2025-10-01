"""
FastAPI server for strategy builder backend.

Provides HTTP API endpoints for frontend integration, focusing on
JSON strategy processing and execution.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
from sqlalchemy.orm import Session

from core.json_converter import JSONToObjectConverter
from core.json_validation import AdvancedJSONValidator
from strategy_runner import StrategyRunner
from strategies.multi_indicator_strategies import get_strategy
from config.settings import settings
from database.connection import get_db, create_tables
from database.models import User, Strategy
from api.auth.router import router as auth_router
from api.auth.dependencies import get_current_active_user
from .models import (
    StrategyRequest,
    StrategyResponse,
    StrategyCreateResponse,
    StrategyBacktestRequest,
    StrategyBacktestResponse,
    ErrorResponse,
    HealthResponse,
    ExampleStrategyResponse,
    PerformanceSummary,
    StrategyResults
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Strategy Builder API",
    description="Backend API for multi-timeframe trading strategy execution",
    version="1.0.0"
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

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    logger.info("Creating database tables...")
    create_tables()
    logger.info("Database initialization complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Strategy Builder API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        components={
            "json_processor": "ready",
            "strategy_runner": "ready",
            "indicators": "ready"
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

    Accepts a multi-timeframe strategy configuration from the frontend,
    validates it, and saves it to the database for later execution.

    Args:
        request: Strategy configuration from frontend

    Returns:
        Strategy creation confirmation with ID and metadata
    """
    try:
        strategy_data = request.strategy.dict()
        logger.info(f"Received strategy creation request from user {current_user.username}: {strategy_data.get('name', 'unnamed')}")

        # Step 1: Validate JSON structure
        validator = AdvancedJSONValidator()
        validation_result = validator.validate_strategy_comprehensive({"strategy": strategy_data})
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid strategy configuration",
                    "validation_errors": validation_result.errors
                }
            )

        # Step 2: Validate conversion (but don't execute)
        try:
            converter = JSONToObjectConverter()
            strategy_config = converter.convert_json_to_strategy({"strategy": strategy_data})
            logger.info(f"Successfully validated strategy: {strategy_config.name}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Failed to validate strategy configuration",
                    "details": str(e)
                }
            )

        # Step 3: Save strategy to database
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

        # Step 4: Return creation confirmation
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


@app.get("/api/strategies/example", response_model=Dict[str, Any])
async def get_example_strategy():
    """
    Get an example strategy configuration for frontend testing.

    Returns:
        Example JSON strategy configuration
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


@app.post("/api/strategies/{strategy_id}/backtest", response_model=StrategyBacktestResponse)
async def backtest_strategy(
    strategy_id: int,
    backtest_request: Optional[StrategyBacktestRequest] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Run backtest for a specific strategy.

    This endpoint loads a saved strategy and executes it through the backtesting engine.

    Args:
        strategy_id: ID of the strategy to backtest
        backtest_request: Optional backtest parameters (dates, initial capital)

    Returns:
        Backtest results including trades, performance metrics, and statistics
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

        logger.info(f"Running backtest for strategy {strategy_id}: {strategy.name}")

        # TODO: Replace with actual JSON strategy execution
        # For now, use mock execution with default strategy
        default_strategy = get_strategy('liquidity_grab_choch')
        runner = StrategyRunner(default_strategy)

        # Execute the backtest
        results = runner.run_backtest(
            start_date=start_date,
            end_date=end_date
        )

        logger.info(f"Backtest completed for strategy {strategy_id}")

        # Format results
        performance_summary = PerformanceSummary(
            total_trades=len(results.get('trades', [])),
            winning_trades=results.get('summary', {}).get('winning_trades', 0),
            losing_trades=results.get('summary', {}).get('losing_trades', 0),
            win_rate=results.get('summary', {}).get('win_rate', 0.0),
            total_return=results.get('summary', {}).get('total_return', 0.0),
            max_drawdown=results.get('summary', {}).get('max_drawdown', 0.0)
        )

        strategy_results = StrategyResults(
            total_trades=len(results.get('trades', [])),
            indicators_used=['liquidity_grab_detector', 'choch_detector'],  # Mock for now
            performance_summary=performance_summary,
            raw_results={
                "status": "backtest_complete",
                "message": f"Backtest executed for strategy: {strategy.name}",
                "dates": {"start": start_date, "end": end_date},
                **results
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

    # Validate the updated strategy
    strategy_data = request.strategy.dict()
    validator = AdvancedJSONValidator()
    validation_result = validator.validate_strategy_comprehensive({"strategy": strategy_data})
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid strategy configuration",
                "validation_errors": validation_result.errors
            }
        )

    # Update strategy
    strategy.name = strategy_data.get('name', strategy.name)
    strategy.configuration = strategy_data
    db.commit()
    db.refresh(strategy)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)