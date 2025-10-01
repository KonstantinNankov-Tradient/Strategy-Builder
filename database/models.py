"""
SQLAlchemy models for the Strategy Builder application.
"""

from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.connection import Base


class User(Base):
    """User model for authentication and strategy ownership."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    strategies = relationship("Strategy", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    backtest_results = relationship("BacktestResult", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class UserSession(Base):
    """User session model for JWT token management."""

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")

    @property
    def is_expired(self):
        """Check if the session is expired."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        return now > self.expires_at

    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, expires_at='{self.expires_at}')>"


class Strategy(Base):
    """Strategy model for storing user-created trading strategies."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    configuration = Column(JSON, nullable=False)  # Store strategy JSON configuration
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="strategies")
    backtest_results = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class BacktestResult(Base):
    """Model for storing backtesting results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    results = Column(JSON, nullable=False)  # Store backtesting results JSON
    performance_summary = Column(JSON, nullable=True)  # Store performance metrics
    execution_time_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    strategy = relationship("Strategy", back_populates="backtest_results")
    user = relationship("User", back_populates="backtest_results")

    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy_id={self.strategy_id}, user_id={self.user_id})>"