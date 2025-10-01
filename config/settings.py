"""
Configuration settings for the Strategy Builder application.
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    db_host: str = "localhost"
    db_name: str = "strategy_builder"
    db_user: str = "postgres"
    db_password: str = ""
    db_port: int = 5432
    database_url: str = ""

    # JWT Configuration
    jwt_secret_key: str = "your-super-secret-jwt-key-change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # Security Configuration
    bcrypt_rounds: int = 12

    # CORS Configuration
    allowed_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:5173"

    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def database_url_computed(self) -> str:
        """Generate database URL if not provided directly."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()