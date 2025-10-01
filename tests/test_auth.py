"""
Tests for authentication system.
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from database.connection import Base, get_db
from database.models import User, UserSession
from api.server import app
from api.auth.utils import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    validate_password_strength
)


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def test_db():
    """Create test database for each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "TestPassword123!"
    }


class TestPasswordUtils:
    """Test password utilities."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

    def test_password_strength_validation(self):
        """Test password strength validation."""
        # Valid password
        is_valid, message = validate_password_strength("TestPassword123!")
        assert is_valid
        assert message == "Password is valid"

        # Too short
        is_valid, message = validate_password_strength("Test1!")
        assert not is_valid
        assert "at least 8 characters" in message

        # No uppercase
        is_valid, message = validate_password_strength("testpassword123!")
        assert not is_valid
        assert "uppercase letter" in message

        # No lowercase
        is_valid, message = validate_password_strength("TESTPASSWORD123!")
        assert not is_valid
        assert "lowercase letter" in message

        # No number
        is_valid, message = validate_password_strength("TestPassword!")
        assert not is_valid
        assert "number" in message


class TestJWTUtils:
    """Test JWT token utilities."""

    def test_create_and_verify_access_token(self):
        """Test access token creation and verification."""
        data = {"sub": 123}
        token = create_access_token(data)

        assert token is not None

        # Verify token
        payload = verify_token(token, expected_type="access")
        assert payload is not None
        assert payload["sub"] == 123
        assert payload["type"] == "access"

    def test_create_and_verify_refresh_token(self):
        """Test refresh token creation and verification."""
        data = {"sub": 123}
        token = create_refresh_token(data)

        assert token is not None

        # Verify token
        payload = verify_token(token, expected_type="refresh")
        assert payload is not None
        assert payload["sub"] == 123
        assert payload["type"] == "refresh"

    def test_verify_invalid_token(self):
        """Test verification of invalid tokens."""
        # Invalid token
        payload = verify_token("invalid_token")
        assert payload is None

        # Wrong token type
        access_token = create_access_token({"sub": 123})
        payload = verify_token(access_token, expected_type="refresh")
        assert payload is None


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_user_registration_success(self, client, test_db, test_user_data):
        """Test successful user registration."""
        response = client.post("/api/auth/register", json=test_user_data)

        assert response.status_code == 201
        data = response.json()

        assert "user" in data
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        user = data["user"]
        assert user["email"] == test_user_data["email"]
        assert user["username"] == test_user_data["username"]
        assert user["is_active"] is True

    def test_user_registration_duplicate_email(self, client, test_db, test_user_data):
        """Test registration with duplicate email."""
        # Register first user
        client.post("/api/auth/register", json=test_user_data)

        # Try to register with same email
        duplicate_data = test_user_data.copy()
        duplicate_data["username"] = "different_username"
        response = client.post("/api/auth/register", json=duplicate_data)

        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    def test_user_registration_duplicate_username(self, client, test_db, test_user_data):
        """Test registration with duplicate username."""
        # Register first user
        client.post("/api/auth/register", json=test_user_data)

        # Try to register with same username
        duplicate_data = test_user_data.copy()
        duplicate_data["email"] = "different@example.com"
        response = client.post("/api/auth/register", json=duplicate_data)

        assert response.status_code == 400
        assert "Username already taken" in response.json()["detail"]

    def test_user_registration_weak_password(self, client, test_db):
        """Test registration with weak password."""
        weak_password_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "weak"
        }
        response = client.post("/api/auth/register", json=weak_password_data)

        assert response.status_code == 400
        assert "at least 8 characters" in response.json()["detail"]

    def test_user_login_success(self, client, test_db, test_user_data):
        """Test successful user login."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)

        # Login with username
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
        response = client.post("/api/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()

        assert "user" in data
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_user_login_with_email(self, client, test_db, test_user_data):
        """Test login with email instead of username."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)

        # Login with email
        login_data = {
            "username": test_user_data["email"],  # Using email as username
            "password": test_user_data["password"]
        }
        response = client.post("/api/auth/login", json=login_data)

        assert response.status_code == 200

    def test_user_login_invalid_credentials(self, client, test_db, test_user_data):
        """Test login with invalid credentials."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)

        # Login with wrong password
        login_data = {
            "username": test_user_data["username"],
            "password": "wrong_password"
        }
        response = client.post("/api/auth/login", json=login_data)

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_token_refresh_success(self, client, test_db, test_user_data):
        """Test successful token refresh."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })

        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/api/auth/refresh", json=refresh_data)

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_token_refresh_invalid_token(self, client, test_db):
        """Test token refresh with invalid token."""
        refresh_data = {"refresh_token": "invalid_token"}
        response = client.post("/api/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]

    def test_user_logout_success(self, client, test_db, test_user_data):
        """Test successful user logout."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })

        refresh_token = login_response.json()["refresh_token"]

        # Logout
        logout_data = {"refresh_token": refresh_token}
        response = client.post("/api/auth/logout", json=logout_data)

        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]

        # Try to use refresh token after logout (should fail)
        refresh_response = client.post("/api/auth/refresh", json=logout_data)
        assert refresh_response.status_code == 401

    def test_get_current_user_success(self, client, test_db, test_user_data):
        """Test getting current user information."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })

        access_token = login_response.json()["access_token"]

        # Get current user
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["email"] == test_user_data["email"]
        assert data["username"] == test_user_data["username"]
        assert data["is_active"] is True

    def test_get_current_user_unauthorized(self, client, test_db):
        """Test getting current user without authentication."""
        response = client.get("/api/auth/me")

        assert response.status_code == 401

    def test_get_current_user_invalid_token(self, client, test_db):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 401


class TestProtectedEndpoints:
    """Test protected strategy endpoints."""

    def test_protected_endpoint_without_auth(self, client, test_db):
        """Test accessing protected endpoint without authentication."""
        strategy_data = {
            "strategy": {
                "name": "Test Strategy",
                "symbol_block": {"symbol": "EURUSD"},
                "timeframe_blocks": [],
                "entry_block": {"conditions": "test"}
            }
        }
        response = client.post("/api/strategies", json=strategy_data)

        assert response.status_code == 401

    def test_protected_endpoint_with_auth(self, client, test_db, test_user_data):
        """Test accessing protected endpoint with authentication."""
        # Register and login user
        client.post("/api/auth/register", json=test_user_data)
        login_response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })

        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Test getting user strategies
        response = client.get("/api/strategies/", headers=headers)

        assert response.status_code == 200
        assert isinstance(response.json(), list)