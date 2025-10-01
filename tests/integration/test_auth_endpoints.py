"""
Integration tests for authentication endpoints.
"""

import pytest
import os
import sys
import requests
import time

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class TestAuthEndpointsIntegration:
    """Integration tests for authentication endpoints."""

    BASE_URL = "http://localhost:8000"

    @pytest.fixture(autouse=True)
    def check_server(self):
        """Check if server is running before each test."""
        try:
            response = requests.get(f"{self.BASE_URL}/api/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Server not running or not healthy")
        except requests.ConnectionError:
            pytest.skip("Server not running on localhost:8000")

    def test_user_registration_success(self):
        """Test successful user registration."""
        timestamp = int(time.time())
        user_data = {
            "email": f"test{timestamp}@example.com",
            "username": f"testuser{timestamp}",
            "password": "TestPassword123!"
        }

        response = requests.post(f"{self.BASE_URL}/api/auth/register", json=user_data)

        if response.status_code == 400 and "already" in response.text:
            pytest.skip("User already exists - database not cleaned")

        assert response.status_code == 201
        data = response.json()

        assert "user" in data
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        user = data["user"]
        assert user["email"] == user_data["email"]
        assert user["username"] == user_data["username"]
        assert user["is_active"] is True

    def test_user_login_success(self):
        """Test successful user login."""
        timestamp = int(time.time())
        user_data = {
            "email": f"test{timestamp}@example.com",
            "username": f"testuser{timestamp}",
            "password": "TestPassword123!"
        }

        # Register user first
        register_response = requests.post(f"{self.BASE_URL}/api/auth/register", json=user_data)
        if register_response.status_code != 201:
            if "already" in register_response.text:
                pass  # User exists, continue with login
            else:
                pytest.fail(f"Registration failed: {register_response.text}")

        # Login with username
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        response = requests.post(f"{self.BASE_URL}/api/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()

        assert "user" in data
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_protected_endpoint_access(self):
        """Test accessing protected endpoint with authentication."""
        timestamp = int(time.time())
        user_data = {
            "email": f"test{timestamp}@example.com",
            "username": f"testuser{timestamp}",
            "password": "TestPassword123!"
        }

        # Register and login user
        register_response = requests.post(f"{self.BASE_URL}/api/auth/register", json=user_data)
        if register_response.status_code == 201:
            auth_data = register_response.json()
        else:
            # Try login if registration failed due to existing user
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"]
            }
            login_response = requests.post(f"{self.BASE_URL}/api/auth/login", json=login_data)
            if login_response.status_code == 200:
                auth_data = login_response.json()
            else:
                pytest.fail("Could not authenticate user")

        access_token = auth_data["access_token"]

        # Test protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.BASE_URL}/api/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["is_active"] is True

    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication."""
        response = requests.get(f"{self.BASE_URL}/api/auth/me")
        assert response.status_code == 401

    def test_token_refresh(self):
        """Test token refresh functionality."""
        timestamp = int(time.time())
        user_data = {
            "email": f"test{timestamp}@example.com",
            "username": f"testuser{timestamp}",
            "password": "TestPassword123!"
        }

        # Register user
        register_response = requests.post(f"{self.BASE_URL}/api/auth/register", json=user_data)
        if register_response.status_code != 201:
            pytest.skip("Could not register user for token refresh test")

        auth_data = register_response.json()
        refresh_token = auth_data["refresh_token"]

        # Refresh token
        refresh_data = {"refresh_token": refresh_token}
        response = requests.post(f"{self.BASE_URL}/api/auth/refresh", json=refresh_data)

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_user_logout(self):
        """Test user logout functionality."""
        timestamp = int(time.time())
        user_data = {
            "email": f"test{timestamp}@example.com",
            "username": f"testuser{timestamp}",
            "password": "TestPassword123!"
        }

        # Register user
        register_response = requests.post(f"{self.BASE_URL}/api/auth/register", json=user_data)
        if register_response.status_code != 201:
            pytest.skip("Could not register user for logout test")

        auth_data = register_response.json()
        refresh_token = auth_data["refresh_token"]

        # Logout
        logout_data = {"refresh_token": refresh_token}
        response = requests.post(f"{self.BASE_URL}/api/auth/logout", json=logout_data)

        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]

        # Try to use refresh token after logout (should fail)
        refresh_response = requests.post(f"{self.BASE_URL}/api/auth/refresh", json=logout_data)
        assert refresh_response.status_code == 401