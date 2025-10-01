"""
Tests for JWT utilities.
"""

import pytest
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from api.auth.utils import (
    create_access_token,
    create_refresh_token,
    verify_token
)


class TestJWTUtils:
    """Test JWT token utilities."""

    def test_create_and_verify_access_token(self):
        """Test access token creation and verification."""
        data = {"sub": 123}  # Integer will be converted to string internally
        token = create_access_token(data)

        assert token is not None

        # Verify token
        payload = verify_token(token, expected_type="access")
        assert payload is not None
        assert payload["sub"] == "123"  # Will be string in payload
        assert payload["type"] == "access"

    def test_create_and_verify_refresh_token(self):
        """Test refresh token creation and verification."""
        data = {"sub": 123}  # Integer will be converted to string internally
        token = create_refresh_token(data)

        assert token is not None

        # Verify token
        payload = verify_token(token, expected_type="refresh")
        assert payload is not None
        assert payload["sub"] == "123"  # Will be string in payload
        assert payload["type"] == "refresh"

    def test_verify_invalid_token(self):
        """Test verification of invalid tokens."""
        # Invalid token
        payload = verify_token("invalid_token")
        assert payload is None

    def test_verify_wrong_token_type(self):
        """Test verification with wrong token type."""
        # Wrong token type
        access_token = create_access_token({"sub": 123})
        payload = verify_token(access_token, expected_type="refresh")
        assert payload is None

    def test_token_contains_expiration(self):
        """Test that tokens contain expiration."""
        data = {"sub": 123}
        access_token = create_access_token(data)
        payload = verify_token(access_token, expected_type="access")

        assert payload is not None
        assert "exp" in payload
        assert payload["exp"] > 0