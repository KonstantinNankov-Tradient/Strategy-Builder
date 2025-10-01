"""
Tests for password utilities.
"""

import pytest
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from api.auth.utils import (
    get_password_hash,
    verify_password,
    validate_password_strength
)


class TestPasswordUtils:
    """Test password utilities."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

    def test_password_strength_validation_valid(self):
        """Test password strength validation with valid password."""
        is_valid, message = validate_password_strength("TestPassword123!")
        assert is_valid
        assert message == "Password is valid"

    def test_password_strength_validation_too_short(self):
        """Test password strength validation - too short."""
        is_valid, message = validate_password_strength("Test1!")
        assert not is_valid
        assert "at least 8 characters" in message

    def test_password_strength_validation_no_uppercase(self):
        """Test password strength validation - no uppercase."""
        is_valid, message = validate_password_strength("testpassword123!")
        assert not is_valid
        assert "uppercase letter" in message

    def test_password_strength_validation_no_lowercase(self):
        """Test password strength validation - no lowercase."""
        is_valid, message = validate_password_strength("TESTPASSWORD123!")
        assert not is_valid
        assert "lowercase letter" in message

    def test_password_strength_validation_no_number(self):
        """Test password strength validation - no number."""
        is_valid, message = validate_password_strength("TestPassword!")
        assert not is_valid
        assert "number" in message