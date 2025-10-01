"""
Authentication utilities for JWT token handling and password hashing.
"""

from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import User, UserSession


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    # Ensure sub is a string (JWT spec requirement)
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()

    # Ensure sub is a string (JWT spec requirement)
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str, expected_type: str = "access") -> Optional[dict]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        token_type = payload.get("type")

        if token_type != expected_type:
            return None

        return payload
    except JWTError:
        return None


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username/email and password."""
    # Try to find user by username or email
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    if not user.is_active:
        return None

    return user


def create_user_session(db: Session, user_id: int, refresh_token: str) -> UserSession:
    """Create a new user session with refresh token."""
    # Invalidate existing sessions for the user
    db.query(UserSession).filter(
        UserSession.user_id == user_id,
        UserSession.is_active == True
    ).update({"is_active": False})

    # Create new session
    from datetime import timezone
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    session = UserSession(
        user_id=user_id,
        refresh_token=refresh_token,
        expires_at=expires_at,
        is_active=True
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def get_user_session(db: Session, refresh_token: str) -> Optional[UserSession]:
    """Get active user session by refresh token."""
    return db.query(UserSession).filter(
        UserSession.refresh_token == refresh_token,
        UserSession.is_active == True
    ).first()


def invalidate_user_session(db: Session, refresh_token: str) -> bool:
    """Invalidate user session."""
    session = get_user_session(db, refresh_token)
    if session:
        session.is_active = False
        db.commit()
        return True
    return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    return True, "Password is valid"