"""
Authentication router for user registration, login, and token management.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database.connection import get_db
from database.models import User
from api.models import (
    UserRegisterRequest,
    UserLoginRequest,
    AuthResponse,
    UserResponse,
    TokenResponse,
    RefreshTokenRequest,
    MessageResponse,
    ErrorResponse
)
from api.auth.utils import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_token,
    create_user_session,
    get_user_session,
    invalidate_user_session,
    validate_password_strength
)
from api.auth.dependencies import get_current_active_user


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with email, username, and password"
)
async def register_user(
    user_data: UserRegisterRequest,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Register a new user."""

    # Validate password strength
    is_valid, message = validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )

    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()

    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        password_hash=hashed_password,
        is_active=True
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    # Create tokens
    access_token = create_access_token(data={"sub": new_user.id})
    refresh_token = create_refresh_token(data={"sub": new_user.id})

    # Create session
    create_user_session(db, new_user.id, refresh_token)

    # Prepare response
    user_response = UserResponse(
        id=new_user.id,
        email=new_user.email,
        username=new_user.username,
        is_active=new_user.is_active,
        created_at=new_user.created_at
    )

    return AuthResponse(
        user=user_response,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login user",
    description="Authenticate user with username/email and password"
)
async def login_user(
    login_data: UserLoginRequest,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Authenticate user and return tokens."""

    # Authenticate user
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    # Create session
    create_user_session(db, user.id, refresh_token)

    # Prepare response
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        is_active=user.is_active,
        created_at=user.created_at
    )

    return AuthResponse(
        user=user_response,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token"
)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
) -> TokenResponse:
    """Refresh access token using refresh token."""

    # Verify refresh token
    payload = verify_token(refresh_data.refresh_token, expected_type="refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Check if session exists and is active
    session = get_user_session(db, refresh_data.refresh_token)
    if not session or session.is_expired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or invalid"
        )

    # Verify user exists and is active
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Create new access token
    new_access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=refresh_data.refresh_token,  # Keep same refresh token
        token_type="bearer"
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout user",
    description="Invalidate user session and refresh token"
)
async def logout_user(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
) -> MessageResponse:
    """Logout user by invalidating session."""

    # Invalidate session
    success = invalidate_user_session(db, refresh_data.refresh_token)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token"
        )

    return MessageResponse(message="Successfully logged out")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get current authenticated user information"
)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """Get current user information."""

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )