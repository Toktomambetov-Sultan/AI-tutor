import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from passlib.hash import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User, UserRole
from app.redis import get_redis
from app.schemas.auth import (
    ChangePasswordRequest,
    ChangeUsernameRequest,
    LoginRequest,
    LoginResponse,
    MessageResponse,
    RegisterRequest,
)

router = APIRouter()
settings = get_settings()


@router.post(
    "/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Student self-registration (public endpoint)."""
    result = await db.execute(
        select(User).where(User.email == body.email, User.deleted_at.is_(None))
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = User(
        full_name=body.full_name,
        email=body.email,
        password_hash=bcrypt.hash(body.password),
        role=UserRole.student,
    )
    db.add(user)
    await db.flush()

    return MessageResponse(detail="Registration successful")


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Authenticate and obtain a Bearer token."""
    result = await db.execute(
        select(User).where(User.email == body.email, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if not user or not bcrypt.verify(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    token = str(uuid.uuid4())
    redis = await get_redis()
    await redis.setex(
        f"session:{token}",
        settings.SESSION_TTL_SECONDS,
        json.dumps({"user_id": str(user.id), "role": user.role.value}),
    )

    return LoginResponse(
        token=token,
        user_id=str(user.id),
        role=user.role.value,
        full_name=user.full_name,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request):
    """Invalidate the current session token."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token"
        )

    token = auth_header.split(" ", 1)[1]
    redis = await get_redis()

    session_data = await redis.get(f"session:{token}")
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    await redis.delete(f"session:{token}")
    return MessageResponse(detail="Logged out successfully")


@router.patch("/change-password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Change own password (requires current password verification)."""
    if not bcrypt.verify(body.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    current_user.password_hash = bcrypt.hash(body.new_password)
    await db.flush()

    return MessageResponse(detail="Password changed successfully")


@router.patch("/change-username", response_model=MessageResponse)
async def change_username(
    body: ChangeUsernameRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Change own display name."""
    current_user.full_name = body.new_full_name
    await db.flush()

    return MessageResponse(detail="Username changed successfully")
