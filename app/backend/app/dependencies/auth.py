import json
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, UserRole
from app.redis import get_redis

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Extract and validate the Bearer token, return the authenticated user."""
    token = credentials.credentials
    redis = await get_redis()

    session_data = await redis.get(f"session:{token}")
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    data = json.loads(session_data)
    user_id = uuid.UUID(data["user_id"])

    result = await db.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or has been deleted",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


def require_role(*roles: UserRole):
    """Dependency factory that restricts access to specific roles."""

    async def role_checker(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_checker


# Convenience dependencies
require_admin = require_role(UserRole.admin)
require_teacher = require_role(UserRole.teacher)
require_student = require_role(UserRole.student)
