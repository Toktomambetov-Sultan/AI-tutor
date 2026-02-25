"""Seed the first Admin account into the database."""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passlib.hash import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_factory
from app.models.user import User, UserRole


async def seed_admin():
    settings = get_settings()

    async with async_session_factory() as session:
        # Check if admin already exists
        result = await session.execute(
            select(User).where(
                User.email == settings.ADMIN_EMAIL,
                User.role == UserRole.admin,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            print(f"Admin account already exists: {settings.ADMIN_EMAIL}")
            return

        admin = User(
            full_name=settings.ADMIN_FULL_NAME,
            email=settings.ADMIN_EMAIL,
            password_hash=bcrypt.hash(settings.ADMIN_PASSWORD),
            role=UserRole.admin,
        )
        session.add(admin)
        await session.commit()
        print(f"Admin account created: {settings.ADMIN_EMAIL}")


if __name__ == "__main__":
    asyncio.run(seed_admin())
