"""Hard-delete records that have been soft-deleted beyond the retention period."""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_factory
from app.models import User, Course, Class, Lesson, Material


async def hard_delete(older_than_days: int):
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    async with async_session_factory() as session:
        # Delete materials first (leaf nodes), clean up PDF files
        result = await session.execute(
            select(Material).where(
                Material.deleted_at.is_not(None),
                Material.deleted_at < cutoff,
            )
        )
        materials = result.scalars().all()

        for mat in materials:
            if mat.type.value == "pdf" and mat.file_path:
                full_path = os.path.join(settings.UPLOAD_DIR, mat.file_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"  Deleted file: {full_path}")
            await session.delete(mat)

        print(f"Hard-deleted {len(materials)} materials")

        # Delete lessons
        result = await session.execute(
            delete(Lesson).where(
                Lesson.deleted_at.is_not(None),
                Lesson.deleted_at < cutoff,
            )
        )
        print(f"Hard-deleted {result.rowcount} lessons")

        # Delete classes
        result = await session.execute(
            delete(Class).where(
                Class.deleted_at.is_not(None),
                Class.deleted_at < cutoff,
            )
        )
        print(f"Hard-deleted {result.rowcount} classes")

        # Delete courses
        result = await session.execute(
            delete(Course).where(
                Course.deleted_at.is_not(None),
                Course.deleted_at < cutoff,
            )
        )
        print(f"Hard-deleted {result.rowcount} courses")

        # Delete users
        result = await session.execute(
            delete(User).where(
                User.deleted_at.is_not(None),
                User.deleted_at < cutoff,
            )
        )
        print(f"Hard-deleted {result.rowcount} users")

        await session.commit()
        print("Hard delete complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard-delete old soft-deleted records")
    settings = get_settings()
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=settings.HARD_DELETE_RETENTION_DAYS,
        help=f"Delete records soft-deleted more than N days ago (default: {settings.HARD_DELETE_RETENTION_DAYS})",
    )
    args = parser.parse_args()

    print(
        f"Hard-deleting records soft-deleted more than {args.older_than_days} days ago..."
    )
    asyncio.run(hard_delete(args.older_than_days))
