import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from passlib.hash import bcrypt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models import User, UserRole, Course, Class, Lesson, Material
from app.schemas.auth import MessageResponse
from app.schemas.user import CreateUserRequest, UserListResponse, UserResponse

router = APIRouter()


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new Teacher or Student account (Admin only)."""
    # Check email uniqueness
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
        role=UserRole(body.role),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return UserResponse(
        id=str(user.id),
        full_name=user.full_name,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        deleted_at=user.deleted_at,
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    role: str | None = Query(None, pattern="^(admin|teacher|student)$"),
    deleted: bool = Query(False),
):
    """List users with optional filters for role and soft-delete status."""
    query = select(User)

    if role:
        query = query.where(User.role == UserRole(role))

    if not deleted:
        query = query.where(User.deleted_at.is_(None))

    query = query.order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    # Count
    count_query = select(func.count(User.id))
    if role:
        count_query = count_query.where(User.role == UserRole(role))
    if not deleted:
        count_query = count_query.where(User.deleted_at.is_(None))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    return UserListResponse(
        users=[
            UserResponse(
                id=str(u.id),
                full_name=u.full_name,
                email=u.email,
                role=u.role.value,
                is_active=u.is_active,
                created_at=u.created_at,
                updated_at=u.updated_at,
                deleted_at=u.deleted_at,
            )
            for u in users
        ],
        total=total,
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get a single user (including soft-deleted)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponse(
        id=str(user.id),
        full_name=user.full_name,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        deleted_at=user.deleted_at,
    )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_user(
    user_id: uuid.UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a user. If teacher, cascade soft-delete all their courses."""
    result = await db.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete yourself"
        )

    now = datetime.now(timezone.utc)
    user.deleted_at = now
    user.deleted_by = admin.id

    # If teacher, cascade soft-delete their courses
    if user.role == UserRole.teacher:
        courses_result = await db.execute(
            select(Course).where(
                Course.teacher_id == user.id, Course.deleted_at.is_(None)
            )
        )
        courses = courses_result.scalars().all()

        for course in courses:
            course.deleted_at = now
            course.deleted_by = admin.id

            # Cascade to classes
            classes_result = await db.execute(
                select(Class).where(
                    Class.course_id == course.id, Class.deleted_at.is_(None)
                )
            )
            classes = classes_result.scalars().all()

            for cls in classes:
                cls.deleted_at = now
                cls.deleted_by = admin.id

                # Cascade to lessons
                lessons_result = await db.execute(
                    select(Lesson).where(
                        Lesson.class_id == cls.id, Lesson.deleted_at.is_(None)
                    )
                )
                lessons = lessons_result.scalars().all()

                for lesson in lessons:
                    lesson.deleted_at = now
                    lesson.deleted_by = admin.id

                    # Cascade to materials
                    materials_result = await db.execute(
                        select(Material).where(
                            Material.lesson_id == lesson.id,
                            Material.deleted_at.is_(None),
                        )
                    )
                    materials = materials_result.scalars().all()

                    for material in materials:
                        material.deleted_at = now
                        material.deleted_by = admin.id

    await db.flush()


@router.post("/users/{user_id}/restore", response_model=UserResponse)
async def restore_user(
    user_id: uuid.UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Restore a soft-deleted user."""
    result = await db.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_not(None))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Deleted user not found"
        )

    deleted_timestamp = user.deleted_at

    user.deleted_at = None
    user.deleted_by = None

    # If teacher, restore courses (and cascade) that share the same deleted_at timestamp
    if user.role == UserRole.teacher:
        courses_result = await db.execute(
            select(Course).where(
                Course.teacher_id == user.id,
                Course.deleted_at == deleted_timestamp,
            )
        )
        courses = courses_result.scalars().all()

        for course in courses:
            course.deleted_at = None
            course.deleted_by = None

            classes_result = await db.execute(
                select(Class).where(
                    Class.course_id == course.id,
                    Class.deleted_at == deleted_timestamp,
                )
            )
            classes = classes_result.scalars().all()

            for cls in classes:
                cls.deleted_at = None
                cls.deleted_by = None

                lessons_result = await db.execute(
                    select(Lesson).where(
                        Lesson.class_id == cls.id,
                        Lesson.deleted_at == deleted_timestamp,
                    )
                )
                lessons = lessons_result.scalars().all()

                for lesson in lessons:
                    lesson.deleted_at = None
                    lesson.deleted_by = None

                    materials_result = await db.execute(
                        select(Material).where(
                            Material.lesson_id == lesson.id,
                            Material.deleted_at == deleted_timestamp,
                        )
                    )
                    materials = materials_result.scalars().all()

                    for material in materials:
                        material.deleted_at = None
                        material.deleted_by = None

    await db.flush()
    await db.refresh(user)

    return UserResponse(
        id=str(user.id),
        full_name=user.full_name,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        deleted_at=user.deleted_at,
    )
