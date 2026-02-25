import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_teacher
from app.models import User, Course, Class, Lesson, LessonAccess
from app.schemas.user import UserResponse, UserListResponse

router = APIRouter()


@router.get("/students", response_model=UserListResponse)
async def list_all_teacher_students(
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List all students who accessed content in any of the teacher's courses."""
    result = await db.execute(
        select(User)
        .join(LessonAccess, LessonAccess.student_id == User.id)
        .join(Lesson, Lesson.id == LessonAccess.lesson_id)
        .join(Class, Class.id == Lesson.class_id)
        .join(Course, Course.id == Class.course_id)
        .where(
            Course.teacher_id == teacher.id,
            User.deleted_at.is_(None),
        )
        .distinct()
        .order_by(User.full_name)
    )
    students = result.scalars().all()

    return UserListResponse(
        users=[
            UserResponse(
                id=str(s.id),
                full_name=s.full_name,
                email=s.email,
                role=s.role.value,
                is_active=s.is_active,
                created_at=s.created_at,
                updated_at=s.updated_at,
                deleted_at=s.deleted_at,
            )
            for s in students
        ],
        total=len(students),
    )


@router.get("/courses/{course_id}/students", response_model=UserListResponse)
async def list_course_students(
    course_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List students who accessed content in a specific course."""
    # Verify course belongs to teacher
    course_result = await db.execute(
        select(Course).where(
            Course.id == course_id,
            Course.teacher_id == teacher.id,
        )
    )
    if not course_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    result = await db.execute(
        select(User)
        .join(LessonAccess, LessonAccess.student_id == User.id)
        .join(Lesson, Lesson.id == LessonAccess.lesson_id)
        .join(Class, Class.id == Lesson.class_id)
        .where(
            Class.course_id == course_id,
            User.deleted_at.is_(None),
        )
        .distinct()
        .order_by(User.full_name)
    )
    students = result.scalars().all()

    return UserListResponse(
        users=[
            UserResponse(
                id=str(s.id),
                full_name=s.full_name,
                email=s.email,
                role=s.role.value,
                is_active=s.is_active,
                created_at=s.created_at,
                updated_at=s.updated_at,
                deleted_at=s.deleted_at,
            )
            for s in students
        ],
        total=len(students),
    )
