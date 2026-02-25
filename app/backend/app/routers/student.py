import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.dependencies.auth import require_student
from app.models import User, Course, Class, Lesson, Material, LessonAccess
from app.schemas.course import (
    CourseDetailResponse,
    CourseListResponse,
    CourseResponse,
    ClassResponse,
    LessonDetailResponse,
    LessonResponse,
    MaterialResponse,
)

router = APIRouter()
settings = get_settings()


@router.get("/lessons/{lesson_id}", response_model=LessonDetailResponse)
async def get_lesson(
    lesson_id: uuid.UUID,
    student: Annotated[User, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """View lesson content and materials (Student only). Records access."""
    result = await db.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.deleted_at.is_(None))
    )
    lesson = result.scalar_one_or_none()

    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    # Record access (upsert - ignore if already exists)
    existing = await db.execute(
        select(LessonAccess).where(
            LessonAccess.student_id == student.id,
            LessonAccess.lesson_id == lesson_id,
        )
    )
    if not existing.scalar_one_or_none():
        access = LessonAccess(student_id=student.id, lesson_id=lesson_id)
        db.add(access)
        await db.flush()

    # Get materials
    materials_result = await db.execute(
        select(Material).where(
            Material.lesson_id == lesson_id,
            Material.deleted_at.is_(None),
        )
    )
    materials = materials_result.scalars().all()

    return LessonDetailResponse(
        id=str(lesson.id),
        class_id=str(lesson.class_id),
        title=lesson.title,
        order=lesson.order,
        created_at=lesson.created_at,
        materials=[
            MaterialResponse(
                id=str(m.id),
                lesson_id=str(m.lesson_id),
                type=m.type.value,
                content=m.content,
                file_path=m.file_path,
                created_at=m.created_at,
            )
            for m in materials
        ],
    )


@router.get("/materials/{material_id}")
async def get_material(
    material_id: uuid.UUID,
    student: Annotated[User, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Download or view a material file (Student only)."""
    result = await db.execute(
        select(Material).where(
            Material.id == material_id,
            Material.deleted_at.is_(None),
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Material not found"
        )

    if material.type.value == "text":
        return MaterialResponse(
            id=str(material.id),
            lesson_id=str(material.lesson_id),
            type=material.type.value,
            content=material.content,
            file_path=material.file_path,
            created_at=material.created_at,
        )

    # PDF - return file
    full_path = os.path.join(settings.UPLOAD_DIR, material.file_path)
    if not os.path.exists(full_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found on disk"
        )

    return FileResponse(
        full_path,
        media_type="application/pdf",
        filename=os.path.basename(material.file_path),
    )
