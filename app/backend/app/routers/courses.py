import os
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.dependencies.auth import get_current_user, require_teacher
from app.models import User, UserRole, Course, Class, Lesson, Material, MaterialType
from app.schemas.course import (
    ClassDetailResponse,
    ClassResponse,
    CourseDetailResponse,
    CourseListResponse,
    CourseResponse,
    CreateClassRequest,
    CreateCourseRequest,
    CreateLessonRequest,
    CreateTextMaterialRequest,
    LessonDetailResponse,
    LessonResponse,
    MaterialListResponse,
    MaterialResponse,
    UpdateClassRequest,
    UpdateCourseRequest,
    UpdateLessonRequest,
)
from app.schemas.auth import MessageResponse

router = APIRouter()
settings = get_settings()


# ─── Course CRUD ───


@router.post(
    "/courses", response_model=CourseResponse, status_code=status.HTTP_201_CREATED
)
async def create_course(
    body: CreateCourseRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new course (Teacher only)."""
    course = Course(
        teacher_id=teacher.id,
        title=body.title,
        description=body.description,
    )
    db.add(course)
    await db.flush()
    await db.refresh(course)

    return CourseResponse(
        id=str(course.id),
        teacher_id=str(course.teacher_id),
        title=course.title,
        description=course.description,
        teacher_name=teacher.full_name,
        created_at=course.created_at,
        updated_at=course.updated_at,
    )


@router.get("/courses", response_model=CourseListResponse)
async def list_courses(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    List courses.
    - Teachers: see own courses.
    - Students: see all non-deleted courses.
    """
    query = select(Course).where(Course.deleted_at.is_(None))

    if current_user.role == UserRole.teacher:
        query = query.where(Course.teacher_id == current_user.id)

    query = query.order_by(Course.created_at.desc())
    result = await db.execute(query)
    courses = result.scalars().all()

    return CourseListResponse(
        courses=[
            CourseResponse(
                id=str(c.id),
                teacher_id=str(c.teacher_id),
                title=c.title,
                description=c.description,
                teacher_name=c.teacher.full_name if c.teacher else None,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in courses
        ],
        total=len(courses),
    )


@router.get("/courses/{course_id}", response_model=CourseDetailResponse)
async def get_course(
    course_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get course details with classes and lessons."""
    result = await db.execute(
        select(Course).where(Course.id == course_id, Course.deleted_at.is_(None))
    )
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    # Teachers can only access their own courses
    if current_user.role == UserRole.teacher and course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your course"
        )

    # Build class list with lessons
    classes_result = await db.execute(
        select(Class)
        .where(Class.course_id == course.id, Class.deleted_at.is_(None))
        .order_by(Class.order)
    )
    classes = classes_result.scalars().all()

    class_responses = []
    for cls in classes:
        lessons_result = await db.execute(
            select(Lesson)
            .where(Lesson.class_id == cls.id, Lesson.deleted_at.is_(None))
            .order_by(Lesson.order)
        )
        lessons = lessons_result.scalars().all()

        class_responses.append(
            ClassDetailResponse(
                id=str(cls.id),
                course_id=str(cls.course_id),
                title=cls.title,
                order=cls.order,
                created_at=cls.created_at,
                lessons=[
                    LessonResponse(
                        id=str(l.id),
                        class_id=str(l.class_id),
                        title=l.title,
                        order=l.order,
                        created_at=l.created_at,
                    )
                    for l in lessons
                ],
            )
        )

    return CourseDetailResponse(
        id=str(course.id),
        teacher_id=str(course.teacher_id),
        title=course.title,
        description=course.description,
        teacher_name=course.teacher.full_name if course.teacher else None,
        created_at=course.created_at,
        updated_at=course.updated_at,
        classes=class_responses,
    )


@router.put("/courses/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: uuid.UUID,
    body: UpdateCourseRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update course metadata (Teacher only, own courses)."""
    result = await db.execute(
        select(Course).where(
            Course.id == course_id,
            Course.teacher_id == teacher.id,
            Course.deleted_at.is_(None),
        )
    )
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    if body.title is not None:
        course.title = body.title
    if body.description is not None:
        course.description = body.description

    await db.flush()
    await db.refresh(course)

    return CourseResponse(
        id=str(course.id),
        teacher_id=str(course.teacher_id),
        title=course.title,
        description=course.description,
        teacher_name=teacher.full_name,
        created_at=course.created_at,
        updated_at=course.updated_at,
    )


@router.delete("/courses/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a course with cascade (Teacher only, own courses)."""
    result = await db.execute(
        select(Course).where(
            Course.id == course_id,
            Course.teacher_id == teacher.id,
            Course.deleted_at.is_(None),
        )
    )
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    now = datetime.now(timezone.utc)
    course.deleted_at = now
    course.deleted_by = teacher.id

    # Cascade to classes
    classes_result = await db.execute(
        select(Class).where(Class.course_id == course.id, Class.deleted_at.is_(None))
    )
    for cls in classes_result.scalars().all():
        cls.deleted_at = now
        cls.deleted_by = teacher.id

        lessons_result = await db.execute(
            select(Lesson).where(Lesson.class_id == cls.id, Lesson.deleted_at.is_(None))
        )
        for lesson in lessons_result.scalars().all():
            lesson.deleted_at = now
            lesson.deleted_by = teacher.id

            materials_result = await db.execute(
                select(Material).where(
                    Material.lesson_id == lesson.id, Material.deleted_at.is_(None)
                )
            )
            for mat in materials_result.scalars().all():
                mat.deleted_at = now
                mat.deleted_by = teacher.id

    await db.flush()


# ─── Class Management ───


@router.post(
    "/courses/{course_id}/classes",
    response_model=ClassResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_class(
    course_id: uuid.UUID,
    body: CreateClassRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Add a class to a course (Teacher only)."""
    result = await db.execute(
        select(Course).where(
            Course.id == course_id,
            Course.teacher_id == teacher.id,
            Course.deleted_at.is_(None),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Course not found"
        )

    cls = Class(
        course_id=course_id,
        title=body.title,
        order=body.order,
    )
    db.add(cls)
    await db.flush()
    await db.refresh(cls)

    return ClassResponse(
        id=str(cls.id),
        course_id=str(cls.course_id),
        title=cls.title,
        order=cls.order,
        created_at=cls.created_at,
    )


@router.put("/classes/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: uuid.UUID,
    body: UpdateClassRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update class details (Teacher only)."""
    result = await db.execute(
        select(Class)
        .join(Course)
        .where(
            Class.id == class_id,
            Course.teacher_id == teacher.id,
            Class.deleted_at.is_(None),
        )
    )
    cls = result.scalar_one_or_none()

    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Class not found"
        )

    if body.title is not None:
        cls.title = body.title
    if body.order is not None:
        cls.order = body.order

    await db.flush()
    await db.refresh(cls)

    return ClassResponse(
        id=str(cls.id),
        course_id=str(cls.course_id),
        title=cls.title,
        order=cls.order,
        created_at=cls.created_at,
    )


@router.delete("/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(
    class_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a class with cascade (Teacher only)."""
    result = await db.execute(
        select(Class)
        .join(Course)
        .where(
            Class.id == class_id,
            Course.teacher_id == teacher.id,
            Class.deleted_at.is_(None),
        )
    )
    cls = result.scalar_one_or_none()

    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Class not found"
        )

    now = datetime.now(timezone.utc)
    cls.deleted_at = now
    cls.deleted_by = teacher.id

    # Cascade
    lessons_result = await db.execute(
        select(Lesson).where(Lesson.class_id == cls.id, Lesson.deleted_at.is_(None))
    )
    for lesson in lessons_result.scalars().all():
        lesson.deleted_at = now
        lesson.deleted_by = teacher.id

        materials_result = await db.execute(
            select(Material).where(
                Material.lesson_id == lesson.id, Material.deleted_at.is_(None)
            )
        )
        for mat in materials_result.scalars().all():
            mat.deleted_at = now
            mat.deleted_by = teacher.id

    await db.flush()


# ─── Lesson Management ───


@router.post(
    "/classes/{class_id}/lessons",
    response_model=LessonResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_lesson(
    class_id: uuid.UUID,
    body: CreateLessonRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Add a lesson to a class (Teacher only)."""
    result = await db.execute(
        select(Class)
        .join(Course)
        .where(
            Class.id == class_id,
            Course.teacher_id == teacher.id,
            Class.deleted_at.is_(None),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Class not found"
        )

    lesson = Lesson(
        class_id=class_id,
        title=body.title,
        order=body.order,
    )
    db.add(lesson)
    await db.flush()
    await db.refresh(lesson)

    return LessonResponse(
        id=str(lesson.id),
        class_id=str(lesson.class_id),
        title=lesson.title,
        order=lesson.order,
        created_at=lesson.created_at,
    )


@router.put("/lessons/{lesson_id}", response_model=LessonResponse)
async def update_lesson(
    lesson_id: uuid.UUID,
    body: UpdateLessonRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update lesson details (Teacher only)."""
    result = await db.execute(
        select(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Lesson.id == lesson_id,
            Course.teacher_id == teacher.id,
            Lesson.deleted_at.is_(None),
        )
    )
    lesson = result.scalar_one_or_none()

    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    if body.title is not None:
        lesson.title = body.title
    if body.order is not None:
        lesson.order = body.order

    await db.flush()
    await db.refresh(lesson)

    return LessonResponse(
        id=str(lesson.id),
        class_id=str(lesson.class_id),
        title=lesson.title,
        order=lesson.order,
        created_at=lesson.created_at,
    )


@router.delete("/lessons/{lesson_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lesson(
    lesson_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a lesson with cascade to materials (Teacher only)."""
    result = await db.execute(
        select(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Lesson.id == lesson_id,
            Course.teacher_id == teacher.id,
            Lesson.deleted_at.is_(None),
        )
    )
    lesson = result.scalar_one_or_none()

    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    now = datetime.now(timezone.utc)
    lesson.deleted_at = now
    lesson.deleted_by = teacher.id

    materials_result = await db.execute(
        select(Material).where(
            Material.lesson_id == lesson.id, Material.deleted_at.is_(None)
        )
    )
    for mat in materials_result.scalars().all():
        mat.deleted_at = now
        mat.deleted_by = teacher.id

    await db.flush()


# ─── Material Management ───


@router.post(
    "/lessons/{lesson_id}/materials",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_text_material(
    lesson_id: uuid.UUID,
    body: CreateTextMaterialRequest,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Upload a text material to a lesson (Teacher only)."""
    result = await db.execute(
        select(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Lesson.id == lesson_id,
            Course.teacher_id == teacher.id,
            Lesson.deleted_at.is_(None),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    material = Material(
        lesson_id=lesson_id,
        type=MaterialType.text,
        content=body.content,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    return MaterialResponse(
        id=str(material.id),
        lesson_id=str(material.lesson_id),
        type=material.type.value,
        content=material.content,
        file_path=material.file_path,
        created_at=material.created_at,
    )


@router.post(
    "/lessons/{lesson_id}/materials/upload",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf_material(
    lesson_id: uuid.UUID,
    file: UploadFile = File(...),
    teacher: Annotated[User, Depends(require_teacher)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Upload a PDF material to a lesson (Teacher only)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    result = await db.execute(
        select(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Lesson.id == lesson_id,
            Course.teacher_id == teacher.id,
            Lesson.deleted_at.is_(None),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    # Save file
    file_id = str(uuid.uuid4())
    relative_path = f"{lesson_id}/{file_id}_{file.filename}"
    full_path = os.path.join(settings.UPLOAD_DIR, relative_path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    content = await file.read()
    with open(full_path, "wb") as f:
        f.write(content)

    material = Material(
        lesson_id=lesson_id,
        type=MaterialType.pdf,
        file_path=relative_path,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    return MaterialResponse(
        id=str(material.id),
        lesson_id=str(material.lesson_id),
        type=material.type.value,
        content=material.content,
        file_path=material.file_path,
        created_at=material.created_at,
    )


@router.get("/lessons/{lesson_id}/materials", response_model=MaterialListResponse)
async def list_materials(
    lesson_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List materials for a lesson."""
    # Verify lesson exists
    result = await db.execute(
        select(Lesson).where(Lesson.id == lesson_id, Lesson.deleted_at.is_(None))
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found"
        )

    materials_result = await db.execute(
        select(Material).where(
            Material.lesson_id == lesson_id,
            Material.deleted_at.is_(None),
        )
    )
    materials = materials_result.scalars().all()

    return MaterialListResponse(
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
        total=len(materials),
    )


@router.delete("/materials/{material_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_material(
    material_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a material (Teacher only)."""
    result = await db.execute(
        select(Material)
        .join(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Material.id == material_id,
            Course.teacher_id == teacher.id,
            Material.deleted_at.is_(None),
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Material not found"
        )

    material.deleted_at = datetime.now(timezone.utc)
    material.deleted_by = teacher.id
    await db.flush()


@router.get("/teacher/materials/{material_id}")
async def teacher_preview_material(
    material_id: uuid.UUID,
    teacher: Annotated[User, Depends(require_teacher)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Preview / download a material (Teacher only, own courses)."""
    result = await db.execute(
        select(Material)
        .join(Lesson)
        .join(Class)
        .join(Course)
        .where(
            Material.id == material_id,
            Course.teacher_id == teacher.id,
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

    # PDF — return file
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
