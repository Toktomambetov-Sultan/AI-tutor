from datetime import datetime
from pydantic import BaseModel, Field


class CreateCourseRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class UpdateCourseRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class CourseResponse(BaseModel):
    id: str
    teacher_id: str
    title: str
    description: str | None = None
    teacher_name: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class CourseListResponse(BaseModel):
    courses: list[CourseResponse]
    total: int


class CourseDetailResponse(CourseResponse):
    classes: list["ClassDetailResponse"] = []


# --- Class schemas ---


class CreateClassRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    order: int = 0


class UpdateClassRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    order: int | None = None


class ClassResponse(BaseModel):
    id: str
    course_id: str
    title: str
    order: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ClassDetailResponse(ClassResponse):
    lessons: list["LessonResponse"] = []


# --- Lesson schemas ---


class CreateLessonRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    order: int = 0


class UpdateLessonRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    order: int | None = None


class LessonResponse(BaseModel):
    id: str
    class_id: str
    title: str
    order: int
    created_at: datetime

    model_config = {"from_attributes": True}


class LessonDetailResponse(LessonResponse):
    materials: list["MaterialResponse"] = []


# --- Material schemas ---


class CreateTextMaterialRequest(BaseModel):
    content: str = Field(..., min_length=1)


class MaterialResponse(BaseModel):
    id: str
    lesson_id: str
    type: str
    content: str | None = None
    file_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MaterialListResponse(BaseModel):
    materials: list[MaterialResponse]
    total: int
