from app.models.base import Base, SoftDeleteMixin, TimestampMixin
from app.models.user import User, UserRole
from app.models.course import Course
from app.models.class_ import Class
from app.models.lesson import Lesson
from app.models.material import Material, MaterialType
from app.models.lesson_access import LessonAccess

__all__ = [
    "Base",
    "SoftDeleteMixin",
    "TimestampMixin",
    "User",
    "UserRole",
    "Course",
    "Class",
    "Lesson",
    "Material",
    "MaterialType",
    "LessonAccess",
]
