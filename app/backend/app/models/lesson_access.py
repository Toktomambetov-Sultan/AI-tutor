import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class LessonAccess(Base):
    __tablename__ = "lesson_access"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lessons.id"), nullable=False
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    student = relationship("User", foreign_keys=[student_id], lazy="selectin")
    lesson = relationship("Lesson", foreign_keys=[lesson_id], lazy="selectin")

    __table_args__ = (
        Index(
            "ix_lesson_access_student_lesson", "student_id", "lesson_id", unique=True
        ),
    )

    def __repr__(self) -> str:
        return f"<LessonAccess student={self.student_id} lesson={self.lesson_id}>"
