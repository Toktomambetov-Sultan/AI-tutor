import uuid

from sqlalchemy import ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class Class(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "classes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    # Relationships
    course = relationship(
        "Course", back_populates="classes", foreign_keys=[course_id], lazy="selectin"
    )
    lessons = relationship(
        "Lesson",
        back_populates="class_",
        foreign_keys="[Lesson.class_id]",
        lazy="selectin",
        order_by="Lesson.order",
    )

    def __repr__(self) -> str:
        return f"<Class {self.title}>"
