import uuid

from sqlalchemy import ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class Course(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    teacher_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    teacher = relationship(
        "User", back_populates="courses", foreign_keys=[teacher_id], lazy="selectin"
    )
    classes = relationship(
        "Class",
        back_populates="course",
        foreign_keys="[Class.course_id]",
        lazy="selectin",
        order_by="Class.order",
    )

    def __repr__(self) -> str:
        return f"<Course {self.title}>"
