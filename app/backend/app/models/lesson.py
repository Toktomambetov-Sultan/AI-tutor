import uuid

from sqlalchemy import ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class Lesson(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "lessons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    class_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("classes.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    # Relationships
    class_ = relationship(
        "Class", back_populates="lessons", foreign_keys=[class_id], lazy="selectin"
    )
    materials = relationship(
        "Material",
        back_populates="lesson",
        foreign_keys="[Material.lesson_id]",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Lesson {self.title}>"
