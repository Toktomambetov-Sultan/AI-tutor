import uuid
import enum

from sqlalchemy import Enum, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class MaterialType(str, enum.Enum):
    text = "text"
    pdf = "pdf"


class Material(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "materials"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lessons.id"), nullable=False
    )
    type: Mapped[MaterialType] = mapped_column(
        Enum(MaterialType, name="material_type"), nullable=False
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Relationships
    lesson = relationship(
        "Lesson", back_populates="materials", foreign_keys=[lesson_id], lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Material {self.type.value} for lesson {self.lesson_id}>"
