import uuid
import enum

from sqlalchemy import Boolean, Enum, Index, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class UserRole(str, enum.Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"


class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true")
    )

    # Relationships
    courses = relationship(
        "Course",
        back_populates="teacher",
        foreign_keys="[Course.teacher_id]",
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "ix_users_email_active",
            "email",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role.value})>"
