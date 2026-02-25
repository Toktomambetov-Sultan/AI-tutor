import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=datetime.now(timezone.utc),
    )


class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    deleted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self, acting_user_id: uuid.UUID) -> None:
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = acting_user_id

    def restore(self) -> None:
        self.deleted_at = None
        self.deleted_by = None
