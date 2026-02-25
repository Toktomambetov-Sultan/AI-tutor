from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class CreateUserRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(teacher|student)$")


class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime | None = None
    deleted_at: datetime | None = None

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
