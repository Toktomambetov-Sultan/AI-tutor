from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://ai_teacher:ai_teacher_pass@db:5432/ai_teacher"
    )

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Auth
    SECRET_KEY: str = "change-me"
    SESSION_TTL_SECONDS: int = 86400

    # File uploads
    UPLOAD_DIR: str = "/app/uploads"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Environment
    ENVIRONMENT: str = "development"

    # Seed Admin
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "change_me_on_first_login"
    ADMIN_FULL_NAME: str = "Platform Admin"

    # Hard Delete
    HARD_DELETE_RETENTION_DAYS: int = 90

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
