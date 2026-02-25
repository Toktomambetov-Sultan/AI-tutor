from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.redis import redis_client
from app.routers import auth, admin, courses, teacher, student, agent


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await redis_client.aclose()


app = FastAPI(
    title="AI Teacher Platform",
    description="E-learning platform with AI tutoring capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /api/v1
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(courses.router, prefix="/api/v1", tags=["Courses"])
app.include_router(teacher.router, prefix="/api/v1/teacher", tags=["Teacher"])
app.include_router(student.router, prefix="/api/v1", tags=["Student"])
app.include_router(agent.router, prefix="/api/v1", tags=["AI Agent"])


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
