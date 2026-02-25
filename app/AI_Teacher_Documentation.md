# AI Teacher Platform — Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [User Roles & Permissions](#3-user-roles--permissions)
4. [Authentication & Account Management](#4-authentication--account-management)
5. [Feature Modules](#5-feature-modules)
   - 5.1 [Admin Module](#51-admin-module)
   - 5.2 [Teacher Module](#52-teacher-module)
   - 5.3 [Student Module](#53-student-module)
6. [Data Models](#6-data-models)
7. [Soft Delete](#7-soft-delete)
8. [API Reference](#8-api-reference)
9. [Frontend Structure](#9-frontend-structure)
10. [Infrastructure & DevOps](#10-infrastructure--devops)
11. [AI Agent (Stub)](#11-ai-agent-stub)

---

## 1. Project Overview

**AI Teacher** is a web-based e-learning platform where:

- **Teachers** upload course materials (text or PDF) structured as Courses → Classes → Lessons.
- **Students** browse and study those materials at their own pace. All published courses are open to all students by default — no explicit enrollment required.
- **An AI Agent** (currently a stub — not implemented) will eventually allow students to conduct voice/chat calls with an artificial tutor based on the uploaded content.
- **Admins** manage the user base by creating teacher and student accounts via a seed script for the first Admin and the Admin UI for all subsequent users.

### Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Database | PostgreSQL |
| Migrations | Alembic |
| Caching / Sessions | Redis |
| Frontend | React (SPA) |
| Containerization | Docker / Docker Compose |

### Key Design Decisions

| Decision | Choice |
|---|---|
| Auth token delivery | `Authorization: Bearer <token>` header |
| PDF file storage | Local filesystem (`UPLOAD_DIR`); path stored in DB |
| Teacher account creation | Admin only — no self-registration for teachers |
| First Admin account | Seed script run on first deployment |
| Course access | Open to all active students by default |
| Deactivated teacher courses | Still accessible to students |
| Student visibility for teachers | Only students who accessed their own courses |
| Deleted records | Soft delete (`deleted_at`) — data preserved, recoverable |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        Client                           │
│                   React SPA (Browser)                   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS / REST API
                         │ Authorization: Bearer <token>
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                       │
│   Auth  │  Admin  │  Teacher  │  Student  │  AI Agent   │
└───┬─────────┬──────────────────────────────────┬────────┘
    │         │                                  │
┌───▼───┐ ┌──▼──────────────────────────────┐ ┌─▼──────┐
│ Redis │ │          PostgreSQL              │ │ AI     │
│ Token │ │  Users, Courses, Classes,       │ │ Agent  │
│ Store │ │  Lessons, Materials             │ │ (stub) │
└───────┘ └─────────────────────────────────┘ └────────┘
                         │
              ┌──────────▼──────────┐
              │   Local Filesystem  │
              │  /uploads/  (PDFs)  │
              └─────────────────────┘
```

All services are orchestrated via **Docker Compose**. The FastAPI backend is the single entry point for all client requests.

---

## 3. User Roles & Permissions

There are three roles in the system. Role assignment is controlled by the Admin.

| Action | Admin | Teacher | Student |
|---|:---:|:---:|:---:|
| Create Teacher accounts | ✅ | ❌ | ❌ |
| Create Student accounts | ✅ | ❌ | ❌ |
| Self-register | ❌ | ❌ | ✅ |
| Soft-delete / restore users | ✅ | ❌ | ❌ |
| Create / Edit / Delete Courses | ❌ | ✅ | ❌ |
| Upload course materials | ❌ | ✅ | ❌ |
| View students who accessed their courses | ❌ | ✅ | ❌ |
| Browse all courses (open access) | ❌ | ❌ | ✅ |
| Access course content | ❌ | ❌ | ✅ |
| Access courses of deactivated teachers | ❌ | ❌ | ✅ |
| Call AI Agent on a lesson | ❌ | ❌ | ✅ (stub) |
| Change own password / username | ✅ | ✅ | ✅ |

---

## 4. Authentication & Account Management

### 4.1 Token Strategy

On login, the server generates a session token (opaque UUID or signed JWT), stores it in **Redis** with a configurable TTL, and returns it to the client in the response body. The client stores the token and attaches it to every subsequent request via the `Authorization` header:

```
Authorization: Bearer <token>
```

On logout, the token is deleted from Redis, immediately invalidating the session regardless of any remaining TTL.

### 4.2 Endpoints Summary

| Method | Endpoint | Who | Description |
|---|---|---|---|
| `POST` | `/auth/register` | Student (public) | Self-registration |
| `POST` | `/auth/login` | All roles | Obtain Bearer token |
| `POST` | `/auth/logout` | All roles | Invalidate token in Redis |
| `PATCH` | `/auth/change-password` | All roles | Change own password |
| `PATCH` | `/auth/change-username` | All roles | Change own username |

### 4.3 Registration (Students Only)

Students may self-register via a public endpoint. Required fields:

- Full name
- Email address (must be unique among non-deleted users)
- Password (minimum 8 characters)

Teacher and Admin accounts **cannot** be self-registered. They are created exclusively by an Admin (see Section 5.1).

### 4.4 Login Flow

1. Client sends `email` + `password` to `POST /auth/login`.
2. Backend looks up the user by email, verifies `deleted_at IS NULL` and `is_active = TRUE`.
3. Backend verifies the password hash.
4. On success, a token is generated, stored in Redis as `session:<token> → { user_id, role }`, and returned in the response body.
5. All subsequent requests include `Authorization: Bearer <token>`.
6. On `POST /auth/logout`, the Redis key is deleted.

### 4.5 Password & Username Change

Both operations require an authenticated request. Password change additionally requires the user's current password for verification.

---

## 5. Feature Modules

### 5.1 Admin Module

Admins manage the user base. They do **not** interact with course content.

**Capabilities:**

- Create Teacher and Student accounts (name, email, temporary password)
- View a list of all users, filterable by role and by soft-delete status
- Soft-delete and restore user accounts

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/admin/users` | Create a new Teacher or Student |
| `GET` | `/admin/users` | List users (supports `?role=` and `?deleted=true` filters) |
| `GET` | `/admin/users/{user_id}` | Get a single user (including soft-deleted) |
| `DELETE` | `/admin/users/{user_id}` | Soft-delete a user (sets `deleted_at`) |
| `POST` | `/admin/users/{user_id}/restore` | Restore a soft-deleted user |

**Seed Script — First Admin Account**

The first Admin is created by running the seed script on first deployment. It reads credentials from environment variables and inserts the Admin user directly into PostgreSQL.

```bash
# Run inside the backend container
docker compose exec backend python scripts/seed_admin.py
```

Required environment variables for the seed script:

```
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change_me_on_first_login
ADMIN_FULL_NAME="Platform Admin"
```

---

### 5.2 Teacher Module

Teachers are the primary content creators on the platform.

#### Content Hierarchy

```
Course
 └── Class  (a logical grouping within a course, e.g., "Week 1")
      └── Lesson  (a single study unit)
           └── Material  (text content or uploaded PDF file)
```

All DELETE operations on content use **soft delete** — records are marked with `deleted_at` and hidden from student-facing queries rather than being permanently removed (see Section 7).

#### Course CRUD

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/courses` | Create a new course |
| `GET` | `/courses` | List own courses (excludes soft-deleted by default) |
| `GET` | `/courses/{course_id}` | Get course details |
| `PUT` | `/courses/{course_id}` | Update course metadata |
| `DELETE` | `/courses/{course_id}` | Soft-delete course (cascades to classes, lessons, materials) |

#### Class & Lesson Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/courses/{course_id}/classes` | Add a class to a course |
| `PUT` | `/classes/{class_id}` | Update class details |
| `DELETE` | `/classes/{class_id}` | Soft-delete a class (cascades to its lessons and materials) |
| `POST` | `/classes/{class_id}/lessons` | Add a lesson to a class |
| `PUT` | `/lessons/{lesson_id}` | Update lesson details |
| `DELETE` | `/lessons/{lesson_id}` | Soft-delete a lesson (cascades to its materials) |

#### Material Upload

Materials are attached to Lessons. Supported formats: **plain text** and **PDF**.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/lessons/{lesson_id}/materials` | Upload a material (`multipart/form-data` for PDF; JSON body for text) |
| `GET` | `/lessons/{lesson_id}/materials` | List materials for a lesson |
| `DELETE` | `/materials/{material_id}` | Soft-delete a material |

PDF files are stored under the path configured by `UPLOAD_DIR`. The `file_path` column in the database holds the path relative to `UPLOAD_DIR`. The physical file is **not** deleted when a material is soft-deleted — it is cleaned up only during a scheduled hard-delete maintenance pass.

#### Student List

Teachers can view students who have accessed content within their courses.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/courses/{course_id}/students` | List students who accessed content in a specific course |
| `GET` | `/teacher/students` | List all such students across all the teacher's own courses |

---

### 5.3 Student Module

All courses are open to all active students. No enrollment step is required.

#### Browsing & Access

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/courses` | Browse all available courses (excludes soft-deleted) |
| `GET` | `/courses/{course_id}` | View course structure (classes + lessons) |
| `GET` | `/lessons/{lesson_id}` | View lesson content and materials |
| `GET` | `/materials/{material_id}` | Download or view a material file |

Courses belonging to soft-deleted or deactivated teachers remain visible and accessible to students. Only content that has itself been soft-deleted is hidden.

#### AI Agent Call

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/lessons/{lesson_id}/agent/call` | Initiate an AI tutoring session (stub — returns `501 Not Implemented`) |

---

## 6. Data Models

All soft-deletable tables carry `deleted_at` and `deleted_by` columns (see Section 7).

### users

```
users
├── id             UUID, PK
├── full_name      VARCHAR(255), NOT NULL
├── email          VARCHAR(255), NOT NULL
├── password_hash  VARCHAR(255), NOT NULL
├── role           ENUM('admin', 'teacher', 'student'), NOT NULL
├── is_active      BOOLEAN, DEFAULT TRUE
├── created_at     TIMESTAMPTZ, NOT NULL
├── updated_at     TIMESTAMPTZ
├── deleted_at     TIMESTAMPTZ, NULLABLE         ← soft delete
└── deleted_by     UUID, FK → users.id, NULLABLE

UNIQUE INDEX: (email) WHERE deleted_at IS NULL
```

### courses

```
courses
├── id             UUID, PK
├── teacher_id     UUID, FK → users.id, NOT NULL
├── title          VARCHAR(255), NOT NULL
├── description    TEXT
├── created_at     TIMESTAMPTZ, NOT NULL
├── updated_at     TIMESTAMPTZ
├── deleted_at     TIMESTAMPTZ, NULLABLE         ← soft delete
└── deleted_by     UUID, FK → users.id, NULLABLE
```

### classes

```
classes
├── id             UUID, PK
├── course_id      UUID, FK → courses.id, NOT NULL
├── title          VARCHAR(255), NOT NULL
├── order          INTEGER, NOT NULL, DEFAULT 0
├── created_at     TIMESTAMPTZ, NOT NULL
├── deleted_at     TIMESTAMPTZ, NULLABLE         ← soft delete
└── deleted_by     UUID, FK → users.id, NULLABLE
```

### lessons

```
lessons
├── id             UUID, PK
├── class_id       UUID, FK → classes.id, NOT NULL
├── title          VARCHAR(255), NOT NULL
├── order          INTEGER, NOT NULL, DEFAULT 0
├── created_at     TIMESTAMPTZ, NOT NULL
├── deleted_at     TIMESTAMPTZ, NULLABLE         ← soft delete
└── deleted_by     UUID, FK → users.id, NULLABLE
```

### materials

```
materials
├── id             UUID, PK
├── lesson_id      UUID, FK → lessons.id, NOT NULL
├── type           ENUM('text', 'pdf'), NOT NULL
├── content        TEXT, NULLABLE          ← used when type = 'text'
├── file_path      VARCHAR(512), NULLABLE  ← used when type = 'pdf'
├── created_at     TIMESTAMPTZ, NOT NULL
├── deleted_at     TIMESTAMPTZ, NULLABLE         ← soft delete
└── deleted_by     UUID, FK → users.id, NULLABLE
```

### lesson_access *(powers teacher student-list)*

Tracks which students opened which lessons, used to populate the teacher's student list view.

```
lesson_access
├── id             UUID, PK
├── student_id     UUID, FK → users.id, NOT NULL
├── lesson_id      UUID, FK → lessons.id, NOT NULL
└── accessed_at    TIMESTAMPTZ, NOT NULL

UNIQUE INDEX: (student_id, lesson_id)
```

---

## 7. Soft Delete

### Overview

Rather than issuing SQL `DELETE` statements, soft delete marks a record as deleted by setting `deleted_at` to the current UTC timestamp and recording which user performed the action in `deleted_by`. The record stays in the database and can be restored. All standard queries filter out deleted records automatically.

This applies to: **users**, **courses**, **classes**, **lessons**, **materials**.

### Core Operations

**Deleting a record:**

```sql
UPDATE <table>
SET    deleted_at = NOW(),
       deleted_by = <acting_user_id>
WHERE  id         = <target_id>
  AND  deleted_at IS NULL;
```

**Fetching active records (applied everywhere by default):**

```sql
SELECT * FROM courses
WHERE  teacher_id = <teacher_id>
  AND  deleted_at IS NULL;
```

**Restoring a record:**

```sql
UPDATE <table>
SET    deleted_at = NULL,
       deleted_by = NULL
WHERE  id = <target_id>;
```

### Cascade Soft Delete

When a parent record is soft-deleted, all its children are soft-deleted in the **same database transaction** and receive the **same `deleted_at` timestamp**. This makes cascaded restoration unambiguous — restoring a parent re-activates all children that share the exact same `deleted_at` value.

```
Soft-delete Course
  └── Soft-delete all Classes of that Course
       └── Soft-delete all Lessons of those Classes
            └── Soft-delete all Materials of those Lessons

Soft-delete Class
  └── Soft-delete all Lessons of that Class
       └── Soft-delete all Materials of those Lessons

Soft-delete Lesson
  └── Soft-delete all Materials of that Lesson

Soft-delete User (Admin action — teacher)
  └── Soft-delete all Courses authored by that Teacher (and cascade)
```

### Restore Rules

Restoring a child record is only valid when its parent is **not** soft-deleted. The backend validates the parent's state before restoring and returns `409 Conflict` if the parent is still deleted.

### Admin Visibility

`GET /admin/users?deleted=true` includes soft-deleted records alongside active ones. The Admin UI marks deleted records visually (e.g., greyed out with a "Deleted" badge and a "Restore" action).

### SQLAlchemy Model Mixin

```python
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

class SoftDeleteMixin:
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    deleted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self, acting_user_id: str) -> None:
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = acting_user_id

    def restore(self) -> None:
        self.deleted_at = None
        self.deleted_by = None
```

All soft-deletable models inherit from `SoftDeleteMixin`. Every repository or service query **must** include `.filter(Model.deleted_at.is_(None))` unless explicitly querying deleted records (Admin only).

### Hard Delete Maintenance

A scheduled CLI command permanently removes records that have been soft-deleted for longer than a configurable retention period (default: 90 days). For `materials` with `type = 'pdf'`, the command also deletes the physical file from the filesystem before removing the database row.

```bash
# Run manually or via a scheduled cron job
docker compose exec backend python scripts/hard_delete.py --older-than-days 90
```

The `HARD_DELETE_RETENTION_DAYS` environment variable controls the default retention period.

---

## 8. API Reference

### General Conventions

- Base URL: `/api/v1`
- All request and response bodies use `application/json` unless uploading files (`multipart/form-data`).
- Authentication: `Authorization: Bearer <token>` on all protected endpoints.
- Timestamps use ISO 8601 format (UTC), e.g. `2025-06-01T12:00:00Z`.

### Standard HTTP Status Codes

| Code | Meaning |
|---|---|
| `200 OK` | Successful GET / PUT |
| `201 Created` | Successful POST that created a resource |
| `204 No Content` | Successful soft-delete |
| `400 Bad Request` | Validation error |
| `401 Unauthorized` | Missing or invalid token |
| `403 Forbidden` | Authenticated but insufficient role |
| `404 Not Found` | Resource does not exist or is soft-deleted |
| `409 Conflict` | Restore attempted while parent is still deleted |
| `422 Unprocessable Entity` | FastAPI input validation failure |
| `501 Not Implemented` | AI Agent endpoint (stub) |

### Error Response Format

```json
{
  "detail": "Human-readable error description"
}
```

---

## 9. Frontend Structure

The React application is a Single Page Application (SPA) that communicates exclusively with the FastAPI backend.

### Token Handling

The Bearer token is stored in memory (React Context / Zustand store) or `localStorage`. It is attached to all API calls via an Axios interceptor:

```js
axios.interceptors.request.use(config => {
  const token = getToken(); // from store
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
```

On logout, the token is cleared from client state and `POST /auth/logout` is called to invalidate it server-side.

### Suggested Route Structure

```
/                          → Landing / Login page
/register                  → Student self-registration
/dashboard                 → Role-aware home (redirects based on role)

/admin/users               → User management (Admin)
/admin/users/new           → Create user form
/admin/users/:id           → User detail / restore

/teacher/courses           → Course list (Teacher)
/teacher/courses/new       → Create course
/teacher/courses/:id       → Course detail & class/lesson management
/teacher/students          → Student list

/courses                   → Course browser (Student)
/courses/:id               → Course overview
/courses/:id/lessons/:lessonId  → Lesson viewer
```

### State Management

- Auth state (current user, role, token) is stored in React Context or Zustand.
- Routes are wrapped in role-aware `<ProtectedRoute>` components that redirect unauthenticated or unauthorized users to `/`.

---

## 10. Infrastructure & DevOps

### Docker Compose Services

```yaml
services:
  backend:     # FastAPI application
  frontend:    # React app (served via Nginx or Vite dev server)
  db:          # PostgreSQL
  redis:       # Redis (token store / cache)
```

### First-Run Setup

```bash
# 1. Start all services
docker compose up -d

# 2. Apply database migrations
docker compose exec backend alembic upgrade head

# 3. Seed the first Admin account
docker compose exec backend python scripts/seed_admin.py
```

### Database Migrations

```bash
# Generate a new migration after model changes
alembic revision --autogenerate -m "description"

# Apply pending migrations
alembic upgrade head

# Roll back one step
alembic downgrade -1
```

### Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `SECRET_KEY` | Secret used to sign/verify tokens |
| `SESSION_TTL_SECONDS` | Token lifetime in Redis (e.g., `86400` for 24 h) |
| `UPLOAD_DIR` | Absolute path on the server for storing uploaded PDFs |
| `ALLOWED_ORIGINS` | CORS-allowed origins (comma-separated frontend URLs) |
| `ENVIRONMENT` | `development` or `production` |
| `ADMIN_EMAIL` | Seed script — first Admin email |
| `ADMIN_PASSWORD` | Seed script — first Admin password |
| `ADMIN_FULL_NAME` | Seed script — first Admin display name |
| `HARD_DELETE_RETENTION_DAYS` | Days before soft-deleted records become eligible for hard delete (default: `90`) |

---

## 11. AI Agent (Stub)

The AI Agent feature will allow a student to start a tutoring session tied to a specific lesson, with the agent drawing on that lesson's materials as its knowledge context.

**Current status:** Not implemented. `POST /lessons/{lesson_id}/agent/call` returns `HTTP 501 Not Implemented`.

### Planned Behaviour (Future)

1. Student navigates to a lesson and clicks "Start AI Session."
2. The frontend calls the agent endpoint and receives a session handle.
3. The frontend connects to a real-time channel (WebSocket or WebRTC).
4. The AI agent uses the lesson's non-deleted materials as context and responds to student queries.

### Design Considerations

- The agent must be scoped to the materials of the selected lesson to prevent out-of-scope responses.
- Soft-deleted materials must be excluded from the agent's context.
- Session transcripts may be stored for teacher review.
- LLM provider selection should account for latency, cost, and data-privacy requirements.

---

*Document version: 1.1 — All design decisions resolved*
