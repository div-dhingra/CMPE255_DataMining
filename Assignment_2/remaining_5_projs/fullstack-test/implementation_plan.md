# Implementation Plan

## Overview
This document serves as the retroactive implementation plan for the **TodoPro** application, an end-to-end task and project management web application. It outlines the built architecture, technical stack, key components, and automated testing strategy that have been implemented.

---

## 🏛 Built Architecture

The system follows a modern decoupled client-server architecture but runs from a unified monolithic repository for simplicity.

1. **Frontend**: A single-page application (SPA) entirely driven by Vanilla JavaScript, HTML5, and CSS3. State management is handled through ES6 classes (`TodoApp`), avoiding the overhead of heavy frameworks like React while remaining responsive.
2. **Backend**: A RESTful JSON API built with FastAPI in Python, providing asynchronous, high-performance endpoint handling.
3. **Database**: A local SQLite database (`todos.db`) managed via SQLAlchemy ORM, ensuring ACID compliance and straightforward deployment without additional dependencies.

---

## 💻 Technical Stack

### Backend
- **Python 3.10+**: Core programming language.
- **FastAPI**: Asynchronous web framework used for rapid API development, routing, and automatic Swagger/OpenAPI documentation.
- **SQLAlchemy (ORM)**: Translates Python models into relational schemas, managing queries and persistence.
- **Pydantic**: Enforces data validation and serialization across API boundaries.
- **Uvicorn**: ASGI web server implementation for FastAPI.
- **Pytest**: Framework for executing the automated backend testing suite.

### Frontend
- **HTML5 & CSS3**: Native DOM structures styled with utility-focused classes, supporting dark/light mode transitions and responsive design.
- **Vanilla JavaScript (ES6+)**: Handles frontend state, asynchronous `fetch` calls, and complex UI interactions (drag-and-drop, modals).
- **Web Audio API & HTML Canvas**: Used for celebratory sound synthesis (zero MP3 dependencies) and confetti particle generation on task completion.

---

## 🧩 Key Components

### 1. `backend/app/main.py`
The API entry point. It configures the FastAPI application, mounts static frontend files, defines database initialization logic on startup, and aggregates the sub-routers.

### 2. `backend/app/models.py` & `backend/app/schemas.py`
- **Models**: SQLAlchemy ORM definitions including `Task`, `Subtask`, and `Tag`, alongside a many-to-many relationship table `task_tags`.
- **Schemas**: Pydantic v2 schemas (`TaskCreate`, `TaskUpdate`, etc.) ensuring robust input validation before data touches the database.

### 3. `backend/app/crud.py`
The core business logic layer. Encapsulates all database interactions to keep routers lean. It manages multi-criteria search filtering, nested relationships (subtask progress), and complex aggregations (such as the productivity score algorithm and 7-day velocity chart data).

### 4. `backend/app/routers/`
Modular route controllers organized by feature:
- `tasks.py`: CRUD operations for tasks, toggling completion, and drag-and-drop reordering.
- `subtasks.py`: Managing nested task items.
- `tags.py`: Managing categories and task tagging.
- `analytics.py`: Serving data for the productivity dashboard.

### 5. `frontend/js/app.js`
The primary client-side application controller. It maintains the UI state, binds event listeners (including keyboard shortcuts and drag-and-drop events), communicates with the REST API via `fetch`, and dynamically renders DOM updates.

---

## 🧪 Testing Strategy

The backend includes a comprehensive automated test suite leveraging `pytest` with an in-memory SQLite database to ensure clean test isolation.

- **`test_tasks.py`**: Verifies primary CRUD operations, complex querying and filtering logic (e.g. status, priority, and date filters), and the batch position reordering logic.
- **`test_subtasks.py`**: Ensures subtasks can be dynamically added, updated, deleted, and that their completion accurately propagates to the parent task's progress calculations.
- **`test_tags.py`**: Validates tag lifecycle and the many-to-many relationships between tasks and custom colored tags.
- **`test_analytics.py`**: Ensures the analytics engine mathematically derives the correct productivity score, daily velocity, and continuous daily streaks based on historical database records.

**Execution**: The full suite is executed via the included `./run_tests.sh` shell script.
