# TodoPro — Modern Fullstack Dynamic Task Management Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Pytest](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)](https://docs.pytest.org/)

**TodoPro** is an end-to-end task and project management web application delivering an industry-leading user experience (UX), rich feature set, and real-time productivity analytics.

---

## 🌟 Key Features

### 1. Dynamic Task Management
- **Full CRUD Operations**: Create, read, update, and delete tasks with instant responsive UI updates.
- **Drag-and-Drop Reordering**: Native HTML5 drag-and-drop handles (`⋮⋮`) with immediate visual feedback and persistent database ordering synced via `/api/tasks/reorder`.
- **Quick Add Bar**: One-click inline task entry with keyboard `Enter` submission.

### 2. Rich Task Attributes
- **Title & Markdown Description**: Clean rich notes and detailed task context.
- **Priority Badging (P1 to P4)**:
  - `P1 Urgent`: Red glow badge
  - `P2 High`: Amber badge
  - `P3 Medium`: Indigo badge
  - `P4 Low`: Subtle gray badge
- **Smart Relative Due Dates**: Automatic real-time status badges:
  - `Overdue (Xd ago)`: Pulsing high-visibility warning badge
  - `Today`: Vibrant emerald badge
  - `Tomorrow`: Informative blue badge
  - `Upcoming / In X days`
- **Dynamic Tags & Labels**: Categorize tasks by custom categories (Work, Engineering, Frontend, Personal, etc.) with custom hex color swatches.
- **Interactive Subtask Checklists**:
  - Live progress meter (`X of Y done`, animated percentage progress bar).
  - Inline checkbox toggle to complete individual checklist items.
  - Add subtasks on the fly directly inside each task card.

### 3. Live Search & Multi-Criteria Filtering
- **Faceted Filters**:
  - By Status: *All*, *Active*, *Completed*
  - By Due Date: *Today*, *Tomorrow*, *Overdue*, *Upcoming*
  - By Priority: *P1*, *P2*, *P3*, *P4*
  - By Category / Tag: One-click tag filter
- **Instant Search**: Sub-millisecond live search (`/` shortcut) across titles, descriptions, and subtasks.
- **Multi-Sort Options**: Sort by manual drag-and-drop position, due date, priority, title, or creation date.

### 4. Productivity Analytics Dashboard
- **Productivity Score (0-100)**: Algorithmic rating combining completion rate, current streak, recent velocity, and on-time vs overdue status.
- **Streak Tracker (`🔥`)**: Continuous daily completion streak tracker with milestone recognition.
- **7-Day Velocity Chart**: Interactive bar chart comparing daily completions vs creations with hover tooltips.
- **Category Distribution**: Color-coded breakdown bar and percentage share across tags.

### 5. State-of-the-Art UX Delights
- **Celebratory Fireworks & Sound**:
  - Multi-colored **Canvas Confetti** burst originating from the completed checkbox.
  - **Harmonic Synthesizer Chime** via Web Audio API (C5-E5-G5-C6 arpeggio) with zero external MP3 file dependencies.
- **Command Palette (`⌘K` / `Ctrl+K`)**: Fast Spotlight/Alfred-like modal to search commands, jump between views, toggle dark mode, or open analytics.
- **Dark / Light Mode**: Smooth theme transitions with persisted user preference.
- **Keyboard-Driven Workflow**:
  - `Cmd+K` / `Ctrl+K`: Command Palette
  - `N`: Create new task
  - `/`: Focus search input
  - `T`: Toggle theme
  - `?`: Open keyboard shortcuts guide
  - `Esc`: Close open modal

---

## 🏛 Architecture & Tech Stack

```
fullstack-test/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint, middleware, static mounting
│   │   ├── database.py        # SQLAlchemy SQLite engine & session management
│   │   ├── models.py          # ORM models (Task, Subtask, Tag, task_tags)
│   │   ├── schemas.py         # Pydantic v2 schemas for validation & serialization
│   │   ├── crud.py            # Business logic, query filters, analytics calculations
│   │   ├── seed.py            # Demo seed data generator
│   │   └── routers/
│   │       ├── tasks.py       # Task CRUD, reordering, toggle endpoints
│   │       ├── subtasks.py    # Subtask lifecycle endpoints
│   │       ├── tags.py        # Tag creation, listing, and deletion
│   │       └── analytics.py   # Productivity metrics and velocity endpoints
│   └── tests/
│       ├── conftest.py        # In-memory SQLite fixtures and TestClient
│       ├── test_tasks.py      # Task CRUD, filters, and drag-and-drop tests
│       ├── test_subtasks.py   # Subtask cascade and progress tests
│       ├── test_tags.py       # Tag management and association tests
│       └── test_analytics.py  # Productivity score and streak tests
├── frontend/
│   ├── index.html             # Single-page application UI template
│   ├── css/
│   │   └── style.css          # Modern animations, scrollbars, drag states
│   └── js/
│       ├── app.js             # Main state management, drag-drop, keyboard events
│       ├── analytics.js       # SVG productivity gauges & velocity bar chart
│       ├── audio.js           # Web Audio API harmonic sound synthesizer
│       └── confetti.js        # Self-contained Canvas confetti particle system
├── run_tests.sh               # Executable test runner
├── run.sh                     # Executable server startup script
├── requirements.txt           # Python dependencies
└── README.md                  # Complete documentation
```

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+ (SQLite3 built-in)
- Dependencies installed in virtual environment (`./venv` or pre-configured environment)

### 2. Running Automated Tests
Run the test suite with a single command:
```bash
./run_tests.sh
```

### 3. Launching the Application
Start the unified fullstack server:
```bash
./run.sh
```
Or directly via Python:
```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. ./venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
- **Application UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/tasks` | List tasks with filters (`search`, `status`, `priority`, `tag_id`, `due_filter`, `sort_by`, `order`) |
| `POST` | `/api/tasks` | Create a new task with tags and subtasks |
| `GET` | `/api/tasks/{id}` | Retrieve task details with subtasks and tags |
| `PUT` | `/api/tasks/{id}` | Update task attributes |
| `DELETE` | `/api/tasks/{id}` | Delete a task (cascades to subtasks) |
| `POST` | `/api/tasks/{id}/toggle` | Toggle task completion status |
| `POST` | `/api/tasks/reorder` | Batch reorder task position indices (drag & drop) |
| `POST` | `/api/tasks/{id}/subtasks` | Add a subtask to a task |
| `PUT` | `/api/subtasks/{id}` | Update subtask completion or title |
| `DELETE` | `/api/subtasks/{id}` | Delete a subtask |
| `GET` | `/api/tags` | List all tags with task counts |
| `POST` | `/api/tags` | Create a new tag with custom color |
| `DELETE` | `/api/tags/{id}` | Delete a tag |
| `GET` | `/api/analytics` | Fetch productivity score, streak, and daily velocity |
| `GET` | `/health` | Service health status check |
