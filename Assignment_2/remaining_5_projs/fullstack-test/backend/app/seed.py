from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from backend.app.models import Task, Subtask, Tag

def seed_initial_data(db: Session):
    if db.query(Task).count() > 0:
        return

    print("Seeding initial demo data for Todo Application...")
    
    # Predefined Tags with elegant colors
    tag_work = Tag(name="Work", color="#6366f1")       # Indigo
    tag_school = Tag(name="School", color="#3b82f6")   # Blue
    tag_eng = Tag(name="Engineering", color="#10b981") # Emerald
    tag_frontend = Tag(name="Frontend", color="#8b5cf6") # Violet
    tag_personal = Tag(name="Personal", color="#ec4899") # Pink
    tag_docs = Tag(name="Docs", color="#f59e0b")       # Amber

    db.add_all([tag_work, tag_school, tag_eng, tag_frontend, tag_personal, tag_docs])
    db.commit()

    today = date.today()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    tomorrow = today + timedelta(days=1)
    in_three_days = today + timedelta(days=3)
    past_due = today - timedelta(days=3)

    # Task 1: Due Today, Urgent P1
    t1 = Task(
        title="Prepare Project Showcase & Live Presentation",
        description="Review all system benchmarks, ensure tests pass cleanly, and present key UX features.",
        priority="P1",
        status="in_progress",
        due_date=today.isoformat(),
        position=0,
        created_at=datetime.utcnow() - timedelta(days=2)
    )
    t1.tags.extend([tag_work, tag_school])
    t1.subtasks.extend([
        Subtask(title="Review slide deck visuals and flow", is_completed=True, position=0),
        Subtask(title="Verify all test assertions pass (100% green)", is_completed=True, position=1),
        Subtask(title="Record quick responsive UX walkthrough", is_completed=False, position=2)
    ])

    # Task 2: Due Tomorrow, High P2
    t2 = Task(
        title="Implement Drag & Drop Reordering and Confetti celebration",
        description="Add smooth drag feedback, persist order changes to `/api/tasks/reorder`, and trigger confetti fireworks on completion.",
        priority="P2",
        status="todo",
        due_date=tomorrow.isoformat(),
        position=1,
        created_at=datetime.utcnow() - timedelta(days=1)
    )
    t2.tags.extend([tag_frontend, tag_eng])
    t2.subtasks.extend([
        Subtask(title="Bind HTML5 dragstart and dragover listeners", is_completed=True, position=0),
        Subtask(title="Implement Web Audio synth chime generator", is_completed=True, position=1),
        Subtask(title="Hook canvas confetti fireworks trigger", is_completed=False, position=2)
    ])

    # Task 3: Overdue Task (active)
    t3 = Task(
        title="Review Database Indexing & Query Optimizations",
        description="Verify query plans for SQLite full-text and tag association tables.",
        priority="P2",
        status="todo",
        due_date=past_due.isoformat(),
        position=2,
        created_at=datetime.utcnow() - timedelta(days=5)
    )
    t3.tags.extend([tag_eng])
    t3.subtasks.extend([
        Subtask(title="Check index on tasks.position and tasks.status", is_completed=True, position=0),
        Subtask(title="Profile analytics query execution time", is_completed=False, position=1)
    ])

    # Task 4: Completed today (Streak maintenance!)
    t4 = Task(
        title="Build High-Performance FastAPI REST Backend",
        description="Full CRUD endpoints for tasks, subtasks, tags, and productivity analytics.",
        priority="P1",
        status="completed",
        due_date=today.isoformat(),
        position=3,
        created_at=datetime.utcnow() - timedelta(days=1),
        completed_at=datetime.utcnow()
    )
    t4.tags.extend([tag_eng, tag_work])
    t4.subtasks.extend([
        Subtask(title="Setup SQLAlchemy models & SQLite engine", is_completed=True, position=0),
        Subtask(title="Implement velocity and streak analytics calculations", is_completed=True, position=1)
    ])

    # Task 5: Completed yesterday (Contributes to streak)
    t5 = Task(
        title="Design Command Palette & Keyboard Shortcuts System",
        description="Global Cmd/Ctrl+K palette modal, quick 'n' to add task, and '/' to search.",
        priority="P3",
        status="completed",
        due_date=yesterday.isoformat(),
        position=4,
        created_at=datetime.utcnow() - timedelta(days=3),
        completed_at=datetime.combine(yesterday, datetime.min.time()) + timedelta(hours=15)
    )
    t5.tags.extend([tag_frontend])
    t5.subtasks.extend([
        Subtask(title="Listen for global window keydown events", is_completed=True, position=0),
        Subtask(title="Style modal backdrop and command list", is_completed=True, position=1)
    ])

    # Task 6: Completed 2 days ago
    t6 = Task(
        title="Setup UI Color Palette & Glassmorphism Theme",
        description="Tailwind CSS theme tokens with high-contrast dark mode support.",
        priority="P3",
        status="completed",
        due_date=two_days_ago.isoformat(),
        position=5,
        created_at=datetime.utcnow() - timedelta(days=4),
        completed_at=datetime.combine(two_days_ago, datetime.min.time()) + timedelta(hours=14)
    )
    t6.tags.extend([tag_frontend, tag_personal])

    # Task 7: Upcoming Task (in 3 days)
    t7 = Task(
        title="Write Comprehensive Automated Pytest Suite",
        description="Cover models, CRUD endpoints, filtering logic, and analytics metrics.",
        priority="P2",
        status="todo",
        due_date=in_three_days.isoformat(),
        position=6,
        created_at=datetime.utcnow()
    )
    t7.tags.extend([tag_eng, tag_docs])
    t7.subtasks.extend([
        Subtask(title="Test Task CRUD endpoints", is_completed=False, position=0),
        Subtask(title="Test Subtasks & Tags associations", is_completed=False, position=1),
        Subtask(title="Test Analytics calculations & streak logic", is_completed=False, position=2)
    ])

    # Task 8: Low Priority Backlog Task
    t8 = Task(
        title="Write User Documentation & Setup Guide",
        description="Document startup scripts, keyboard shortcuts, API endpoints, and project architecture.",
        priority="P4",
        status="todo",
        due_date=None,
        position=7,
        created_at=datetime.utcnow()
    )
    t8.tags.extend([tag_docs])

    db.add_all([t1, t2, t3, t4, t5, t6, t7, t8])
    db.commit()
    print("Demo data seeded successfully.")
