from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, desc, asc

from backend.app.models import Task, Subtask, Tag, task_tags
from backend.app.schemas import (
    TaskCreate, TaskUpdate, TaskReorderItem,
    SubtaskCreate, SubtaskUpdate,
    TagCreate
)

def parse_iso_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    clean_str = date_str.strip()
    try:
        if "T" in clean_str:
            return datetime.fromisoformat(clean_str.replace("Z", "")).date()
        return date.fromisoformat(clean_str[:10])
    except Exception:
        return None

def calculate_relative_due(due_date_str: Optional[str], is_completed: bool) -> tuple[Optional[str], bool]:
    if not due_date_str:
        return None, False
    d = parse_iso_date(due_date_str)
    if not d:
        return None, False

    today = date.today()
    delta = (d - today).days

    if is_completed:
        if delta == 0:
            return "Due was Today", False
        elif delta == -1:
            return "Due was Yesterday", False
        elif delta < 0:
            return f"Due {abs(delta)}d ago", False
        elif delta == 1:
            return "Due Tomorrow", False
        else:
            return f"Due {d.strftime('%b %d')}", False

    if delta < 0:
        days_ago = abs(delta)
        return (f"Overdue ({days_ago}d ago)" if days_ago > 1 else "Overdue (Yesterday)"), True
    elif delta == 0:
        return "Today", False
    elif delta == 1:
        return "Tomorrow", False
    elif 1 < delta <= 7:
        return f"In {delta} days", False
    else:
        return d.strftime("%b %d"), False

def enrich_task_data(task: Task) -> dict:
    subtasks = sorted(task.subtasks, key=lambda s: s.position)
    subtask_total = len(subtasks)
    subtask_completed = sum(1 for s in subtasks if s.is_completed)
    progress_pct = int((subtask_completed / subtask_total) * 100) if subtask_total > 0 else (100 if task.status == "completed" else 0)

    rel_due, is_overdue = calculate_relative_due(task.due_date, task.status == "completed")

    return {
        "id": task.id,
        "title": task.title,
        "description": task.description or "",
        "priority": task.priority,
        "status": task.status,
        "due_date": task.due_date,
        "position": task.position,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "completed_at": task.completed_at,
        "tags": [{"id": t.id, "name": t.name, "color": t.color, "task_count": len(t.tasks)} for t in task.tags],
        "subtasks": [
            {"id": s.id, "task_id": s.task_id, "title": s.title, "is_completed": s.is_completed, "position": s.position, "created_at": s.created_at}
            for s in subtasks
        ],
        "subtask_total": subtask_total,
        "subtask_completed": subtask_completed,
        "progress_pct": progress_pct,
        "relative_due": rel_due,
        "is_overdue": is_overdue
    }

# Task operations
def get_tasks(
    db: Session,
    search: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    tag_id: Optional[int] = None,
    tag_name: Optional[str] = None,
    due_filter: Optional[str] = None,
    sort_by: Optional[str] = "position",
    order: Optional[str] = "asc"
) -> List[dict]:
    query = db.query(Task)

    # Status filter
    if status:
        status_clean = status.strip().lower()
        if status_clean == "active":
            query = query.filter(Task.status.in_(["todo", "in_progress"]))
        elif status_clean == "completed":
            query = query.filter(Task.status == "completed")
        elif status_clean != "all":
            query = query.filter(Task.status == status_clean)

    # Priority filter
    if priority and priority.upper() in ["P1", "P2", "P3", "P4"]:
        query = query.filter(Task.priority == priority.upper())

    # Tag filters
    if tag_id:
        query = query.filter(Task.tags.any(Tag.id == tag_id))
    elif tag_name:
        query = query.filter(Task.tags.any(Tag.name.ilike(tag_name.strip())))

    # Search filter
    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Task.title.ilike(s),
                Task.description.ilike(s),
                Task.subtasks.any(Subtask.title.ilike(s))
            )
        )

    # Sorting
    if sort_by == "priority":
        # P1, P2, P3, P4 ordering
        col = Task.priority
        query = query.order_by(asc(col) if order == "asc" else desc(col))
    elif sort_by == "due_date":
        col = Task.due_date
        query = query.order_by(asc(col) if order == "asc" else desc(col))
    elif sort_by == "title":
        col = Task.title
        query = query.order_by(asc(col) if order == "asc" else desc(col))
    elif sort_by == "created_at":
        col = Task.created_at
        query = query.order_by(asc(col) if order == "asc" else desc(col))
    else:
        # Default: position asc, then created_at desc
        query = query.order_by(asc(Task.position), desc(Task.created_at))

    tasks = query.all()
    enriched = [enrich_task_data(t) for t in tasks]

    # Date filter in memory to handle relative date logic accurately
    if due_filter:
        df = due_filter.lower().strip()
        today_iso = date.today().isoformat()
        tomorrow_iso = (date.today() + timedelta(days=1)).isoformat()

        if df == "today":
            enriched = [t for t in enriched if t["due_date"] and t["due_date"][:10] == today_iso]
        elif df == "tomorrow":
            enriched = [t for t in enriched if t["due_date"] and t["due_date"][:10] == tomorrow_iso]
        elif df == "overdue":
            enriched = [t for t in enriched if t["is_overdue"]]
        elif df == "upcoming":
            enriched = [t for t in enriched if t["due_date"] and t["due_date"][:10] >= today_iso and not t["is_overdue"]]
        elif df == "no_date":
            enriched = [t for t in enriched if not t["due_date"]]

    return enriched

def get_task(db: Session, task_id: int) -> Optional[dict]:
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return None
    return enrich_task_data(task)

def create_task(db: Session, task_in: TaskCreate) -> dict:
    max_pos = db.query(func.max(Task.position)).scalar()
    new_pos = (max_pos + 1) if max_pos is not None else 0

    task = Task(
        title=task_in.title,
        description=task_in.description or "",
        priority=task_in.priority or "P3",
        status=task_in.status or "todo",
        due_date=task_in.due_date,
        position=new_pos,
        completed_at=datetime.utcnow() if task_in.status == "completed" else None
    )
    db.add(task)
    db.flush()

    # Attach existing tags by ID
    if task_in.tag_ids:
        tags = db.query(Tag).filter(Tag.id.in_(task_in.tag_ids)).all()
        task.tags.extend(tags)

    # Attach/create new tags by name
    if task_in.tag_names:
        for tname in task_in.tag_names:
            tname_clean = tname.strip()
            if not tname_clean:
                continue
            existing = db.query(Tag).filter(Tag.name.ilike(tname_clean)).first()
            if existing:
                if existing not in task.tags:
                    task.tags.append(existing)
            else:
                new_tag = Tag(name=tname_clean, color="#6366f1")
                db.add(new_tag)
                db.flush()
                task.tags.append(new_tag)

    # Initial subtasks
    if task_in.subtasks:
        for idx, st_title in enumerate(task_in.subtasks):
            st_clean = st_title.strip()
            if st_clean:
                st = Subtask(task_id=task.id, title=st_clean, is_completed=False, position=idx)
                task.subtasks.append(st)
    db.commit()
    db.refresh(task)
    return enrich_task_data(task)

def update_task(db: Session, task_id: int, task_in: TaskUpdate) -> Optional[dict]:
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return None

    update_data = task_in.model_dump(exclude_unset=True)

    # Handle completion timestamp
    if "status" in update_data:
        new_status = update_data["status"]
        if new_status == "completed" and task.status != "completed":
            task.completed_at = datetime.utcnow()
        elif new_status != "completed":
            task.completed_at = None

    if "tag_ids" in update_data:
        tag_ids = update_data.pop("tag_ids")
        if tag_ids is not None:
            tags = db.query(Tag).filter(Tag.id.in_(tag_ids)).all()
            task.tags = tags

    for key, value in update_data.items():
        setattr(task, key, value)

    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return enrich_task_data(task)

def toggle_task_status(db: Session, task_id: int) -> Optional[dict]:
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return None

    if task.status == "completed":
        task.status = "todo"
        task.completed_at = None
    else:
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        # Mark all subtasks as completed too
        for sub in task.subtasks:
            sub.is_completed = True

    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    return enrich_task_data(task)

def delete_task(db: Session, task_id: int) -> bool:
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return False
    db.delete(task)
    db.commit()
    return True

def reorder_tasks(db: Session, orders: List[TaskReorderItem]) -> bool:
    for item in orders:
        db.query(Task).filter(Task.id == item.id).update({"position": item.position})
    db.commit()
    return True

# Subtask operations
def create_subtask(db: Session, task_id: int, subtask_in: SubtaskCreate) -> Optional[dict]:
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return None

    max_pos = db.query(func.max(Subtask.position)).filter(Subtask.task_id == task_id).scalar()
    new_pos = (max_pos + 1) if max_pos is not None else 0

    subtask = Subtask(
        task_id=task_id,
        title=subtask_in.title,
        position=subtask_in.position if subtask_in.position is not None else new_pos,
        is_completed=False
    )
    db.add(subtask)
    db.commit()
    db.refresh(subtask)
    return {
        "id": subtask.id,
        "task_id": subtask.task_id,
        "title": subtask.title,
        "is_completed": subtask.is_completed,
        "position": subtask.position,
        "created_at": subtask.created_at
    }

def update_subtask(db: Session, subtask_id: int, subtask_in: SubtaskUpdate) -> Optional[dict]:
    subtask = db.query(Subtask).filter(Subtask.id == subtask_id).first()
    if not subtask:
        return None

    update_data = subtask_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(subtask, key, value)

    db.commit()
    db.refresh(subtask)
    return {
        "id": subtask.id,
        "task_id": subtask.task_id,
        "title": subtask.title,
        "is_completed": subtask.is_completed,
        "position": subtask.position,
        "created_at": subtask.created_at
    }

def delete_subtask(db: Session, subtask_id: int) -> bool:
    subtask = db.query(Subtask).filter(Subtask.id == subtask_id).first()
    if not subtask:
        return False
    db.delete(subtask)
    db.commit()
    return True

# Tag operations
def get_tags(db: Session) -> List[dict]:
    tags = db.query(Tag).order_by(Tag.name.asc()).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "color": t.color,
            "created_at": t.created_at,
            "task_count": len(t.tasks)
        }
        for t in tags
    ]

def create_tag(db: Session, tag_in: TagCreate) -> dict:
    existing = db.query(Tag).filter(Tag.name.ilike(tag_in.name.strip())).first()
    if existing:
        return {
            "id": existing.id,
            "name": existing.name,
            "color": existing.color,
            "created_at": existing.created_at,
            "task_count": len(existing.tasks)
        }
    tag = Tag(name=tag_in.name.strip(), color=tag_in.color or "#6366f1")
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return {
        "id": tag.id,
        "name": tag.name,
        "color": tag.color,
        "created_at": tag.created_at,
        "task_count": 0
    }

def delete_tag(db: Session, tag_id: int) -> bool:
    tag = db.query(Tag).filter(Tag.id == tag_id).first()
    if not tag:
        return False
    db.delete(tag)
    db.commit()
    return True

# Analytics calculations
def get_analytics(db: Session) -> dict:
    tasks = db.query(Task).all()
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.status == "completed")
    active_tasks = total_tasks - completed_tasks
    completion_rate = round((completed_tasks / total_tasks * 100), 1) if total_tasks > 0 else 0.0

    today = date.today()
    today_iso = today.isoformat()

    # Overdue count & due today count
    overdue_count = 0
    due_today_count = 0
    for t in tasks:
        if t.due_date:
            d = parse_iso_date(t.due_date)
            if d:
                if t.status != "completed" and d < today:
                    overdue_count += 1
                elif d == today:
                    due_today_count += 1

    # Daily Velocity (last 7 days including today)
    velocity_days = []
    completion_dates_set = set()

    for i in range(6, -1, -1):
        day_d = today - timedelta(days=i)
        day_str = day_d.isoformat()
        day_name = day_d.strftime("%a")

        # completions on this day
        completed_on_day = sum(
            1 for t in tasks
            if t.completed_at and t.completed_at.date() == day_d
        )
        # created on this day
        created_on_day = sum(
            1 for t in tasks
            if t.created_at and t.created_at.date() == day_d
        )
        velocity_days.append({
            "date": day_str,
            "day_name": day_name,
            "completed_count": completed_on_day,
            "created_count": created_on_day
        })
        if completed_on_day > 0:
            completion_dates_set.add(day_d)

    # Streak calculation
    all_completion_dates = {
        t.completed_at.date() for t in tasks if t.completed_at is not None
    }

    current_streak = 0
    check_date = today
    # If no completion today yet, check if there was a streak ending yesterday
    if check_date not in all_completion_dates:
        check_date = today - timedelta(days=1)

    while check_date in all_completion_dates:
        current_streak += 1
        check_date -= timedelta(days=1)

    # Longest streak calculation
    sorted_dates = sorted(list(all_completion_dates))
    longest_streak = 0
    if sorted_dates:
        temp_streak = 1
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                temp_streak += 1
            else:
                if temp_streak > longest_streak:
                    longest_streak = temp_streak
                temp_streak = 1
        if temp_streak > longest_streak:
            longest_streak = temp_streak
    if current_streak > longest_streak:
        longest_streak = current_streak

    # Priority breakdown
    priority_counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    for t in tasks:
        p = (t.priority or "P3").upper()
        if p in priority_counts:
            priority_counts[p] += 1
        else:
            priority_counts["P3"] += 1

    # Tag breakdown
    tags = db.query(Tag).all()
    tag_counts = [
        {"id": tag.id, "name": tag.name, "color": tag.color, "count": len(tag.tasks)}
        for tag in tags
    ]
    tag_counts.sort(key=lambda x: x["count"], reverse=True)

    # Productivity Score (0 to 100)
    # 1. Completion Rate component (up to 40 pts)
    score_rate = (completion_rate / 100.0) * 40.0

    # 2. Streak component (up to 25 pts: 5 pts per streak day, max 5 days)
    score_streak = min(current_streak * 5.0, 25.0)

    # 3. Recent Velocity component (up to 25 pts: completions in last 7 days)
    recent_completions = sum(v["completed_count"] for v in velocity_days)
    score_velocity = min(recent_completions * 5.0, 25.0)

    # 4. Consistency baseline (10 pts if active tasks exist and no overdue, else penalized)
    score_bonus = 10.0
    overdue_penalty = overdue_count * 4.0

    raw_score = score_rate + score_streak + score_velocity + score_bonus - overdue_penalty
    productivity_score = max(0, min(100, int(round(raw_score))))
    if total_tasks == 0:
        productivity_score = 0

    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "active_tasks": active_tasks,
        "completion_rate": completion_rate,
        "productivity_score": productivity_score,
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "overdue_tasks": overdue_count,
        "due_today_tasks": due_today_count,
        "daily_velocity": velocity_days,
        "priority_breakdown": priority_counts,
        "tag_breakdown": tag_counts
    }
