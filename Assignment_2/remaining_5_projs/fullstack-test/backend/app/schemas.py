from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

# Tag Schemas
class TagBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    color: Optional[str] = Field("#6366f1", max_length=20)

class TagCreate(TagBase):
    pass

class TagRead(TagBase):
    id: int
    created_at: Optional[datetime] = None
    task_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)

# Subtask Schemas
class SubtaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    is_completed: bool = False
    position: int = 0

class SubtaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    position: Optional[int] = 0

class SubtaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    is_completed: Optional[bool] = None
    position: Optional[int] = None

class SubtaskRead(SubtaskBase):
    id: int
    task_id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# Task Schemas
class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = ""
    priority: Optional[str] = Field("P3", description="P1, P2, P3, P4")
    status: Optional[str] = Field("todo", description="todo, in_progress, completed")
    due_date: Optional[str] = None
    position: Optional[int] = 0

class TaskCreate(TaskBase):
    tag_ids: Optional[List[int]] = []
    tag_names: Optional[List[str]] = []
    subtasks: Optional[List[str]] = []

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[str] = None
    position: Optional[int] = None
    tag_ids: Optional[List[int]] = None

class TaskReorderItem(BaseModel):
    id: int
    position: int

class TaskReorderRequest(BaseModel):
    orders: List[TaskReorderItem]

class TaskRead(TaskBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: List[TagRead] = []
    subtasks: List[SubtaskRead] = []
    subtask_total: int = 0
    subtask_completed: int = 0
    progress_pct: int = 0
    relative_due: Optional[str] = None
    is_overdue: bool = False

    model_config = ConfigDict(from_attributes=True)

# Analytics Schemas
class DailyVelocityItem(BaseModel):
    date: str
    day_name: str
    completed_count: int
    created_count: int

class TagBreakdownItem(BaseModel):
    id: int
    name: str
    color: str
    count: int

class AnalyticsResponse(BaseModel):
    total_tasks: int
    completed_tasks: int
    active_tasks: int
    completion_rate: float
    productivity_score: int
    current_streak: int
    longest_streak: int
    overdue_tasks: int
    due_today_tasks: int
    daily_velocity: List[DailyVelocityItem]
    priority_breakdown: Dict[str, int]
    tag_breakdown: List[TagBreakdownItem]
