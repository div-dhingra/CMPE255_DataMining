from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas import (
    TaskRead, TaskCreate, TaskUpdate, TaskReorderRequest
)
from backend.app import crud

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])

@router.get("", response_model=List[TaskRead])
def list_tasks(
    search: Optional[str] = Query(None, description="Search keyword in title, description, or subtasks"),
    status: Optional[str] = Query("all", description="all, active, completed, todo, in_progress"),
    priority: Optional[str] = Query(None, description="P1, P2, P3, P4"),
    tag_id: Optional[int] = Query(None, description="Filter by Tag ID"),
    tag_name: Optional[str] = Query(None, description="Filter by Tag Name"),
    due_filter: Optional[str] = Query(None, description="today, tomorrow, overdue, upcoming, no_date"),
    sort_by: Optional[str] = Query("position", description="position, due_date, priority, title, created_at"),
    order: Optional[str] = Query("asc", description="asc, desc"),
    db: Session = Depends(get_db)
):
    return crud.get_tasks(
        db,
        search=search,
        status=status,
        priority=priority,
        tag_id=tag_id,
        tag_name=tag_name,
        due_filter=due_filter,
        sort_by=sort_by,
        order=order
    )

@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def create_new_task(task_in: TaskCreate, db: Session = Depends(get_db)):
    return crud.create_task(db, task_in)

@router.get("/{task_id}", response_model=TaskRead)
def get_task_by_id(task_id: int, db: Session = Depends(get_db)):
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    return task

@router.put("/{task_id}", response_model=TaskRead)
def update_task_by_id(task_id: int, task_in: TaskUpdate, db: Session = Depends(get_db)):
    task = crud.update_task(db, task_id, task_in)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    return task

@router.post("/{task_id}/toggle", response_model=TaskRead)
def toggle_task(task_id: int, db: Session = Depends(get_db)):
    task = crud.toggle_task_status(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    return task

@router.delete("/{task_id}", status_code=status.HTTP_200_OK)
def remove_task(task_id: int, db: Session = Depends(get_db)):
    success = crud.delete_task(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    return {"message": f"Task {task_id} deleted successfully", "id": task_id}

@router.post("/reorder", status_code=status.HTTP_200_OK)
def reorder_task_list(payload: TaskReorderRequest, db: Session = Depends(get_db)):
    crud.reorder_tasks(db, payload.orders)
    return {"message": "Tasks reordered successfully", "count": len(payload.orders)}
