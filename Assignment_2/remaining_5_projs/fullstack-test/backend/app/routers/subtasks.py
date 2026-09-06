from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas import SubtaskRead, SubtaskCreate, SubtaskUpdate
from backend.app import crud

router = APIRouter(prefix="/api", tags=["Subtasks"])

@router.post("/tasks/{task_id}/subtasks", response_model=SubtaskRead, status_code=status.HTTP_201_CREATED)
def add_subtask(task_id: int, subtask_in: SubtaskCreate, db: Session = Depends(get_db)):
    subtask = crud.create_subtask(db, task_id, subtask_in)
    if not subtask:
        raise HTTPException(status_code=404, detail=f"Task with id {task_id} not found")
    return subtask

@router.put("/subtasks/{subtask_id}", response_model=SubtaskRead)
def update_subtask(subtask_id: int, subtask_in: SubtaskUpdate, db: Session = Depends(get_db)):
    subtask = crud.update_subtask(db, subtask_id, subtask_in)
    if not subtask:
        raise HTTPException(status_code=404, detail=f"Subtask with id {subtask_id} not found")
    return subtask

@router.delete("/subtasks/{subtask_id}", status_code=status.HTTP_200_OK)
def remove_subtask(subtask_id: int, db: Session = Depends(get_db)):
    success = crud.delete_subtask(db, subtask_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Subtask with id {subtask_id} not found")
    return {"message": f"Subtask {subtask_id} deleted successfully", "id": subtask_id}
