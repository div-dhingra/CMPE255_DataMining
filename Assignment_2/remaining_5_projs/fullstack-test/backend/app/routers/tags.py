from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas import TagRead, TagCreate
from backend.app import crud

router = APIRouter(prefix="/api/tags", tags=["Tags"])

@router.get("", response_model=List[TagRead])
def list_tags(db: Session = Depends(get_db)):
    return crud.get_tags(db)

@router.post("", response_model=TagRead, status_code=status.HTTP_201_CREATED)
def create_tag(tag_in: TagCreate, db: Session = Depends(get_db)):
    return crud.create_tag(db, tag_in)

@router.delete("/{tag_id}", status_code=status.HTTP_200_OK)
def delete_tag(tag_id: int, db: Session = Depends(get_db)):
    success = crud.delete_tag(db, tag_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tag with id {tag_id} not found")
    return {"message": f"Tag {tag_id} deleted successfully", "id": tag_id}
