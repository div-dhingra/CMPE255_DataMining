from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas import AnalyticsResponse
from backend.app import crud

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("", response_model=AnalyticsResponse)
def get_productivity_analytics(db: Session = Depends(get_db)):
    return crud.get_analytics(db)
