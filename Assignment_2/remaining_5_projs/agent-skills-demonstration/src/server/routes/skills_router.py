"""FastAPI Router for Skills Catalog and Execution."""

from fastapi import APIRouter, HTTPException
from src.skills_registry import execute_skill_by_id, get_skill_details, list_all_skills

router = APIRouter(prefix="/api/skills", tags=["Skills Catalog"])


@router.get("")
def get_skills_catalog():
    """List all 46 skills registered across Agent-ML and Data-Analytics categories."""
    return {"total": 46, "skills": list_all_skills()}


@router.get("/{skill_id}")
def get_skill(skill_id: str):
    """Retrieve full details and raw Markdown specification for a specific skill."""
    details = get_skill_details(skill_id)
    if not details:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found.")
    return details


@router.post("/{skill_id}/run")
def run_skill(skill_id: str):
    """Execute a single skill on demand and return structured analytical outputs."""
    try:
        result = execute_skill_by_id(skill_id)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing skill: {str(e)}")
