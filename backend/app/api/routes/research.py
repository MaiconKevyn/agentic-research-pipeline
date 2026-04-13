from fastapi import APIRouter

from backend.app.schemas.research import ResearchRequest, ResearchResponse
from backend.app.services.research_service import run_research_pipeline


router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest) -> ResearchResponse:
    return run_research_pipeline(payload)
