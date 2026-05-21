from fastapi import APIRouter, Depends

from backend.app.api.auth import require_api_key
from backend.app.schemas.research import ResearchRequest, ResearchResponse
from backend.app.services.research_service import run_research_pipeline


router = APIRouter(tags=["research"], dependencies=[Depends(require_api_key)])


@router.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest) -> ResearchResponse:
    return run_research_pipeline(payload)
