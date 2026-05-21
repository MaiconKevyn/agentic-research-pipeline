from fastapi import APIRouter, Depends, Query

from backend.app.api.auth import require_api_key
from backend.app.schemas.research import RunMetricsSummary
from backend.app.services.document_repository import get_run_metrics_summary


router = APIRouter(tags=["operations"], dependencies=[Depends(require_api_key)])


@router.get("/ops/run-metrics", response_model=RunMetricsSummary)
def get_run_metrics(days: int = Query(default=30, ge=1, le=365)) -> RunMetricsSummary:
    return get_run_metrics_summary(days=days)
