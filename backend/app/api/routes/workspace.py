from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Response

from backend.app.schemas.research import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackEvalCase,
    RunSourceDetail,
    RunSummary,
    WorkspaceRun,
)
from backend.app.services.document_repository import (
    get_research_run,
    get_research_run_source,
    list_feedback_as_eval_cases,
    list_research_runs,
    record_run_feedback,
)
from backend.app.services.export_service import (
    export_run_csv,
    export_run_json,
    export_run_markdown,
)


router = APIRouter(tags=["workspace"])


@router.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[RunSummary]:
    return list_research_runs(limit=limit)


@router.get("/runs/{run_id}", response_model=WorkspaceRun)
def get_run(run_id: str) -> WorkspaceRun:
    run = get_research_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Research run not found.")
    return run


@router.get("/runs/{run_id}/sources/{source_id:path}", response_model=RunSourceDetail)
def get_run_source(run_id: str, source_id: str) -> RunSourceDetail:
    source = get_research_run_source(run_id, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Run source not found.")
    return source


@router.get("/runs/{run_id}/export")
def export_run(
    run_id: str,
    export_format: Literal["markdown", "csv", "json"] = Query(default="markdown", alias="format"),
) -> Response:
    run = get_research_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Research run not found.")

    if export_format == "csv":
        body = export_run_csv(run)
        filename = f"{run_id}.csv"
        media_type = "text/csv"
    elif export_format == "json":
        body = export_run_json(run)
        filename = f"{run_id}.json"
        media_type = "application/json"
    else:
        body = export_run_markdown(run)
        filename = f"{run_id}.md"
        media_type = "text/markdown"

    return Response(
        content=body,
        media_type=media_type,
        headers={"content-disposition": f"attachment; filename={filename}"},
    )


@router.post("/runs/{run_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(run_id: str, payload: FeedbackRequest) -> FeedbackResponse:
    feedback_id = record_run_feedback(run_id=run_id, feedback=payload)
    return FeedbackResponse(feedback_id=feedback_id)


@router.get("/feedback/eval-cases", response_model=list[FeedbackEvalCase])
def list_feedback_eval_cases(limit: int = Query(default=100, ge=1, le=500)) -> list[FeedbackEvalCase]:
    return list_feedback_as_eval_cases(limit=limit)
