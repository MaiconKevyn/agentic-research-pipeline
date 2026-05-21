from agent.graph import run_research
from backend.app.schemas.research import ResearchRequest, ResearchResponse


def run_research_pipeline(payload: ResearchRequest) -> ResearchResponse:
    return run_research(
        question=payload.question,
        top_k=payload.top_k,
        answer_mode=payload.answer_mode,
        workspace_id=payload.workspace_id,
    )
