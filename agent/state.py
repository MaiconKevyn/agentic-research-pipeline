from typing import TypedDict

from backend.app.schemas.research import (
    AnswerClaim,
    EvaluationScore,
    EvaluationResult,
    EvidenceCollection,
    QuestionClassification,
    ResearchPlan,
    SourceItem,
    SynthesisOutput,
)


class ResearchState(TypedDict):
    run_id: str
    corpus_version_id: str
    question: str
    top_k: int
    classification: QuestionClassification | None
    plan: ResearchPlan | None
    evidence_collection: EvidenceCollection | None
    retrieval_quality: str
    synthesis: SynthesisOutput | None
    evaluation_result: EvaluationResult | None
    selected_tools: list[str]
    sources: list[SourceItem]
    answer: str
    claims: list[AnswerClaim]
    evaluation: list[EvaluationScore]
    execution_trace: list[str]
