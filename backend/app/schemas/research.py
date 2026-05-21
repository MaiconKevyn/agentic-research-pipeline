from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import settings


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=3, description="Research question from the user.")
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=10,
        description="Maximum evidence items to keep.",
    )


class SourceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    title: str
    snippet: str
    source_type: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class CorpusStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_document_count: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    corpus_version_id: str


ToolName = Literal["vector_search", "web_search", "sql_query"]


class QuestionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_kind: Literal["research", "operational", "off_topic", "security_blocked"] = "research"
    selected_tools: list[ToolName] = Field(default_factory=list)
    rationale: str = Field(..., min_length=3)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(..., min_length=3)
    selected_tools: list[ToolName] = Field(default_factory=list)
    top_k: int = Field(..., ge=1, le=10)
    execution_notes: list[str] = Field(default_factory=list)


class EvidenceCollection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_count: int = Field(..., ge=0)
    kept_count: int = Field(..., ge=0)
    retrieval_quality: Literal["sufficient", "partial", "weak", "irrelevant"] = "weak"
    source_type_breakdown: dict[str, int] = Field(default_factory=dict)


class SecurityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    severity: Literal["low", "medium", "high"]
    message: str
    evidence: str | None = None


class SafetyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool
    action: Literal["allow", "block", "sanitize"]
    findings: list[SecurityFinding] = Field(default_factory=list)
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(..., min_length=3)


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: str = Field(..., min_length=3)
    supporting_source_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[ClaimEvidence] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    limitations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    support_status: Literal["supported", "unsupported", "conflicting"] = "supported"


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_summary: str = Field(..., min_length=3)
    claims: list[AnswerClaim] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    limitations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    uncertainty_note: str | None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: list[EvaluationScore] = Field(default_factory=list)
    summary: str = Field(..., min_length=3)


class ResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    corpus_version_id: str
    corpus_stats: CorpusStats
    question: str
    answer: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    sources: list[SourceItem]
    evaluation: list[EvaluationScore]
    execution_trace: list[str]
