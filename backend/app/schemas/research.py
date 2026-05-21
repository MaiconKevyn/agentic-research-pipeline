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
    query_kind: Literal["research", "operational", "off_topic"] = "research"
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


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(..., min_length=3)
    confidence: Literal["low", "medium", "high"]
    cited_source_ids: list[str]
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
    sources: list[SourceItem]
    evaluation: list[EvaluationScore]
    execution_trace: list[str]
