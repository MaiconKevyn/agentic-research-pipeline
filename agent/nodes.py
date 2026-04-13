from agent.state import ResearchState
from agent.tools.sql_query import query_structured_data
from agent.tools.vector_search import search_documents
from agent.tools.web_search import search_web
from backend.app.core.config import settings
from backend.app.schemas.research import (
    EvaluationResult,
    EvaluationScore,
    EvidenceCollection,
    QuestionClassification,
    ResearchPlan,
    SourceItem,
    SynthesisOutput,
)
from backend.app.services.llm_service import LLMServiceError, generate_research_answer
from backend.app.services.rerank_service import RerankServiceError, rerank_sources_global


PROJECT_SCOPE_KEYWORDS = (
    "project",
    "this project",
    "internal corpus",
    "internal base",
    "indexed corpus",
    "indexed chunks",
    "indexed documents",
    "research agent",
    "agentic",
    "agent workflow",
    "workflow",
    "workflows",
    "langgraph",
    "langchain",
    "langsmith",
    "rag",
    "retrieval",
    "vector search",
    "web search",
    "pgvector",
    "postgresql",
    "fastapi",
    "pydantic",
    "tool calling",
    "evaluation",
    "embeddings",
    "embedding",
    "openai",
    "schema",
    "projeto",
    "corpus interno",
    "base interna",
    "agente de pesquisa",
    "agentic ai",
    "fluxo do agente",
    "orquestracao",
    "lang graph",
    "busca vetorial",
    "busca web",
    "avaliacao",
    "embeddings",
)

OPERATIONAL_SCOPE_MARKERS = (
    "internal base",
    "internal corpus",
    "indexed corpus",
    "base interna",
    "corpus interno",
)

COUNT_KEYWORDS = ("count", "total", "sum", "quantos", "how many")

INDEXED_ENTITY_KEYWORDS = (
    "indexed",
    "indexado",
    "chunk",
    "chunks",
    "document",
    "documents",
    "documento",
    "documentos",
    "source",
    "sources",
    "fonte",
    "fontes",
)

OFF_TOPIC_MESSAGE = (
    "This assistant only answers questions about this project and its indexed domain, "
    "such as the internal corpus, RAG, agentic workflows, LangGraph, LangChain, "
    "FastAPI, pgvector, and evaluation. Please ask a question within that scope."
)


def _source_type_breakdown(sources: list[SourceItem]) -> str:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.source_type] = counts.get(source.source_type, 0) + 1
    return ",".join(f"{source_type}:{count}" for source_type, count in sorted(counts.items()))


def _is_in_project_scope(question: str) -> bool:
    return any(keyword in question for keyword in PROJECT_SCOPE_KEYWORDS)


def _is_operational_query(question: str) -> bool:
    if any(marker in question for marker in OPERATIONAL_SCOPE_MARKERS):
        return True

    has_index_hint = "indexed" in question or "indexado" in question
    has_entity_hint = any(keyword in question for keyword in INDEXED_ENTITY_KEYWORDS)
    return has_index_hint and has_entity_hint


def classify_question(state: ResearchState) -> ResearchState:
    lowered = state["question"].lower()
    execution_trace = [*state["execution_trace"], "classified_question"]
    selected_tools = ["vector_search", "web_search"]
    query_kind = "research"
    rationale = "General research question; use the internal corpus and web search as evidence sources."

    if not _is_in_project_scope(lowered):
        query_kind = "off_topic"
        selected_tools = []
        rationale = "Question is outside the scope of this project and its indexed domain."
    elif _is_operational_query(lowered):
        query_kind = "operational"
        selected_tools = ["vector_search"]
        rationale = "Operational question about the internal knowledge base; prioritize internal evidence and structured data."

    if query_kind != "off_topic" and any(keyword in lowered for keyword in COUNT_KEYWORDS):
        if "sql_query" not in selected_tools:
            selected_tools.append("sql_query")
    classification = QuestionClassification(
        query_kind=query_kind,
        selected_tools=selected_tools,
        rationale=rationale,
    )
    execution_trace.append(f"selected_tools={','.join(classification.selected_tools)}")
    execution_trace.append(f"classification_query_kind={classification.query_kind}")
    if classification.query_kind == "off_topic":
        execution_trace.append("scope_guardrail_triggered")
    return {
        "classification": classification,
        "selected_tools": classification.selected_tools,
        "execution_trace": execution_trace,
    }


def plan_research(state: ResearchState) -> ResearchState:
    classification = state["classification"] or QuestionClassification(
        selected_tools=state["selected_tools"],
        rationale="Fallback classification.",
    )
    objective = f"Answer the question: {state['question']}"
    execution_notes = [
        classification.rationale,
        "Run evidence collection, apply global reranking, and synthesize only from the selected evidence.",
    ]
    if classification.query_kind == "off_topic":
        objective = f"Decline out-of-scope question: {state['question']}"
        execution_notes = [
            classification.rationale,
            "Do not call retrieval tools. Return the scope guardrail response.",
        ]
    plan = ResearchPlan(
        objective=objective,
        selected_tools=classification.selected_tools,
        top_k=state["top_k"],
        execution_notes=execution_notes,
    )
    return {
        "plan": plan,
        "execution_trace": [
            *state["execution_trace"],
            f"planned_research_with_tools={','.join(plan.selected_tools)}",
            f"planned_top_k={plan.top_k}",
        ]
    }


def collect_evidence(state: ResearchState) -> ResearchState:
    collected: list[SourceItem] = []
    execution_trace = [*state["execution_trace"], "collecting_evidence"]
    classification = state["classification"]

    if classification and classification.query_kind == "off_topic":
        execution_trace.append("scope_guardrail_skipped_evidence_collection")
        return {
            "evidence_collection": EvidenceCollection(
                candidate_count=0,
                kept_count=0,
                source_type_breakdown={},
            ),
            "sources": [],
            "execution_trace": execution_trace,
        }

    if "vector_search" in state["selected_tools"]:
        vector_results = search_documents(state["question"], state["top_k"])
        execution_trace.append(f"vector_search_results={len(vector_results)}")
        collected.extend(vector_results)
    if "web_search" in state["selected_tools"]:
        web_results = search_web(state["question"], state["top_k"])
        execution_trace.append(f"web_search_results={len(web_results)}")
        collected.extend(web_results)
    if "sql_query" in state["selected_tools"]:
        sql_results = query_structured_data(state["question"])
        execution_trace.append(f"sql_query_results={len(sql_results)}")
        collected.extend(sql_results)

    deduplicated: dict[str, SourceItem] = {}
    for item in collected:
        deduplicated[item.source_id] = item

    reranked_sources = list(deduplicated.values())
    execution_trace.append(f"sources_after_dedup={len(reranked_sources)}")
    try:
        reranked_sources = rerank_sources_global(
            question=state["question"],
            sources=reranked_sources,
        )
        execution_trace.append(f"global_rerank_candidates={len(reranked_sources)}")
        execution_trace.append("global_rerank_applied")
    except RerankServiceError as exc:
        execution_trace.append(f"global_rerank_error={exc}")

    kept_sources = reranked_sources[: state["top_k"]]
    evidence_collection = EvidenceCollection(
        candidate_count=len(reranked_sources),
        kept_count=len(kept_sources),
        source_type_breakdown={
            source.source_type: sum(1 for item in kept_sources if item.source_type == source.source_type)
            for source in kept_sources
        },
    )
    execution_trace.append(f"sources_kept={len(kept_sources)}")
    execution_trace.append(f"sources_kept_breakdown={_source_type_breakdown(kept_sources)}")
    return {
        "evidence_collection": evidence_collection,
        "sources": kept_sources,
        "execution_trace": execution_trace,
    }


def synthesize_answer(state: ResearchState) -> ResearchState:
    execution_trace = [*state["execution_trace"], "synthesizing_answer"]
    classification = state["classification"]

    if classification and classification.query_kind == "off_topic":
        execution_trace.append("scope_guardrail_response")
        synthesis = SynthesisOutput(
            answer=OFF_TOPIC_MESSAGE,
            confidence="high",
            cited_source_ids=[],
            uncertainty_note="Question was rejected because it is out of scope for this project.",
        )
        return {
            "synthesis": synthesis,
            "answer": synthesis.answer,
            "execution_trace": execution_trace,
        }

    if not state["sources"]:
        return {
            "answer": (
                "No evidence was collected yet. The initial pipeline exists, "
                "but the required tools still need to return usable evidence."
            ),
            "execution_trace": execution_trace,
        }

    execution_trace.append(f"llm_model={settings.openai_model}")
    try:
        synthesis = generate_research_answer(
            question=state["question"],
            sources=state["sources"],
        )
        answer = synthesis.answer
        execution_trace.append("llm_synthesis_success")
        execution_trace.append(f"synthesis_confidence={synthesis.confidence}")
        execution_trace.append(f"synthesis_citations={len(synthesis.cited_source_ids)}")
    except LLMServiceError as exc:
        execution_trace.append(f"llm_synthesis_error={exc}")
        source_titles = ", ".join(source.title for source in state["sources"][:3])
        answer = (
            "Failed to synthesize the answer with the configured model. "
            f"Collected evidence: {source_titles}. "
            f"Detail: {exc}"
        )
        synthesis = None
    return {
        "synthesis": synthesis,
        "answer": answer,
        "execution_trace": execution_trace,
    }


def evaluate_answer(state: ResearchState) -> ResearchState:
    classification = state["classification"]
    if classification and classification.query_kind == "off_topic":
        evaluation_result = EvaluationResult(
            scores=[
                EvaluationScore(
                    metric="scope_compliance",
                    score=1.0,
                    rationale="The assistant correctly rejected an out-of-scope question before calling retrieval tools.",
                ),
                EvaluationScore(
                    metric="schema_validity",
                    score=1.0,
                    rationale="The refusal still follows the structured API schema.",
                ),
            ],
            summary="Out-of-scope request correctly rejected by the scope guardrail.",
        )
        return {
            "evaluation_result": evaluation_result,
            "evaluation": evaluation_result.scores,
            "execution_trace": [*state["execution_trace"], "evaluating_answer"],
        }

    citation_score = 1.0 if state["sources"] else 0.0
    completeness_score = 0.7 if state["answer"] else 0.0
    groundedness_score = 1.0 if state["sources"] and "[" in state["answer"] and "]" in state["answer"] else 0.5 if state["sources"] else 0.0
    evidence_sufficiency = min(len(state["sources"]) / max(state["top_k"], 1), 1.0)

    evaluation = [
        EvaluationScore(
            metric="citation_coverage",
            score=citation_score,
            rationale="The answer includes sources when evidence collection returns usable evidence.",
        ),
        EvaluationScore(
            metric="groundedness",
            score=groundedness_score,
            rationale="Initial heuristic: checks whether the final answer explicitly cites collected evidence.",
        ),
        EvaluationScore(
            metric="answer_completeness",
            score=completeness_score,
            rationale="The answer summarizes the question and the result of evidence collection.",
        ),
        EvaluationScore(
            metric="evidence_sufficiency",
            score=evidence_sufficiency,
            rationale="Compares how many valid evidence items were kept relative to the requested limit.",
        ),
        EvaluationScore(
            metric="schema_validity",
            score=1.0,
            rationale="The output follows the structured schema defined for the API.",
        ),
    ]
    evaluation_result = EvaluationResult(
        scores=evaluation,
        summary="Initial heuristic evaluation completed for the synthesized answer.",
    )
    return {
        "evaluation_result": evaluation_result,
        "evaluation": evaluation_result.scores,
        "execution_trace": [*state["execution_trace"], "evaluating_answer"],
    }
