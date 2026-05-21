import time
import uuid

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    assess_input_safety,
    assess_output_safety,
    classify_question,
    collect_evidence,
    evaluate_answer,
    plan_research,
    synthesize_answer,
    verify_synthesis,
)
from agent.state import ResearchState
from backend.app.core.config import settings
from backend.app.schemas.research import CorpusStats, ResearchResponse
from backend.app.services.document_repository import (
    DocumentRepositoryError,
    ResearchRunRecord,
    get_corpus_stats,
    get_or_create_current_corpus_version_id,
    record_research_run,
)


def build_research_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("assess_input_safety", assess_input_safety)
    builder.add_node("classify_question", classify_question)
    builder.add_node("plan_research", plan_research)
    builder.add_node("collect_evidence", collect_evidence)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("verify_synthesis", verify_synthesis)
    builder.add_node("assess_output_safety", assess_output_safety)
    builder.add_node("evaluate_answer", evaluate_answer)

    builder.add_edge(START, "assess_input_safety")
    builder.add_edge("assess_input_safety", "classify_question")
    builder.add_edge("classify_question", "plan_research")
    builder.add_edge("plan_research", "collect_evidence")
    builder.add_edge("collect_evidence", "synthesize_answer")
    builder.add_edge("synthesize_answer", "verify_synthesis")
    builder.add_edge("verify_synthesis", "assess_output_safety")
    builder.add_edge("assess_output_safety", "evaluate_answer")
    builder.add_edge("evaluate_answer", END)
    return builder.compile()


research_graph = build_research_graph()


def run_research(question: str, top_k: int = 5) -> ResearchResponse:
    run_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    try:
        corpus_version_id = get_or_create_current_corpus_version_id()
    except DocumentRepositoryError as exc:
        corpus_version_id = "corpus-unavailable"
        version_trace = [f"corpus_version_error={exc}"]
    else:
        version_trace = [f"corpus_version_id={corpus_version_id}"]

    state: ResearchState = {
        "run_id": run_id,
        "corpus_version_id": corpus_version_id,
        "question": question,
        "top_k": top_k,
        "classification": None,
        "input_safety": None,
        "retrieved_content_safety": None,
        "output_safety": None,
        "plan": None,
        "evidence_collection": None,
        "retrieval_quality": "weak",
        "synthesis": None,
        "evaluation_result": None,
        "selected_tools": [],
        "sources": [],
        "answer": "",
        "claims": [],
        "evaluation": [],
        "execution_trace": [f"run_id={run_id}", *version_trace],
    }
    final_state = research_graph.invoke(state)
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    try:
        corpus_stats = get_corpus_stats()
    except DocumentRepositoryError as exc:
        corpus_stats = CorpusStats(
            source_document_count=0,
            chunk_count=0,
            corpus_version_id=corpus_version_id,
        )
        final_state["execution_trace"] = [
            *final_state["execution_trace"],
            f"corpus_stats_error={exc}",
        ]

    classification = final_state.get("classification")
    selected_tools = final_state.get("selected_tools", [])
    try:
        record_research_run(
            ResearchRunRecord(
                run_id=run_id,
                corpus_version_id=corpus_version_id,
                question=final_state["question"],
                classification=classification.query_kind if classification else None,
                selected_tools=selected_tools,
                model=settings.openai_model,
                answer=final_state["answer"],
                claims=final_state["claims"],
                sources=final_state["sources"],
                evaluation_scores=final_state["evaluation"],
                execution_trace=final_state["execution_trace"],
                latency_ms=latency_ms,
            )
        )
        final_state["execution_trace"] = [*final_state["execution_trace"], "research_run_persisted"]
    except DocumentRepositoryError as exc:
        final_state["execution_trace"] = [
            *final_state["execution_trace"],
            f"research_run_persistence_error={exc}",
        ]

    return ResearchResponse(
        run_id=run_id,
        corpus_version_id=corpus_version_id,
        corpus_stats=corpus_stats,
        question=final_state["question"],
        answer=final_state["answer"],
        claims=final_state["claims"],
        sources=final_state["sources"],
        evaluation=final_state["evaluation"],
        execution_trace=final_state["execution_trace"],
    )
