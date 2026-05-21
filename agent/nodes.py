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
from backend.app.services.security_guardrail import rate_limiter, security_guardrail
from backend.app.services.synthesis_verifier import verify_synthesis_claims


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

SECURITY_REFUSAL_MESSAGE = (
    "I cannot help with prompt injection, hidden prompt extraction, tool hijacking, "
    "or sensitive data exfiltration. Please ask a project-scoped research question."
)

OUTPUT_SECURITY_REFUSAL_MESSAGE = (
    "I cannot return this answer because the safety checks detected potential leakage "
    "of hidden instructions or personal data."
)


def _finding_categories(decision) -> str:
    if not decision or not decision.findings:
        return "none"
    return ",".join(finding.category for finding in decision.findings)


def _source_type_breakdown(sources: list[SourceItem]) -> str:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.source_type] = counts.get(source.source_type, 0) + 1
    return ",".join(f"{source_type}:{count}" for source_type, count in sorted(counts.items()))


def _grade_retrieval_quality(
    *,
    sources: list[SourceItem],
    top_k: int,
    query_kind: str,
) -> str:
    if not sources:
        return "weak"
    if query_kind == "operational" and any(source.source_type == "sql" for source in sources):
        return "sufficient"
    if len(sources) >= top_k:
        return "sufficient"
    if len(sources) >= max(1, top_k // 2):
        return "partial"
    return "weak"


def _is_in_project_scope(question: str) -> bool:
    return any(keyword in question for keyword in PROJECT_SCOPE_KEYWORDS)


def _is_operational_query(question: str) -> bool:
    if any(marker in question for marker in OPERATIONAL_SCOPE_MARKERS):
        return True

    has_index_hint = "indexed" in question or "indexado" in question
    has_entity_hint = any(keyword in question for keyword in INDEXED_ENTITY_KEYWORDS)
    return has_index_hint and has_entity_hint


def assess_input_safety(state: ResearchState) -> ResearchState:
    rate_decision = rate_limiter.check("research-api")
    input_decision = security_guardrail.assess_input(state["question"])
    execution_trace = [
        *state["execution_trace"],
        f"rate_limit_action={rate_decision.action}",
        f"input_estimated_tokens={input_decision.metadata.get('estimated_tokens', 0)}",
        f"max_input_chars={input_decision.metadata.get('max_input_chars', settings.max_input_chars)}",
        f"security_input_action={input_decision.action if rate_decision.allowed else 'block'}",
        f"security_input_findings={_finding_categories(input_decision if input_decision.findings else rate_decision)}",
    ]
    if not rate_decision.allowed:
        input_decision = rate_decision
    return {
        "input_safety": input_decision,
        "execution_trace": execution_trace,
    }


def classify_question(state: ResearchState) -> ResearchState:
    lowered = state["question"].lower()
    execution_trace = [*state["execution_trace"], "classified_question"]
    selected_tools = ["vector_search"]
    query_kind = "research"
    rationale = "General research question; use hybrid internal retrieval first and escalate only when evidence is weak."
    input_safety = state.get("input_safety")

    if input_safety and not input_safety.allowed:
        classification = QuestionClassification(
            query_kind="security_blocked",
            selected_tools=[],
            rationale=input_safety.rationale,
        )
        execution_trace.append("tool_policy_allowed=none")
        execution_trace.append("selected_tools=")
        execution_trace.append(f"classification_query_kind={classification.query_kind}")
        execution_trace.append("security_guardrail_triggered")
        return {
            "classification": classification,
            "selected_tools": [],
            "execution_trace": execution_trace,
        }

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
    execution_trace.append(
        f"tool_policy_allowed={','.join(classification.selected_tools) if classification.selected_tools else 'none'}"
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
        f"Use the requested answer mode: {state['answer_mode']}.",
        "Run evidence collection, apply global reranking, and synthesize only from the selected evidence.",
    ]
    if classification.query_kind == "off_topic":
        objective = f"Decline out-of-scope question: {state['question']}"
        execution_notes = [
            classification.rationale,
            "Do not call retrieval tools. Return the scope guardrail response.",
        ]
    elif classification.query_kind == "security_blocked":
        objective = f"Reject unsafe request: {state['question']}"
        execution_notes = [
            classification.rationale,
            "Do not call retrieval or synthesis tools. Return the security guardrail response.",
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
            f"answer_mode={state['answer_mode']}",
        ]
    }


def collect_evidence(state: ResearchState) -> ResearchState:
    collected: list[SourceItem] = []
    execution_trace = [*state["execution_trace"], "collecting_evidence"]
    classification = state["classification"]

    if classification and classification.query_kind in {"off_topic", "security_blocked"}:
        execution_trace.append("scope_guardrail_skipped_evidence_collection")
        return {
            "evidence_collection": EvidenceCollection(
                candidate_count=0,
                kept_count=0,
                retrieval_quality="irrelevant",
                source_type_breakdown={},
            ),
            "retrieval_quality": "irrelevant",
            "sources": [],
            "execution_trace": execution_trace,
        }

    if "vector_search" in state["selected_tools"]:
        vector_results = search_documents(state["question"], state["top_k"])
        execution_trace.append(f"hybrid_search_results={len(vector_results)}")
        collected.extend(vector_results)

    initial_quality = _grade_retrieval_quality(
        sources=collected,
        top_k=state["top_k"],
        query_kind=classification.query_kind if classification else "research",
    )
    execution_trace.append(f"retrieval_quality={initial_quality}")

    should_escalate_web = (
        classification is not None
        and classification.query_kind == "research"
        and initial_quality in {"partial", "weak"}
        and settings.max_web_searches_per_run > 0
    )
    web_searches_used = 0
    if should_escalate_web:
        execution_trace.append("corrective_web_search_triggered")
        web_results = search_web(state["question"], state["top_k"])
        web_searches_used = 1
        execution_trace.append(f"web_search_results={len(web_results)}")
        collected.extend(web_results)
    else:
        execution_trace.append("corrective_web_search_skipped")
    execution_trace.append(f"web_searches_used={web_searches_used}")
    execution_trace.append(f"max_web_searches={settings.max_web_searches_per_run}")

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
    kept_sources, content_safety = security_guardrail.assess_retrieved_content(kept_sources)
    kept_sources, token_decision = security_guardrail.apply_retrieved_token_limit(kept_sources)
    execution_trace.append(f"security_retrieved_content_action={content_safety.action}")
    execution_trace.append(f"security_retrieved_content_findings={_finding_categories(content_safety)}")
    execution_trace.append("retrieved_token_limit_applied")
    execution_trace.append(f"max_retrieved_tokens={token_decision.metadata.get('max_retrieved_tokens')}")
    retrieval_quality = _grade_retrieval_quality(
        sources=kept_sources,
        top_k=state["top_k"],
        query_kind=classification.query_kind if classification else "research",
    )
    if retrieval_quality != initial_quality:
        execution_trace.append(f"retrieval_quality_after_correction={retrieval_quality}")
    evidence_collection = EvidenceCollection(
        candidate_count=len(reranked_sources),
        kept_count=len(kept_sources),
        retrieval_quality=retrieval_quality,
        source_type_breakdown={
            source.source_type: sum(1 for item in kept_sources if item.source_type == source.source_type)
            for source in kept_sources
        },
    )
    execution_trace.append(f"sources_kept={len(kept_sources)}")
    execution_trace.append(f"sources_kept_breakdown={_source_type_breakdown(kept_sources)}")
    return {
        "evidence_collection": evidence_collection,
        "retrieval_quality": retrieval_quality,
        "sources": kept_sources,
        "retrieved_content_safety": content_safety,
        "execution_trace": execution_trace,
    }


def synthesize_answer(state: ResearchState) -> ResearchState:
    execution_trace = [*state["execution_trace"], "synthesizing_answer"]
    classification = state["classification"]

    if classification and classification.query_kind == "security_blocked":
        execution_trace.append("security_guardrail_response")
        synthesis = SynthesisOutput(
            answer_summary=SECURITY_REFUSAL_MESSAGE,
            confidence="high",
            claims=[],
            limitations=["Unsafe input was blocked before tool execution."],
            conflicts=[],
            follow_up_questions=[],
            uncertainty_note="Request was rejected by the input safety policy.",
        )
        return {
            "synthesis": synthesis,
            "answer": synthesis.answer_summary,
            "claims": [],
            "execution_trace": execution_trace,
        }

    if classification and classification.query_kind == "off_topic":
        execution_trace.append("scope_guardrail_response")
        synthesis = SynthesisOutput(
            answer_summary=OFF_TOPIC_MESSAGE,
            confidence="high",
            claims=[],
            limitations=[],
            conflicts=[],
            follow_up_questions=[],
            uncertainty_note="Question was rejected because it is out of scope for this project.",
        )
        return {
            "synthesis": synthesis,
            "answer": synthesis.answer_summary,
            "claims": [],
            "execution_trace": execution_trace,
        }

    if state["retrieval_quality"] in {"weak", "irrelevant"}:
        execution_trace.append("weak_retrieval_abstention")
        answer = (
            "Insufficient evidence: the retrieval step did not find enough relevant support "
            "to answer this question without risking an unsupported claim."
        )
        synthesis = SynthesisOutput(
            answer_summary=answer,
            confidence="low",
            claims=[],
            limitations=["Retrieval quality was too weak for grounded synthesis."],
            conflicts=[],
            follow_up_questions=[],
            uncertainty_note="Retrieval quality was too weak for grounded synthesis.",
        )
        return {
            "synthesis": synthesis,
            "answer": answer,
            "claims": [],
            "execution_trace": execution_trace,
        }

    if not state["sources"]:
        return {
            "answer": (
                "Insufficient evidence: no usable sources were collected for grounded synthesis."
            ),
            "claims": [],
            "execution_trace": execution_trace,
        }

    budget_decision = security_guardrail.assess_model_budget(
        question=state["question"],
        sources=state["sources"],
    )
    execution_trace.append(f"model_budget_action={budget_decision.action}")
    execution_trace.append(f"estimated_model_cost_usd={budget_decision.metadata.get('estimated_model_cost_usd')}")
    if not budget_decision.allowed:
        execution_trace.append("model_budget_abstention")
        answer = "Insufficient evidence: model budget limits prevented safe grounded synthesis."
        synthesis = SynthesisOutput(
            answer_summary=answer,
            confidence="low",
            claims=[],
            limitations=["Model budget policy blocked synthesis for this run."],
            conflicts=[],
            follow_up_questions=[],
            uncertainty_note="Synthesis skipped because resource limits were exceeded.",
        )
        return {
            "synthesis": synthesis,
            "answer": answer,
            "claims": [],
            "execution_trace": execution_trace,
        }

    execution_trace.append(f"llm_model={settings.openai_model}")
    try:
        synthesis = generate_research_answer(
            question=state["question"],
            sources=state["sources"],
            answer_mode=state["answer_mode"],
        )
        answer = synthesis.answer_summary
        execution_trace.append("llm_synthesis_success")
        execution_trace.append(f"synthesis_confidence={synthesis.confidence}")
        execution_trace.append(f"synthesis_claims={len(synthesis.claims)}")
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
        "claims": synthesis.claims if synthesis else [],
        "execution_trace": execution_trace,
    }


def verify_synthesis(state: ResearchState) -> ResearchState:
    execution_trace = [*state["execution_trace"], "verifying_claims"]
    synthesis = state["synthesis"]
    if synthesis is None:
        execution_trace.append("claim_verification_skipped")
        return {
            "claims": [],
            "execution_trace": execution_trace,
        }

    verified = verify_synthesis_claims(
        synthesis=synthesis,
        sources=state["sources"],
    )
    removed_count = len(synthesis.claims) - len(verified.claims)
    execution_trace.append(f"claim_verification_removed={removed_count}")
    execution_trace.append("claim_verification_complete")
    return {
        "synthesis": verified,
        "answer": verified.answer_summary,
        "claims": verified.claims,
        "execution_trace": execution_trace,
    }


def assess_output_safety(state: ResearchState) -> ResearchState:
    execution_trace = [*state["execution_trace"], "checking_output_safety"]
    classification = state["classification"]
    if classification and classification.query_kind in {"security_blocked", "off_topic"}:
        execution_trace.append("security_output_action=allow")
        execution_trace.append("security_output_findings=none")
        return {
            "execution_trace": execution_trace,
        }
    decision = security_guardrail.assess_output(state["answer"])
    execution_trace.append(f"security_output_action={decision.action}")
    execution_trace.append(f"security_output_findings={_finding_categories(decision)}")
    if not decision.allowed:
        synthesis = SynthesisOutput(
            answer_summary=OUTPUT_SECURITY_REFUSAL_MESSAGE,
            confidence="high",
            claims=[],
            limitations=["Output safety policy blocked the generated answer."],
            conflicts=[],
            follow_up_questions=[],
            uncertainty_note="Potential leakage was detected after synthesis.",
        )
        return {
            "output_safety": decision,
            "synthesis": synthesis,
            "answer": synthesis.answer_summary,
            "claims": [],
            "execution_trace": execution_trace,
        }
    return {
        "output_safety": decision,
        "execution_trace": execution_trace,
    }


def evaluate_answer(state: ResearchState) -> ResearchState:
    classification = state["classification"]
    if classification and classification.query_kind == "security_blocked":
        evaluation_result = EvaluationResult(
            scores=[
                EvaluationScore(
                    metric="security_compliance",
                    score=1.0,
                    rationale="The assistant blocked unsafe input before calling tools or the language model.",
                ),
                EvaluationScore(
                    metric="schema_validity",
                    score=1.0,
                    rationale="The refusal still follows the structured API schema.",
                ),
            ],
            summary="Unsafe request correctly rejected by the security guardrail.",
        )
        return {
            "evaluation_result": evaluation_result,
            "evaluation": evaluation_result.scores,
            "execution_trace": [*state["execution_trace"], "evaluating_answer"],
        }
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
    claim_support_score = (
        1.0
        if state["claims"] and all(
            claim.support_status == "supported" and claim.supporting_quotes
            for claim in state["claims"]
        )
        else 0.0 if state["claims"] else 1.0 if state["answer"].startswith("Insufficient evidence") else 0.0
    )
    evidence_sufficiency = min(len(state["sources"]) / max(state["top_k"], 1), 1.0)

    evaluation = [
        EvaluationScore(
            metric="citation_coverage",
            score=citation_score,
            rationale="The answer includes sources when evidence collection returns usable evidence.",
        ),
        EvaluationScore(
            metric="groundedness",
            score=max(groundedness_score, claim_support_score),
            rationale="Checks whether the final answer is backed by collected evidence or supported claim links.",
        ),
        EvaluationScore(
            metric="claim_support",
            score=claim_support_score,
            rationale="Checks whether every returned claim has at least one supporting source quote.",
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
