from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    classify_question,
    collect_evidence,
    evaluate_answer,
    plan_research,
    synthesize_answer,
)
from agent.state import ResearchState
from backend.app.schemas.research import ResearchResponse


def build_research_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("classify_question", classify_question)
    builder.add_node("plan_research", plan_research)
    builder.add_node("collect_evidence", collect_evidence)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("evaluate_answer", evaluate_answer)

    builder.add_edge(START, "classify_question")
    builder.add_edge("classify_question", "plan_research")
    builder.add_edge("plan_research", "collect_evidence")
    builder.add_edge("collect_evidence", "synthesize_answer")
    builder.add_edge("synthesize_answer", "evaluate_answer")
    builder.add_edge("evaluate_answer", END)
    return builder.compile()


research_graph = build_research_graph()


def run_research(question: str, top_k: int = 5) -> ResearchResponse:
    state: ResearchState = {
        "question": question,
        "top_k": top_k,
        "classification": None,
        "plan": None,
        "evidence_collection": None,
        "synthesis": None,
        "evaluation_result": None,
        "selected_tools": [],
        "sources": [],
        "answer": "",
        "evaluation": [],
        "execution_trace": [],
    }
    final_state = research_graph.invoke(state)

    return ResearchResponse(
        question=final_state["question"],
        answer=final_state["answer"],
        sources=final_state["sources"],
        evaluation=final_state["evaluation"],
        execution_trace=final_state["execution_trace"],
    )
