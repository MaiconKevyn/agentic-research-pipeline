from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from agent.graph import run_research
from backend.app.schemas.research import ResearchResponse
from evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


DATASET_PATH = Path(__file__).parent / "datasets" / "sample_questions.json"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden" / "smoke.jsonl"
DEFAULT_REPORT_PATH = Path(__file__).parent / "reports" / "latest.jsonl"
DEFAULT_BASELINE_PATH = Path(__file__).parent / "reports" / "baseline.json"
BASELINE_REGRESSION_TOLERANCE = 0.02

THRESHOLDS = {
    "schema_validity": 1.00,
    "scope_compliance": 0.98,
    "citation_precision": 0.95,
    "groundedness": 0.90,
    "answer_relevance": 0.90,
    "retrieval_recall_at_10": 0.85,
    "retrieval_mrr": 0.85,
    "retrieval_ndcg_at_10": 0.85,
    "abstention_accuracy": 0.90,
}

ExpectedAnswerType = Literal["factual", "comparison", "operational", "insufficient_evidence"]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_answer_type: ExpectedAnswerType
    required_sources: list[str]
    expected_facts: list[str]
    forbidden_claims: list[str]
    answer_rubric: str
    difficulty: str
    query_category: str
    top_k: int = 5


@dataclass(frozen=True)
class EvaluationSummary:
    case_count: int
    metrics: dict[str, float]
    thresholds: dict[str, float]
    thresholds_passed: bool
    output_path: Path
    baseline_path: Path | None = None
    baseline_metrics: dict[str, float] | None = None
    regressions: list[str] | None = None


RunFunc = Callable[[str, int], ResearchResponse]


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def load_golden_cases(dataset_path: Path = GOLDEN_DATASET_PATH) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                cases.append(GoldenCase(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid golden case on line {line_number}: {exc}") from exc
    return cases


def load_all_golden_cases(golden_dir: Path | None = None) -> list[GoldenCase]:
    target_dir = golden_dir or GOLDEN_DATASET_PATH.parent
    cases: list[GoldenCase] = []
    for dataset_path in sorted(target_dir.glob("*.jsonl")):
        cases.extend(load_golden_cases(dataset_path))
    return cases


def run_benchmark() -> list[dict]:
    results: list[dict] = []
    for item in load_dataset():
        response = run_research(question=item["question"], top_k=item.get("top_k", 5))
        results.append(
            {
                "question": item["question"],
                "overall_score": _mean(_score_case(_case_from_legacy_item(item), response).values()),
                "answer": response.answer,
                "trace": response.execution_trace,
            }
        )
    return results


def run_evaluation(
    *,
    dataset_path: Path = GOLDEN_DATASET_PATH,
    output_path: Path = DEFAULT_REPORT_PATH,
    baseline_path: Path | None = None,
    run_func: RunFunc = run_research,
) -> EvaluationSummary:
    cases = load_golden_cases(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_cases: list[dict] = []

    for case in cases:
        response = run_func(case.question, case.top_k)
        scores = _score_case(case, response)
        scored_cases.append(
            {
                "case_id": case.id,
                "question": case.question,
                "query_category": case.query_category,
                "difficulty": case.difficulty,
                "run_id": response.run_id,
                "corpus_version_id": response.corpus_version_id,
                "scores": scores,
                "answer": response.answer,
                "source_ids": [source.source_id for source in response.sources],
                "execution_trace": response.execution_trace,
            }
        )

    with output_path.open("w", encoding="utf-8") as report_file:
        for result in scored_cases:
            report_file.write(json.dumps(result, sort_keys=True) + "\n")

    metrics = _aggregate_metrics([result["scores"] for result in scored_cases])
    baseline_metrics = _load_baseline_metrics(baseline_path)
    regressions = _detect_regressions(metrics=metrics, baseline_metrics=baseline_metrics)
    thresholds_passed = (
        all(metrics.get(metric, 0.0) >= threshold for metric, threshold in THRESHOLDS.items())
        and not regressions
    )
    summary = EvaluationSummary(
        case_count=len(scored_cases),
        metrics=metrics,
        thresholds=THRESHOLDS,
        thresholds_passed=thresholds_passed,
        output_path=output_path,
        baseline_path=baseline_path,
        baseline_metrics=baseline_metrics,
        regressions=regressions,
    )
    _write_summary_report(summary)
    return summary


def _case_from_legacy_item(item: dict) -> GoldenCase:
    return GoldenCase(
        id=item.get("id", item["question"]),
        question=item["question"],
        expected_answer_type=item.get("expected_answer_type", "factual"),
        required_sources=item.get("required_sources", []),
        expected_facts=item.get("expected_facts", []),
        forbidden_claims=item.get("forbidden_claims", []),
        answer_rubric=item.get("answer_rubric", "Legacy sample benchmark item."),
        difficulty=item.get("difficulty", "unknown"),
        query_category=item.get("query_category", "legacy"),
        top_k=item.get("top_k", 5),
    )


def _score_case(case: GoldenCase, response: ResearchResponse) -> dict[str, float]:
    answer_lower = response.answer.lower()
    source_ids = {source.source_id for source in response.sources}
    expected_facts = [fact.lower() for fact in case.expected_facts]
    forbidden_claims = [claim.lower() for claim in case.forbidden_claims]
    required_sources = set(case.required_sources)

    expected_facts_present = (
        1.0
        if not expected_facts
        else sum(1 for fact in expected_facts if fact in answer_lower) / len(expected_facts)
    )
    forbidden_claims_absent = (
        1.0
        if not forbidden_claims
        else 1.0 - (sum(1 for claim in forbidden_claims if claim in answer_lower) / len(forbidden_claims))
    )
    ranked_source_ids = [source.source_id for source in response.sources]
    source_recall = recall_at_k(ranked_source_ids, required_sources, k=10)
    retrieval_mrr = mean_reciprocal_rank(ranked_source_ids, required_sources)
    retrieval_ndcg = ndcg_at_k(ranked_source_ids, required_sources, k=10)
    has_required_citation = 1.0 if not required_sources else float(bool(required_sources & source_ids))
    abstained = "insufficient evidence" in answer_lower or "outside the project scope" in answer_lower
    expects_abstention = case.expected_answer_type == "insufficient_evidence"
    scope_triggered = "scope_guardrail_triggered" in response.execution_trace

    return {
        "schema_validity": 1.0,
        "scope_compliance": 1.0 if (not expects_abstention or abstained or scope_triggered) else 0.0,
        "citation_precision": has_required_citation,
        "groundedness": min(expected_facts_present, forbidden_claims_absent, has_required_citation),
        "answer_relevance": min(expected_facts_present, forbidden_claims_absent),
        "retrieval_recall_at_10": source_recall,
        "retrieval_mrr": retrieval_mrr,
        "retrieval_ndcg_at_10": retrieval_ndcg,
        "abstention_accuracy": 1.0 if abstained == expects_abstention else 0.0,
    }


def _aggregate_metrics(case_scores: list[dict[str, float]]) -> dict[str, float]:
    if not case_scores:
        return {metric: 0.0 for metric in THRESHOLDS}
    return {
        metric: _mean(score[metric] for score in case_scores)
        for metric in THRESHOLDS
    }


def _load_baseline_metrics(baseline_path: Path | None) -> dict[str, float] | None:
    if baseline_path is None or not baseline_path.exists():
        return None
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {
        metric: float(value)
        for metric, value in metrics.items()
        if isinstance(value, int | float)
    }


def _detect_regressions(
    *,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    tolerance: float = BASELINE_REGRESSION_TOLERANCE,
) -> list[str]:
    if not baseline_metrics:
        return []
    regressions: list[str] = []
    for metric, baseline_value in baseline_metrics.items():
        if metric in metrics and metrics[metric] < baseline_value - tolerance:
            regressions.append(metric)
    return regressions


def _write_summary_report(summary: EvaluationSummary) -> None:
    summary_path = summary.output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "case_count": summary.case_count,
                "metrics": summary.metrics,
                "thresholds": summary.thresholds,
                "thresholds_passed": summary.thresholds_passed,
                "baseline_path": str(summary.baseline_path) if summary.baseline_path else None,
                "baseline_metrics": summary.baseline_metrics,
                "regressions": summary.regressions,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _mean(values: object) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)
