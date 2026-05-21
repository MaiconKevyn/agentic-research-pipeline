from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.schemas.research import CorpusStats, ResearchResponse, SourceItem
from evaluation.runner import DEFAULT_REPORT_PATH, GOLDEN_DATASET_PATH, run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set research evaluation harness.")
    parser.add_argument("--dataset", type=Path, default=GOLDEN_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--mode",
        choices=("live", "mock"),
        default="live",
        help="live runs the LangGraph pipeline; mock exercises CI gates without external services.",
    )
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    run_func = _mock_run if args.mode == "mock" else None
    summary = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        **({"run_func": run_func} if run_func else {}),
    )
    for metric, score in sorted(summary.metrics.items()):
        threshold = summary.thresholds[metric]
        print(f"{metric}: {score:.3f} (threshold {threshold:.3f})")
    print(f"cases: {summary.case_count}")
    print(f"report: {summary.output_path}")

    if args.fail_on_threshold and not summary.thresholds_passed:
        return 1
    return 0


def _mock_run(question: str, top_k: int = 5) -> ResearchResponse:
    source_id = "mock-source"
    answer = "Insufficient evidence: this request is outside the project scope."
    sources: list[SourceItem] = []
    trace = ["scope_guardrail_triggered"]
    if "football" not in question.lower() and "unrelated" not in question.lower():
        answer = "The corpus says retrieval grounds RAG answers in source evidence [mock-source]."
        sources = [
            SourceItem(
                source_id=source_id,
                title="Mock RAG source",
                snippet="Retrieval grounds generated answers.",
                source_type="pdf_chunk",
                url=None,
                metadata={"page_start": 1},
            )
        ]
        trace = ["mock_evidence_returned"]

    return ResearchResponse(
        run_id="mock-run",
        corpus_version_id="mock-corpus",
        corpus_stats=CorpusStats(
            source_document_count=1,
            chunk_count=1,
            corpus_version_id="mock-corpus",
        ),
        question=question,
        answer=answer,
        sources=sources,
        evaluation=[],
        execution_trace=trace,
    )


if __name__ == "__main__":
    sys.exit(main())
