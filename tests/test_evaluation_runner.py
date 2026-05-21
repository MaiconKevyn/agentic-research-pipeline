from pathlib import Path

from backend.app.schemas.research import CorpusStats, ResearchResponse
from evaluation.runner import run_evaluation


def test_run_evaluation_scores_golden_cases_and_writes_jsonl_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"rag-basics","question":"What does the corpus say about RAG?",'
                    '"expected_answer_type":"factual","required_sources":["source-1"],'
                    '"expected_facts":["retrieval"],"forbidden_claims":["fine-tuning is required"],'
                    '"answer_rubric":"Answer from corpus evidence.","difficulty":"easy",'
                    '"query_category":"internal_corpus_factual_qa"}'
                ),
                (
                    '{"id":"off-topic","question":"Who won the football match?",'
                    '"expected_answer_type":"insufficient_evidence","required_sources":[],'
                    '"expected_facts":[],"forbidden_claims":["football"],'
                    '"answer_rubric":"Decline out-of-scope requests.","difficulty":"easy",'
                    '"query_category":"insufficient_evidence"}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.jsonl"

    def fake_run(question: str, top_k: int = 5) -> ResearchResponse:
        if "football" in question:
            return ResearchResponse(
                run_id="run-2",
                corpus_version_id="corpus-v1",
                corpus_stats=CorpusStats(source_document_count=1, chunk_count=3, corpus_version_id="corpus-v1"),
                question=question,
                answer="Insufficient evidence: this request is outside the project scope.",
                sources=[],
                evaluation=[],
                execution_trace=["scope_guardrail_triggered"],
            )
        return ResearchResponse(
            run_id="run-1",
            corpus_version_id="corpus-v1",
            corpus_stats=CorpusStats(source_document_count=1, chunk_count=3, corpus_version_id="corpus-v1"),
            question=question,
            answer="The corpus says retrieval grounds RAG answers in source evidence [source-1].",
            sources=[
                {
                    "source_id": "source-1",
                    "title": "RAG Source",
                    "snippet": "Retrieval grounds generated answers.",
                    "source_type": "pdf_chunk",
                    "url": None,
                    "metadata": {},
                }
            ],
            evaluation=[],
            execution_trace=[],
        )

    summary = run_evaluation(dataset_path=dataset_path, output_path=report_path, run_func=fake_run)

    assert summary.case_count == 2
    assert summary.thresholds_passed is True
    assert summary.metrics["schema_validity"] == 1.0
    assert summary.metrics["scope_compliance"] == 1.0
    assert summary.metrics["citation_precision"] == 1.0
    assert summary.metrics["retrieval_mrr"] == 1.0
    assert summary.metrics["retrieval_ndcg_at_10"] == 1.0
    assert report_path.exists()
    assert len(report_path.read_text(encoding="utf-8").strip().splitlines()) == 2
