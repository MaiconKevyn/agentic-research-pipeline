from evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_retrieval_metrics_score_ranked_required_sources() -> None:
    ranked_source_ids = ["source-a", "source-b", "source-c"]
    required_source_ids = {"source-b", "source-d"}

    assert recall_at_k(ranked_source_ids, required_source_ids, k=3) == 0.5
    assert mean_reciprocal_rank(ranked_source_ids, required_source_ids) == 0.5
    assert round(ndcg_at_k(ranked_source_ids, required_source_ids, k=3), 4) == 0.3869
