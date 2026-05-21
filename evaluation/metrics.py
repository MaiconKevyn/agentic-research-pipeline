from math import log2

from backend.app.schemas.research import ResearchResponse


def citation_coverage(response: ResearchResponse) -> float:
    return 1.0 if response.sources else 0.0


def answer_completeness(response: ResearchResponse) -> float:
    return 1.0 if response.answer.strip() else 0.0


def overall_score(response: ResearchResponse) -> float:
    scores = [
        citation_coverage(response),
        answer_completeness(response),
    ]
    return sum(scores) / len(scores)


def recall_at_k(
    ranked_source_ids: list[str],
    required_source_ids: set[str],
    k: int,
) -> float:
    if not required_source_ids:
        return 1.0
    retrieved = set(ranked_source_ids[:k])
    return len(retrieved & required_source_ids) / len(required_source_ids)


def mean_reciprocal_rank(
    ranked_source_ids: list[str],
    required_source_ids: set[str],
) -> float:
    if not required_source_ids:
        return 1.0
    for index, source_id in enumerate(ranked_source_ids, start=1):
        if source_id in required_source_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    ranked_source_ids: list[str],
    required_source_ids: set[str],
    k: int,
) -> float:
    if not required_source_ids:
        return 1.0
    dcg = 0.0
    for index, source_id in enumerate(ranked_source_ids[:k], start=1):
        if source_id in required_source_ids:
            dcg += 1.0 / log2(index + 1)
    ideal_relevant_count = min(len(required_source_ids), k)
    ideal_dcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_relevant_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
