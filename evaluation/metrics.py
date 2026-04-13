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
