import json
from pathlib import Path

from agent.graph import run_research
from evaluation.metrics import overall_score


DATASET_PATH = Path(__file__).parent / "datasets" / "sample_questions.json"


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def run_benchmark() -> list[dict]:
    results: list[dict] = []
    for item in load_dataset():
        response = run_research(question=item["question"], top_k=item.get("top_k", 5))
        results.append(
            {
                "question": item["question"],
                "overall_score": overall_score(response),
                "answer": response.answer,
                "trace": response.execution_trace,
            }
        )
    return results
