from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.app.schemas.research import SourceItem
from backend.app.services.llm_service import generate_research_answer


def test_generate_research_answer_uses_structured_output_and_validates_schema() -> None:
    mocked_response = Mock()
    mocked_response.raise_for_status.return_value = None
    mocked_response.json.return_value = {
        "output_text": (
            '{"answer_summary":"Objective answer.","confidence":"medium",'
            '"claims":[{"claim_text":"Objective answer.","supporting_source_ids":["source-1"],'
            '"supporting_quotes":[{"source_id":"source-1","quote":"Important passage"}],'
            '"confidence":"medium","limitations":[],"conflicts":[],"support_status":"supported"}],'
            '"limitations":[],"conflicts":[],"follow_up_questions":[],"uncertainty_note":null}'
        )
    }

    with (
        patch(
            "backend.app.services.llm_service.settings",
            SimpleNamespace(
                openai_api_key="test-key",
                openai_base_url="https://api.openai.com/v1",
                openai_model="gpt-4o-mini",
                request_timeout_seconds=60,
            ),
        ),
        patch("backend.app.services.llm_service.requests.post", return_value=mocked_response) as mocked_post,
    ):
        synthesis = generate_research_answer(
            question="What is the summary?",
            sources=[
                SourceItem(
                    source_id="source-1",
                    title="Document",
                    snippet="Important passage from the document.",
                    source_type="pdf_chunk",
                    url=None,
                    metadata={},
                )
            ],
        )

    _, kwargs = mocked_post.call_args
    assert kwargs["json"]["text"]["format"]["type"] == "json_schema"
    assert kwargs["json"]["text"]["format"]["strict"] is True
    assert synthesis.answer_summary == "Objective answer."
    assert synthesis.claims[0].supporting_source_ids == ["source-1"]
    assert synthesis.claims[0].supporting_quotes[0].quote == "Important passage"
