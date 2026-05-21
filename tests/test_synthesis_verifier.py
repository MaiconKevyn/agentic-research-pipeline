from backend.app.schemas.research import AnswerClaim, ClaimEvidence, SourceItem, SynthesisOutput
from backend.app.services.synthesis_verifier import verify_synthesis_claims


def test_synthesis_verifier_marks_unsupported_claims_and_filters_them_from_summary() -> None:
    synthesis = SynthesisOutput(
        answer_summary="Supported fact. Unsupported fact.",
        confidence="high",
        claims=[
            AnswerClaim(
                claim_text="Supported fact.",
                supporting_source_ids=["source-1"],
                supporting_quotes=[ClaimEvidence(source_id="source-1", quote="Supported fact appears here.")],
                confidence="high",
            ),
            AnswerClaim(
                claim_text="Unsupported fact.",
                supporting_source_ids=[],
                supporting_quotes=[],
                confidence="high",
            ),
        ],
        limitations=[],
        conflicts=[],
        follow_up_questions=[],
        uncertainty_note=None,
    )
    sources = [
        SourceItem(
            source_id="source-1",
            title="Evidence",
            snippet="Supported fact appears here.",
            source_type="pdf_chunk",
            url=None,
            metadata={"page_start": 2},
        )
    ]

    verified = verify_synthesis_claims(synthesis=synthesis, sources=sources)

    assert [claim.claim_text for claim in verified.claims] == ["Supported fact."]
    assert verified.claims[0].support_status == "supported"
    assert verified.answer_summary == "Supported fact."
    assert verified.limitations == ["Removed unsupported claim: Unsupported fact."]
