from __future__ import annotations

from backend.app.schemas.research import AnswerClaim, SourceItem, SynthesisOutput


def verify_synthesis_claims(
    *,
    synthesis: SynthesisOutput,
    sources: list[SourceItem],
) -> SynthesisOutput:
    if not synthesis.claims:
        return synthesis

    valid_source_ids = {source.source_id for source in sources}
    supported_claims: list[AnswerClaim] = []
    removed_limitations: list[str] = []

    for claim in synthesis.claims:
        valid_claim_source_ids = [
            source_id for source_id in claim.supporting_source_ids if source_id in valid_source_ids
        ]
        valid_quotes = [
            quote for quote in claim.supporting_quotes if quote.source_id in valid_source_ids
        ]
        support_status = _support_status_for_claim(
            claim=claim,
            valid_source_ids=valid_claim_source_ids,
            valid_quote_count=len(valid_quotes),
        )
        verified_claim = claim.model_copy(
            update={
                "supporting_source_ids": valid_claim_source_ids,
                "supporting_quotes": valid_quotes,
                "support_status": support_status,
            }
        )
        if support_status == "supported":
            supported_claims.append(verified_claim)
        else:
            removed_limitations.append(f"Removed unsupported claim: {claim.claim_text}")

    answer_summary = _build_verified_summary(supported_claims)
    limitations = [*synthesis.limitations, *removed_limitations]
    if not answer_summary:
        answer_summary = "Insufficient evidence: all synthesized claims were unsupported."
        limitations = [*limitations, "No supported claims remained after verification."]

    confidence = synthesis.confidence if supported_claims else "low"
    return synthesis.model_copy(
        update={
            "answer_summary": answer_summary,
            "claims": supported_claims,
            "confidence": confidence,
            "limitations": limitations,
        }
    )


def _support_status_for_claim(
    *,
    claim: AnswerClaim,
    valid_source_ids: list[str],
    valid_quote_count: int,
) -> str:
    if claim.conflicts:
        return "conflicting"
    if not valid_source_ids or valid_quote_count == 0:
        return "unsupported"
    return "supported"


def _build_verified_summary(claims: list[AnswerClaim]) -> str:
    return " ".join(claim.claim_text.strip() for claim in claims if claim.claim_text.strip()).strip()
