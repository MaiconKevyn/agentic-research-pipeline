from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable

from backend.app.core.config import settings
from backend.app.schemas.research import SafetyDecision, SecurityFinding, SourceItem


PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "override the system",
    "override system",
    "jailbreak",
    "you are now",
)
SYSTEM_PROMPT_PATTERNS = (
    "system prompt",
    "developer message",
    "hidden prompt",
    "hidden prompts",
    "internal instructions",
)
TOOL_ABUSE_PATTERNS = (
    "call the web_search tool",
    "call web_search",
    "use the sql_query tool",
    "disable retrieval",
    "answer without retrieval",
    "fake citations",
)
DATA_EXFILTRATION_PATTERNS = (
    "exfiltrate",
    "api key",
    "secret key",
    "developer message",
    "private credentials",
)
PII_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}[-. ]?\d{2}[-. ]?\d{4}\b")
SECRET_PATTERN = re.compile(r"\b(?:sk|pk|ghp|lsv2)_[A-Za-z0-9_-]{4,}\b|\bsk-[A-Za-z0-9_-]{4,}\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_run_cost_usd(*, question: str, source_snippets: list[str], output_tokens: int = 900) -> float:
    input_tokens = estimate_tokens(question) + sum(estimate_tokens(snippet) for snippet in source_snippets)
    return round((input_tokens + output_tokens) / 1_000_000, 6)


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _finding(category: str, message: str, evidence: str | None = None, severity: str = "high") -> SecurityFinding:
    return SecurityFinding(
        category=category,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        evidence=evidence[:160] if evidence else None,
    )


def _unique_findings(findings: list[SecurityFinding]) -> list[SecurityFinding]:
    seen: set[str] = set()
    unique: list[SecurityFinding] = []
    for finding in findings:
        if finding.category in seen:
            continue
        seen.add(finding.category)
        unique.append(finding)
    return unique


class SecurityGuardrailService:
    def assess_input(self, text: str) -> SafetyDecision:
        findings: list[SecurityFinding] = []
        if len(text) > settings.max_input_chars:
            findings.append(
                _finding(
                    "resource_limit",
                    f"Input exceeds the configured {settings.max_input_chars} character limit.",
                    severity="high",
                )
            )
        if _contains_any(text, PROMPT_INJECTION_PATTERNS):
            findings.append(_finding("prompt_injection", "Input attempts to override assistant instructions.", text))
        if _contains_any(text, SYSTEM_PROMPT_PATTERNS):
            findings.append(_finding("system_prompt_leakage", "Input asks for hidden system or developer instructions.", text))
        if _contains_any(text, TOOL_ABUSE_PATTERNS):
            findings.append(_finding("tool_abuse", "Input attempts to override tool policy.", text))
        if _contains_any(text, DATA_EXFILTRATION_PATTERNS):
            findings.append(_finding("data_exfiltration", "Input asks for sensitive or hidden data.", text))

        findings = _unique_findings(findings)
        metadata = {
            "estimated_tokens": estimate_tokens(text),
            "max_input_chars": settings.max_input_chars,
        }
        if findings:
            return SafetyDecision(
                allowed=False,
                action="block",
                findings=findings,
                rationale="Input safety policy blocked the request before tool execution.",
                metadata=metadata,
            )
        return SafetyDecision(
            allowed=True,
            action="allow",
            findings=[],
            rationale="No input safety issues detected.",
            metadata=metadata,
        )

    def assess_retrieved_content(self, sources: list[SourceItem]) -> tuple[list[SourceItem], SafetyDecision]:
        all_findings: list[SecurityFinding] = []
        sanitized_sources: list[SourceItem] = []

        for source in sources:
            source_findings = self._find_retrieved_content_issues(source.snippet)
            if not source_findings:
                sanitized_sources.append(source)
                continue

            all_findings.extend(source_findings)
            sanitized_snippet = self._sanitize_snippet(source.snippet)
            sanitized_sources.append(
                source.model_copy(
                    update={
                        "snippet": sanitized_snippet,
                        "metadata": {
                            **source.metadata,
                            "security_filtered": True,
                            "security_findings": [finding.category for finding in source_findings],
                        },
                    }
                )
            )

        findings = _unique_findings(all_findings)
        action = "sanitize" if findings else "allow"
        decision = SafetyDecision(
            allowed=True,
            action=action,
            findings=findings,
            rationale=(
                "Retrieved content was sanitized because it contained instructions."
                if findings
                else "No retrieved-content safety issues detected."
            ),
            metadata={"source_count": len(sources), "max_retrieved_tokens": settings.max_retrieved_tokens},
        )
        return sanitized_sources, decision

    def assess_output(self, text: str) -> SafetyDecision:
        findings: list[SecurityFinding] = []
        if _contains_any(text, SYSTEM_PROMPT_PATTERNS) or SECRET_PATTERN.search(text):
            findings.append(_finding("system_prompt_leakage", "Output appears to leak hidden prompts or secrets.", text))
        if PII_PATTERN.search(text):
            findings.append(_finding("pii_leakage", "Output appears to contain personal data.", text))

        findings = _unique_findings(findings)
        if findings:
            return SafetyDecision(
                allowed=False,
                action="block",
                findings=findings,
                rationale="Output safety policy blocked a potentially sensitive answer.",
                metadata={},
            )
        return SafetyDecision(allowed=True, action="allow", findings=[], rationale="Output passed safety checks.", metadata={})

    def apply_retrieved_token_limit(self, sources: list[SourceItem]) -> tuple[list[SourceItem], SafetyDecision]:
        kept: list[SourceItem] = []
        used_tokens = 0
        for source in sources:
            source_tokens = estimate_tokens(source.snippet)
            if used_tokens + source_tokens > settings.max_retrieved_tokens:
                remaining = max(settings.max_retrieved_tokens - used_tokens, 0)
                if remaining <= 0:
                    break
                kept.append(
                    source.model_copy(
                        update={
                            "snippet": source.snippet[: remaining * 4].rstrip(),
                            "metadata": {
                                **source.metadata,
                                "retrieved_token_limit_truncated": True,
                            },
                        }
                    )
                )
                used_tokens = settings.max_retrieved_tokens
                break
            kept.append(source)
            used_tokens += source_tokens

        return kept, SafetyDecision(
            allowed=True,
            action="allow",
            findings=[],
            rationale="Retrieved token budget was applied.",
            metadata={"used_tokens": used_tokens, "max_retrieved_tokens": settings.max_retrieved_tokens},
        )

    def assess_model_budget(self, *, question: str, sources: list[SourceItem]) -> SafetyDecision:
        input_tokens = estimate_tokens(question) + sum(estimate_tokens(source.snippet) for source in sources)
        estimated_cost = estimate_run_cost_usd(
            question=question,
            source_snippets=[source.snippet for source in sources],
        )
        metadata = {
            "estimated_input_tokens": input_tokens,
            "estimated_model_cost_usd": round(estimated_cost, 6),
            "max_estimated_model_cost_usd": settings.max_estimated_model_cost_usd,
        }
        if estimated_cost > settings.max_estimated_model_cost_usd:
            return SafetyDecision(
                allowed=False,
                action="block",
                findings=[
                    _finding(
                        "resource_limit",
                        "Estimated model cost exceeds the configured per-run budget.",
                        severity="high",
                    )
                ],
                rationale="Model budget policy blocked synthesis.",
                metadata=metadata,
            )
        return SafetyDecision(
            allowed=True,
            action="allow",
            findings=[],
            rationale="Model budget policy allowed synthesis.",
            metadata=metadata,
        )

    def _find_retrieved_content_issues(self, text: str) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        if _contains_any(text, PROMPT_INJECTION_PATTERNS):
            findings.append(_finding("malicious_document", "Retrieved content contains instruction override text.", text))
        if _contains_any(text, SYSTEM_PROMPT_PATTERNS):
            findings.append(_finding("system_prompt_leakage", "Retrieved content asks for hidden prompts.", text))
        if _contains_any(text, TOOL_ABUSE_PATTERNS):
            findings.append(_finding("tool_abuse", "Retrieved content attempts to control tool policy.", text))
        return _unique_findings(findings)

    def _sanitize_snippet(self, text: str) -> str:
        safe_sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_PATTERN.split(text.strip())
            if sentence.strip() and not self._find_retrieved_content_issues(sentence)
        ]
        return " ".join(safe_sentences).strip() or "[untrusted instructions removed]"


class TokenBucketRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str) -> SafetyDecision:
        now = time.monotonic()
        requests = self._requests[client_id]
        while requests and now - requests[0] >= self.window_seconds:
            requests.popleft()
        if len(requests) >= self.max_requests:
            return SafetyDecision(
                allowed=False,
                action="block",
                findings=[
                    _finding(
                        "rate_limit",
                        f"Client exceeded {self.max_requests} requests per {self.window_seconds} seconds.",
                        severity="high",
                    )
                ],
                rationale="Rate limit exceeded.",
                metadata={"max_requests": self.max_requests, "window_seconds": self.window_seconds},
            )
        requests.append(now)
        return SafetyDecision(
            allowed=True,
            action="allow",
            findings=[],
            rationale="Request allowed by rate limit policy.",
            metadata={
                "remaining_requests": self.max_requests - len(requests),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
            },
        )


security_guardrail = SecurityGuardrailService()
rate_limiter = TokenBucketRateLimiter(
    max_requests=settings.rate_limit_requests_per_minute,
    window_seconds=60,
)
