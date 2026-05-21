from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backend.app.api.auth import require_api_key
from backend.app.api.routes.health import healthcheck, readiness
from backend.app.api.routes.operations import get_run_metrics
from backend.app.schemas.research import RunMetricsSummary


def test_readiness_reports_database_ready() -> None:
    with patch("backend.app.api.routes.health.get_corpus_stats") as mocked_stats:
        mocked_stats.return_value = SimpleNamespace(
            source_document_count=2,
            chunk_count=10,
            corpus_version_id="corpus-v1",
        )
        response = readiness()

    assert response["status"] == "ready"
    assert response["checks"]["database"] == "ok"
    assert response["corpus"]["chunk_count"] == 10


def test_readiness_reports_database_unavailable() -> None:
    with patch("backend.app.api.routes.health.get_corpus_stats", side_effect=RuntimeError("db down")):
        response = readiness()

    assert response["status"] == "not_ready"
    assert response["checks"]["database"] == "error"
    assert "db down" in response["errors"][0]


def test_healthcheck_includes_environment_and_version() -> None:
    response = healthcheck()

    assert response["status"] == "ok"
    assert "environment" in response
    assert "version" in response


def test_require_api_key_accepts_open_endpoint_when_token_unset() -> None:
    with patch("backend.app.api.auth.settings", SimpleNamespace(api_auth_token="")):
        assert require_api_key(api_key=None) is None


def test_require_api_key_rejects_invalid_token_when_configured() -> None:
    with patch("backend.app.api.auth.settings", SimpleNamespace(api_auth_token="secret")):
        try:
            require_api_key(api_key="wrong")
        except HTTPException as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError("Expected invalid API key to raise HTTPException")


def test_operations_route_returns_run_metrics() -> None:
    metrics = RunMetricsSummary(
        run_count=3,
        failure_count=1,
        average_latency_ms=125.0,
        average_cost_estimate_usd=0.002,
        average_scores={"groundedness": 0.95},
        runs_by_day=[{"date": "2026-05-21", "count": 3}],
        quality_trend=[{"date": "2026-05-21", "groundedness": 0.95}],
    )
    with patch("backend.app.api.routes.operations.get_run_metrics_summary", return_value=metrics):
        response = get_run_metrics(days=7)

    assert response.run_count == 3
    assert response.failure_count == 1
    assert response.average_scores["groundedness"] == 0.95
