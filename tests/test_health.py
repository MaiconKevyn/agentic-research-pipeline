from backend.app.api.routes.health import healthcheck


def test_healthcheck() -> None:
    response = healthcheck()

    assert response["status"] == "ok"
    assert response["environment"]
    assert response["version"]
