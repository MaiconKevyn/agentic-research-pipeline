from backend.app.api.routes.health import healthcheck


def test_healthcheck() -> None:
    assert healthcheck() == {"status": "ok"}
