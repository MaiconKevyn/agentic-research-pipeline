import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Research Agent")
    app_env: str = os.getenv("APP_ENV", "development")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_auth_token: str = os.getenv("API_AUTH_TOKEN", "")
    default_workspace_id: str = os.getenv("DEFAULT_WORKSPACE_ID", "default")
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "agentic-research-pipeline")
    otel_traces_enabled: bool = os.getenv("OTEL_TRACES_ENABLED", "true").lower() in {"1", "true", "yes"}
    worker_poll_interval_seconds: int = int(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "60"))
    raw_pdf_dir: str = os.getenv("RAW_PDF_DIR", str(REPO_ROOT / "data" / "raw"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "4000"))
    max_retrieved_tokens: int = int(os.getenv("MAX_RETRIEVED_TOKENS", "1800"))
    max_web_searches_per_run: int = int(os.getenv("MAX_WEB_SEARCHES_PER_RUN", "1"))
    max_estimated_model_cost_usd: float = float(os.getenv("MAX_ESTIMATED_MODEL_COST_USD", "0.05"))
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "600"))
    frontend_origins: str = os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    database_url: str = os.getenv("DATABASE_URL", "")
    database_host: str = os.getenv("DATABASE_HOST", "127.0.0.1")
    database_port: int = int(os.getenv("DATABASE_PORT", "55432"))
    database_name: str = os.getenv("DATABASE_NAME", "research_agent")
    database_user: str = os.getenv("DATABASE_USER", "postgres")
    database_password: str = os.getenv("DATABASE_PASSWORD", "postgres")

    @property
    def postgres_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]


settings = Settings()
