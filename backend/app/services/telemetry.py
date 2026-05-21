from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import settings


@dataclass
class TelemetrySpan:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: float = field(default_factory=time.perf_counter)
    attributes: dict[str, Any] = field(default_factory=dict)

    def finish(self, **attributes: Any) -> list[str]:
        elapsed_ms = int((time.perf_counter() - self.started_at) * 1000)
        merged = {**self.attributes, **attributes}
        trace = [
            f"otel_service={settings.otel_service_name}",
            f"otel_span={self.name}",
            f"otel_span_id={self.span_id}",
            f"otel_duration_ms={elapsed_ms}",
        ]
        for key, value in sorted(merged.items()):
            if value is not None:
                trace.append(f"otel_attr_{key}={value}")
        return trace if settings.otel_traces_enabled else []


def start_span(name: str, **attributes: Any) -> TelemetrySpan:
    return TelemetrySpan(name=name, attributes=attributes)
