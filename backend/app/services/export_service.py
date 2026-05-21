from __future__ import annotations

import csv
import io

from backend.app.schemas.research import WorkspaceRun


def export_run_markdown(run: WorkspaceRun) -> str:
    lines = [
        f"# Research Run {run.run_id}",
        "",
        f"- question: {run.question}",
        f"- answer_mode: {run.answer_mode}",
        f"- corpus_version_id: {run.corpus_version_id}",
        f"- created_at: {run.created_at}",
        "",
        "## Answer",
        "",
        run.answer,
        "",
        "## Sources",
        "",
    ]
    for source in run.sources:
        lines.extend(
            [
                f"### {source.source_id}: {source.title}",
                "",
                f"- type: {source.source_type}",
                f"- page_start: {source.metadata.get('page_start', '')}",
                f"- chunk_id: {source.metadata.get('chunk_id', source.source_id)}",
                "",
                source.snippet,
                "",
            ]
        )
    lines.extend(
        [
            "## Replay",
            "",
            f"- question: {run.replay_request.question}",
            f"- top_k: {run.replay_request.top_k}",
            f"- answer_mode: {run.replay_request.answer_mode}",
            "",
        ]
    )
    return "\n".join(lines)


def export_run_csv(run: WorkspaceRun) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "run_id",
            "source_id",
            "title",
            "source_type",
            "page_start",
            "chunk_id",
            "snippet",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for source in run.sources:
        writer.writerow(
            {
                "run_id": run.run_id,
                "source_id": source.source_id,
                "title": source.title,
                "source_type": source.source_type,
                "page_start": source.metadata.get("page_start", ""),
                "chunk_id": source.metadata.get("chunk_id", source.source_id),
                "snippet": source.snippet,
            }
        )
    return output.getvalue()


def export_run_json(run: WorkspaceRun) -> str:
    return run.model_dump_json(indent=2)
