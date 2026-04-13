#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker compose up -d postgres

printf 'Waiting for postgres healthcheck'
for _ in $(seq 1 30); do
  STATUS="$(docker inspect --format='{{json .State.Health.Status}}' research-agent-postgres 2>/dev/null || true)"
  if [[ "$STATUS" == "\"healthy\"" ]]; then
    printf '\n'
    PYTHONPATH="$ROOT_DIR" ./.venv/bin/python scripts/seed_documents.py
    exit 0
  fi
  printf '.'
  sleep 2
done

printf '\nPostgres container did not become healthy in time.\n' >&2
exit 1
