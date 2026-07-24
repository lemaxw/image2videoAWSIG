#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_runner="$(command -v uv || true)"

if [[ -z "$uv_runner" ]]; then
  echo "uv is required to launch the pipeline MCP server." >&2
  exit 1
fi

cd "$repo_root"
exec "$uv_runner" run --with 'mcp>=1.27,<2' python -m services.pipeline_mcp.server

