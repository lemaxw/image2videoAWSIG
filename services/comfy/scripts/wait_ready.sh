#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://127.0.0.1:8188/system_stats}"
MAX_TRIES="${2:-90}"

for i in $(seq 1 "$MAX_TRIES"); do
  if curl -fsS "$URL" >/dev/null; then
    echo "ComfyUI is ready"
    exit 0
  fi
  sleep 5
done

echo "Timed out waiting for ComfyUI" >&2
exit 1
