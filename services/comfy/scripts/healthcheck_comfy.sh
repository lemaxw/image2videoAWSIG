#!/usr/bin/env bash
set -euo pipefail
for _ in 1 2 3; do
  if curl -fsS http://localhost:8188/system_stats >/dev/null; then
    exit 0
  fi
  sleep 1
done
exit 1
