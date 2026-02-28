#!/usr/bin/env bash
set -euo pipefail
curl -fsS http://localhost:8188/system_stats >/dev/null
