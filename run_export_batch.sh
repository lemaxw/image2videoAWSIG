#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO_ROOT/.env"
INPUT_DIR="$REPO_ROOT/video_input"
OUTPUT_DIR="$REPO_ROOT/video_output"
SOURCE_DIR="${SOURCE_DIR:-/mnt/c/Documents and Settings/mpshater/Pictures/export/e}"
JOB_ID="${JOB_ID:-dry-008}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-out}"

COMPOSE_CMD=(
  docker compose
  --env-file "$ENV_FILE"
  -f "$REPO_ROOT/services/comfy/docker-compose.yml"
  -f "$REPO_ROOT/services/comfy/docker-compose.gpu.yml"
)

clear_dir() {
  local dir="$1"

  if ! find "$dir" -mindepth 1 -delete; then
    echo "Direct delete failed for $dir, retrying via Docker root helper" >&2
    if ! docker run --rm -v "$dir:/target" alpine:3.20 sh -c 'find /target -mindepth 1 -delete'; then
      echo "Failed to delete existing contents of $dir" >&2
      exit 1
    fi
  fi

  if find "$dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Directory is not empty after delete attempt: $dir" >&2
    exit 1
  fi
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

env_input_dir="$(grep -E '^LOCAL_INPUT_DIR=' "$ENV_FILE" | cut -d= -f2- || true)"
env_output_dir="$(grep -E '^LOCAL_OUTPUT_DIR=' "$ENV_FILE" | cut -d= -f2- || true)"

if [[ "$env_input_dir" != "$INPUT_DIR" ]]; then
  echo "LOCAL_INPUT_DIR in .env does not match $INPUT_DIR" >&2
  echo "Found: ${env_input_dir:-<unset>}" >&2
  exit 1
fi

if [[ "$env_output_dir" != "$OUTPUT_DIR" ]]; then
  echo "LOCAL_OUTPUT_DIR in .env does not match $OUTPUT_DIR" >&2
  echo "Found: ${env_output_dir:-<unset>}" >&2
  exit 1
fi

clear_dir "$INPUT_DIR"
clear_dir "$OUTPUT_DIR"

mapfile -d '' jpg_files < <(find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | sort -z)

if [[ "${#jpg_files[@]}" -eq 0 ]]; then
  echo "No JPG files found in $SOURCE_DIR" >&2
  exit 1
fi

for src in "${jpg_files[@]}"; do
  cp -p "$src" "$INPUT_DIR/"
done

"${COMPOSE_CMD[@]}" up -d

EXEC_ARGS=(
  pipeline-orchestrator
  python
  /app/services/orchestrator/run_batch.py
  --job-id "$JOB_ID"
  --input-prefix .
  --output-prefix "$OUTPUT_PREFIX"
  --local-input-dir /data/local_inputs
  --local-output-dir /data/local_outputs
)

if [[ -t 0 && -t 1 ]]; then
  EXEC_CMD=(docker exec -it "${EXEC_ARGS[@]}")
else
  EXEC_CMD=(docker exec -i "${EXEC_ARGS[@]}")
fi

"${EXEC_CMD[@]}"
