#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR="${MODEL_DIR:-/data/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs}"
export AUDIO_CACHE_DIR="${AUDIO_CACHE_DIR:-/data/audio-cache}"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR" "$AUDIO_CACHE_DIR"
docker compose -f services/comfy/docker-compose.yml up -d --build
bash services/comfy/scripts/wait_ready.sh
