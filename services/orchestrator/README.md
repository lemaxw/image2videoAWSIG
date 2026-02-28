# Orchestrator Service

## Entry point

`run_batch.py` drives the full pipeline for a batch:

1. List image keys from S3 (or local folder in dry-run mode)
2. Download image
3. Call OpenAI decision service for structured decision JSON
4. Render with ComfyUI (with fallbacks)
5. Generate audio from local audio service
6. Mux video+audio with ffmpeg
7. Upload `final.mp4` + `debug.json`

## CLI

```bash
python services/orchestrator/run_batch.py \
  --job-id demo-001 \
  --input-bucket my-input \
  --input-prefix jobs/demo-001/input \
  --output-bucket my-output \
  --output-prefix jobs/demo-001/output
```

## Dry run

```bash
python services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-prefix . \
  --output-prefix out \
  --dry-run \
  --local-input-dir ./local_inputs \
  --local-output-dir ./local_outputs
```
