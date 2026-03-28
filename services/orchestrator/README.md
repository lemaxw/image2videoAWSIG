# Orchestrator (Local Only)

Entry point: `services/orchestrator/run_batch.py`

## Required args

- `--job-id`
- `--input-prefix`
- `--output-prefix`
- `--local-input-dir`
- `--local-output-dir`

## Optional args

- `--input-file` process exactly one file relative to `--local-input-dir`
- `--video-params-json` (JSON object)
- `--animation-directions` append extra animation/style directions to the anime redraw prompt
- `--debug` keep intermediate artifacts
- `--max-fail-ratio` (default `0.3`)

## Default video params

Applied on every run (unless overridden by `--video-params-json`):

```json
{"fps":5,"frames":25,"resolution_width":768,"steps":24,"motion_bucket_id":24}
```

`seed` is auto-generated unless explicitly provided.

## Current pipeline

Every preset now runs the same two-stage flow:

1. input image -> anime redraw still
2. anime still -> SVD img2vid

Preset names act as styling profiles, not separate backends.

## Example

```bash
python /app/services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-prefix . \
  --output-prefix out \
  --local-input-dir /data/local_inputs \
  --local-output-dir /data/local_outputs
```

Single file example:

```bash
python /app/services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-file _MG_6609.jpg \
  --output-prefix out \
  --local-input-dir /data/local_inputs \
  --local-output-dir /data/local_outputs
```

Example with extra animation directions:

```bash
python /app/services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-file _MG_6609.jpg \
  --output-prefix out \
  --local-input-dir /data/local_inputs \
  --local-output-dir /data/local_outputs \
  --animation-directions "subtle hair movement, gentle camera drift, breeze through clothing"
```

JSON override users can also pass `anime_prompt_hint` or `animation_directions` inside `--video-params-json`.
