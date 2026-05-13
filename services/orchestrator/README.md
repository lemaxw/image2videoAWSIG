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
- `--animation-directions` append extra motion/style directions to video prompts
- `--debug` keep intermediate artifacts
- `--max-fail-ratio` (default `0.3`)

## Default video params

Applied on every run (unless overridden by `--video-params-json`):

```json
{"render_variants":"selected_pair"}
```

`seed` is auto-generated unless explicitly provided.

## Current pipeline

The current pipeline has three backend families:

1. Hunyuan 1.5 presets: cropped input image, or original image for wide pan scenes -> direct image-to-video
2. SVD presets: cropped input image -> direct SVD img2vid
3. AnimateDiff presets: cropped input image -> SD 1.5 + AnimateDiff motion module

By default each image renders the selected primary preset plus the first fallback from a different backend family.
For wide compositions where a vertical crop loses the story, Hunyuan can use the original image when `video.params.use_original_input_for_video=true` or when a moon + skyline/building scene suggests a lateral pan.

OpenAI selects the preset. Local defaults are quality defaults and clamp per backend.

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

JSON override users can also pass `animation_directions` or `render_variants` inside `--video-params-json`.
