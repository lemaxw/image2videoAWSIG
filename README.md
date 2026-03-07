# Image-to-Video Local Pipeline

Local-only image-to-video pipeline (no AWS/S3/AMI/infra flow).

## Services

- `services/decision`:
  - Sends image + prompt to OpenAI and returns structured decision JSON:
    - scene (`tags`, `has_people`, `confidence`)
    - framing (`target_aspect`, `crop_anchor`)
    - video preset/fallbacks
    - audio prompt/mix
- `services/comfy`:
  - Runs ComfyUI and executes render workflows.
  - Uses workflow templates in `services/comfy/workflow_templates/`.
  - Outputs rendered media to `/data/outputs/comfy`.
- `services/audio`:
  - FastAPI service for audio generation.
  - Backends:
    - `audioldm` (model generation)
    - `mock` / `mock-fallback` (simple synthetic tone/noise fallback)
  - AudioLDM path generates multiple candidates, selects the best via RMS/clipping score,
    then applies ffmpeg post-processing (`loudnorm`, limiter, bass, stereo widen, reverb, fade).
  - Final output wav is post-processed to `48kHz` stereo.
  - Outputs wav files to `/data/outputs/audio`.
- `services/orchestrator`:
  - Main batch runner (`run_batch.py`).
  - Loads local images, calls decision service, crops input for IG 9:16,
    renders with Comfy, generates audio, muxes final mp4, writes `debug.json`.

## Prerequisites

- Docker + Docker Compose plugin
- NVIDIA driver + Docker GPU runtime (for GPU render/audio)
- Local folders:
  - input images: `video_input/`
  - outputs: `video_output/`
  - models: `.local/models/`
  - render output/cache: `.local/outputs/`
  - audio cache: `.local/audio-cache/`

## Model Preparation

Create checkpoint folder:

```bash
mkdir -p "${MODEL_DIR:-$PWD/.local/models}/checkpoints"
```

Download required checkpoints:

```bash
curl -L "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true" \
  -o "${MODEL_DIR:-$PWD/.local/models}/checkpoints/svd_xt.safetensors"

curl -L "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true" \
  -o "${MODEL_DIR:-$PWD/.local/models}/checkpoints/sd_xl_base_1.0.safetensors"
```

Verify:

```bash
ls -lh "${MODEL_DIR:-$PWD/.local/models}/checkpoints"
```

## Environment Variables

Set these in `.env`:

- `LOCAL_INPUT_DIR`: host path mounted to `/data/local_inputs`
- `LOCAL_OUTPUT_DIR`: host path mounted to `/data/local_outputs`
- `MODEL_DIR`: host path mounted to `/opt/ComfyUI/models`
- `OUTPUT_DIR`: host path mounted to `/data/outputs`
- `AUDIO_CACHE_DIR`: host path mounted to `/cache`
- `AUDIO_HOST_PORT`: local forwarded audio port (example `8001`)
- `OPENAI_API_KEY`: decision API key
- `OPENAI_MODEL`: decision model (example `gpt-5-mini`)
- `AUDIO_MODEL_BACKEND`: `audioldm` or `mock`
- `AUDIO_DEVICE`: `cuda` or `cpu`
- `AUDIO_INFERENCE_STEPS`: AudioLDM inference steps (default `60`)
- `AUDIO_GUIDANCE_SCALE`: AudioLDM guidance scale (default `3.5`)
- `AUDIO_NUM_SAMPLES`: candidates generated per prompt (default `3`)
- `AUDIO_SEED_BASE`: deterministic seed base (default `42`)
- `AUDIO_TARGET_LUFS`: loudness target for `loudnorm` (default `-16`)
- `AUDIO_TRUE_PEAK_DB`: true-peak ceiling in dB (default `-1.0`)
- `AUDIO_BASS_GAIN_DB`: bass enhancement gain (default `3`)
- `AUDIO_STEREO_MLEV`: stereo widening amount (default `0.03`)
- `AUDIO_REVERB_DELAY_MS`: reverb delay in ms (default `1000`)
- `AUDIO_REVERB_DECAY`: reverb decay (default `0.3`)

Container runtime env is set in compose:
- `COMFY_URL=http://comfyui:8188`
- `AUDIO_URL=http://audio:8000`
- `INPUT_DIR=/data/inputs`
- `OUTPUT_DIR=/data/outputs`

## Start Stack

```bash
docker compose --env-file $(pwd)/.env \
  -f services/comfy/docker-compose.yml \
  -f services/comfy/docker-compose.gpu.yml \
  up -d --build
```

If models changed, restart Comfy:

```bash
docker compose --env-file $(pwd)/.env \
  -f services/comfy/docker-compose.yml \
  -f services/comfy/docker-compose.gpu.yml \
  restart comfyui
```

If `.env` changed for audio backend/device, recreate audio (restart is not enough):

```bash
docker compose --env-file $(pwd)/.env \
  -f services/comfy/docker-compose.yml \
  -f services/comfy/docker-compose.gpu.yml \
  up -d --force-recreate audio
```

## Run Batch

```bash
docker exec pipeline-orchestrator python /app/services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-prefix . \
  --output-prefix out \
  --local-input-dir /data/local_inputs \
  --local-output-dir /data/local_outputs
```

## Defaults

Applied automatically when no explicit override is provided:

```json
{"fps":5,"frames":25,"resolution_width":768,"steps":24,"motion_bucket_id":24,"seed":123}
```

Override at runtime:

```bash
--video-params-json '{"fps":5,"frames":25,"resolution_width":768,"steps":24,"motion_bucket_id":24,"seed":123,"crop_anchor":"center_center"}'
```

## Debug Mode

- `debug.json` is always saved.
- Intermediate artifacts are kept only with `--debug`.

Example:

```bash
docker exec pipeline-orchestrator python /app/services/orchestrator/run_batch.py \
  --job-id dry-001 \
  --input-prefix . \
  --output-prefix out \
  --local-input-dir /data/local_inputs \
  --local-output-dir /data/local_outputs \
  --debug
```

## Outputs

- `video_output/out/<job-id>/<image-basename>/final.mp4`
- `video_output/out/<job-id>/<image-basename>/debug.json`

## Diagnostics

Comfy:

```bash
docker logs pipeline-comfyui --tail 120
```

Audio:

```bash
docker logs pipeline-audio --tail 120
docker exec pipeline-audio /bin/sh -lc 'env | grep ^AUDIO_'
```

Orchestrator:

```bash
docker logs pipeline-orchestrator --tail 120
```

## Stop

```bash
docker compose --env-file $(pwd)/.env \
  -f services/comfy/docker-compose.yml \
  -f services/comfy/docker-compose.gpu.yml \
  down
```
