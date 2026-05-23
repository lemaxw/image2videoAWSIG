# Agent Notes

Read [README.md]($HOME/hobby/image2videoAWSIG/README.md) first.

`README.md` is the source of truth for:
- service explanations
- local setup and model download
- required environment variables
- run/debug commands
- diagnostics and stop commands

## Agent-specific reminders

- Project is local-only (no AWS/S3/AMI/infra flow).
- `services/orchestrator/run_batch.py` is local-only and requires:
  - `--local-input-dir`
  - `--local-output-dir`
- Default video params are applied automatically:
  - The decision service selects `preset`; local defaults do not force it.
  - Preset defaults supply `fps`, `frames`, and `resolution_width`.
  - `render_variants=selected_pair` unless overridden.
  - Backend-specific defaults are clamped in `services/orchestrator/validate.py`.
  - `seed` is auto-generated unless explicitly provided.
- `debug.json` is always saved.
- Intermediate artifacts are only preserved with `--debug`.
- If `.env` audio backend/device values change, recreate `audio` container (not restart).
- `use_original_input_for_video=true` can apply beyond Hunyuan; final 9:16 output should pan/crop original video, not pad it with margins.

## Decision service: image2json + Ollama text model

The decision service now uses a two-step local flow instead of OpenAI:

1. **Step 1**: `image2json` (vision model `qwen3-vl:8b`) analyzes the image and returns structured JSON
2. **Step 2**: Ollama text model (`qwen3:14b`) processes the image2json JSON and outputs the decision JSON

**Environment variables** (in `.env`):
- `IMAGE2JSON_ENABLED`: enable image2json decision integration (default `true`)
- `IMAGE2JSON_URL`: Ollama URL for image2json (default `http://host.docker.internal:11434`)
- `IMAGE2JSON_MODEL`: image2json vision model (default `qwen3-vl:8b`)
- `IMAGE2JSON_TEXT_MODEL`: Ollama text model for decision (default `qwen3:14b`)
- `IMAGE2JSON_TIMEOUT`: image2json analysis timeout in seconds (default `300`)

**Behavior**:
- If `IMAGE2JSON_ENABLED=true` and image2json is installed, the two-step flow is used
- If either step fails, the system exits (no fallback to OpenAI)
- `IMAGE2JSON_MODEL` must remain the vision model `qwen3-vl:8b`; `IMAGE2JSON_TEXT_MODEL` is the text model `qwen3:14b`
- The text model receives both the `TEXT_MODEL_SYSTEM_PROMPT` and `DECISION_SCHEMA` in the prompt
- Decision metadata includes `image2json` section with `vision_model`, `text_model`, `analysis`, and `text_response`
- Ollama vision/text models are unloaded between decision steps so they do not keep GPU memory for later stages

## Runtime logging and model handoff

- Each major orchestrator step prints structured `step.start`, `step.done`, or `step.failed` logs.
- Done logs include elapsed seconds and result details; failed logs include exception type and message.
- Before Comfy rendering, the orchestrator stops the audio container through Docker when `MODEL_SERVICE_CONTROL=docker`.
- Before Comfy rendering, the orchestrator starts Comfy if it was stopped after a previous render.
- After Comfy rendering and before audio generation, the orchestrator stops Comfy through Docker; if Docker socket access is unavailable it falls back to `/free`.
- Before audio generation, the orchestrator starts the audio container again and waits for `/health`.
- After audio generation, the orchestrator stops audio again; if Docker socket access is unavailable it falls back to `/unload`.

**Docker setup**:
- image2json is mounted as a read-only volume in `docker-compose.yml`
- `PYTHONPATH` includes `/app/image2json/src` for imports
- `/var/run/docker.sock` is mounted into orchestrator so it can stop/start `pipeline-audio` and `pipeline-comfyui` for RAM handoff

## MemPalace project memory

For nontrivial debugging, pipeline questions, regressions, or decisions that depend on
project history, consult MemPalace wing `image2videoAWSIG` before answering.

Read rooms in this order when relevant:

1. `diagnostic-playbook`: how to inspect debug JSON, raw Comfy media, final media, and framing issues.
2. `parameters`: current env names, defaults, render variants, pan/output params.
3. `framing-cropping`: aspect ratio, square/9:16, original input, crop and pan behavior.
4. Relevant `case-*` rooms: specific prior bug diagnoses and known failure patterns.
5. Pipeline rooms as needed:
   - `pipeline-hunyuan`
   - `pipeline-svd`
   - `pipeline-animatediff`
6. `decision-prompt`: current decision preset-selection and motion/framing prompt rules.
7. `audio`: audio backend, device, prompt, and recreate-container notes.
8. `validation`: known verification commands/results and recent test artifacts.
9. `future-plans`: open threads and likely next improvements.
10. `git-state`: latest commit/state when history matters.

Keep MemPalace organized and current:

- Keep the room order above unless the project structure changes; if it changes, update this list.
- When a room's subject changes materially, update or add a focused drawer in that room.
- For subtle bugs, add a `case-*` drawer with symptom, exact debug/output paths, key fields,
  raw/final dimensions, root cause, fix direction, and validation artifact.
- When proposing or applying a fix found from a case drawer, explicitly classify it as a
  targeted workaround, safe general fix, or clean architectural fix. Mention if it is broader
  than ideal and what validation would prove it safe.
- Do not file old dead ends unless they explain current behavior or prevent repeating a mistake.
