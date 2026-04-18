# Agent Notes

Read [README.md](/home/mpshater/hobby/image2videoAWSIG/README.md) first.

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
  - `fps=7, frames=35, resolution_width=768, steps=24`
  - `seed` is auto-generated unless explicitly provided.
- `debug.json` is always saved.
- Intermediate artifacts are only preserved with `--debug`.
- If `.env` audio backend/device values change, recreate `audio` container (not restart).
