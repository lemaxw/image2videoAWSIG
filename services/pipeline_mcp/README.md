# Pipeline MCP server

This local STDIO MCP server gives Codex five explicit pipeline tools:

- `analyze_case`: read debug/result metadata and probe produced MP4s.
- `render_with_overrides`: process one `video_input` image with named renderer,
  presentation, seed, and prompt overrides.
- `remux_existing_raw`: make a targeted final clip from an existing raw MP4 and
  audio-bearing artifact.
- `quality_check`: run the production FFmpeg SSIM continuity measurement.
- `record_review`: write explicit human feedback to one candidate result record.

It intentionally has no generic command or shell-execution tool. All paths must
resolve inside the repository. Render input is restricted to `video_input`, and
new mux output is restricted to `video_output`.

The launcher uses the stable 1.x MCP Python SDK:

```bash
scripts/run_pipeline_mcp.sh
```

Register it with Codex:

```bash
codex mcp add image2video-pipeline -- \
  /home/lemaxw/hobby/image2videoAWSIG/scripts/run_pipeline_mcp.sh
```

Then restart Codex and inspect the connection with `/mcp` or:

```bash
codex mcp get image2video-pipeline
```

The FFmpeg-backed tools and rendering expect `pipeline-orchestrator` to be
running. Start the local stack using the commands in the root `README.md`.
