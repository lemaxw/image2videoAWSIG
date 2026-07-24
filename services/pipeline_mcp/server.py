"""Local STDIO MCP server for deliberate image-to-video pipeline operations."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from services.pipeline_mcp import tools

mcp = FastMCP(
    "image2video-pipeline",
    instructions=(
        "Use these narrow local tools to inspect cases, render one explicitly "
        "configured clip, remux an existing raw render, run the production "
        "temporal quality measurement, or record a human review. Paths must "
        "remain inside the project workspace. No tool accepts shell commands."
    ),
    json_response=True,
)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Analyze pipeline case",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def analyze_case(artifact_path: str) -> dict[str, Any]:
    """Inspect a case directory or artifact, including debug decisions, attempts, reviews, and media probes."""

    return tools.analyze_case(artifact_path)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Render clip with explicit overrides",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def render_with_overrides(
    input_file: str,
    job_id: str,
    output_prefix: str = "out",
    preset: str | None = None,
    render_variants: str = "selected",
    output_aspect: str | None = None,
    final_crop_motion: str | None = None,
    pan_start: float | None = None,
    pan_end: float | None = None,
    pan_max_span: float | None = None,
    zoom_end: float | None = None,
    zoom_focus_x: float | None = None,
    zoom_focus_y: float | None = None,
    seed: int | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    use_original_input_for_video: bool | None = None,
    animation_directions: str = "",
    debug: bool = False,
    timeout_seconds: int = 10800,
) -> dict[str, Any]:
    """Render exactly one video_input image while overriding only named pipeline parameters."""

    return tools.render_with_overrides(
        input_file,
        job_id,
        output_prefix=output_prefix,
        preset=preset,
        render_variants=render_variants,
        output_aspect=output_aspect,
        final_crop_motion=final_crop_motion,
        pan_start=pan_start,
        pan_end=pan_end,
        pan_max_span=pan_max_span,
        zoom_end=zoom_end,
        zoom_focus_x=zoom_focus_x,
        zoom_focus_y=zoom_focus_y,
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_original_input_for_video=use_original_input_for_video,
        animation_directions=animation_directions,
        debug=debug,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Remux an existing raw render",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def remux_existing_raw(
    raw_video: str,
    audio_source: str,
    output_file: str,
    video_fit: str,
    output_aspect: str,
    pan_start: float = 0.0,
    pan_end: float = 1.0,
    zoom_end: float = 1.06,
    zoom_focus_x: float = 0.5,
    zoom_focus_y: float = 0.5,
    mix_db: float = -8.0,
    target_duration_s: float = 5.0,
    export_still: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a final presentation clip and optional JPG from existing raw video and an audio-bearing file."""

    return tools.remux_existing_raw(
        raw_video,
        audio_source,
        output_file,
        video_fit=video_fit,
        output_aspect=output_aspect,
        pan_start=pan_start,
        pan_end=pan_end,
        zoom_end=zoom_end,
        zoom_focus_x=zoom_focus_x,
        zoom_focus_y=zoom_focus_y,
        mix_db=mix_db,
        target_duration_s=target_duration_s,
        export_still=export_still,
        overwrite=overwrite,
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Check raw temporal quality",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def quality_check(
    source_image: str, raw_video: str, fps: int, expected_frames: int
) -> dict[str, Any]:
    """Run the same frame-to-source FFmpeg SSIM continuity measurement used by the production quality gate."""

    return tools.quality_check(
        source_image, raw_video, fps=fps, expected_frames=expected_frames
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Record candidate review",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def record_review(
    result_file: str,
    status: str,
    rating: int | None = None,
    issues: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Record accepted, rejected, or pending human feedback in a candidate .result.json."""

    return tools.record_review(
        result_file,
        status,
        rating=rating,
        issues=issues,
        notes=notes,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
