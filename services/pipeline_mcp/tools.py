"""Narrow, auditable operations exposed by the local pipeline MCP server."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from services.orchestrator.quality import needs_temporal_stability_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "video_input"
OUTPUT_ROOT = REPO_ROOT / "video_output"
ORCHESTRATOR_CONTAINER = "pipeline-orchestrator"
CONTAINER_OPS = "/app/services/pipeline_mcp/container_ops.py"
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_PRESETS = {
    "WAN22_NATURAL",
    "HUNYUAN15_I2V_720P",
    "HUNYUAN15_I2V_FAST",
    "DETERMINISTIC_ORIGINAL",
}
_RENDER_VARIANTS = {"selected", "selected_pair", "all", "wan", "hunyuan"}
_ASPECTS = {"instagram_reel_9_16", "square_1_1"}
_MOTIONS = {"static", "pan_left_to_right", "pan_right_to_left", "push_in", "pull_out"}
_VIDEO_FITS = {"contain", "cover", "static_crop", *_MOTIONS - {"static"}}


def _workspace_path(value: str | Path, *, must_exist: bool = True) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else REPO_ROOT / raw
    candidate = candidate.resolve(strict=must_exist)
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path must stay inside the project workspace: {value}"
        ) from exc
    return candidate


def _output_path(value: str | Path) -> Path:
    path = _workspace_path(value, must_exist=False)
    try:
        path.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output path must be inside {OUTPUT_ROOT}") from exc
    return path


def _container_path(path: Path) -> str:
    return f"/app/{path.resolve().relative_to(REPO_ROOT).as_posix()}"


def _run(
    command: list[str], *, timeout: int, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Command failed to start or timed out: {exc}") from exc
    if check and completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(detail[-4000:])
    return completed


def _container_running() -> bool:
    completed = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", ORCHESTRATOR_CONTAINER],
        timeout=20,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _require_container() -> None:
    if not _container_running():
        raise RuntimeError(
            f"{ORCHESTRATOR_CONTAINER} is not running. Start the local stack with "
            "docker compose --env-file .env -f services/comfy/docker-compose.yml "
            "-f services/comfy/docker-compose.gpu.yml up -d"
        )


def _container_json(args: list[str], *, timeout: int = 180) -> dict[str, Any]:
    _require_container()
    completed = _run(
        [
            "docker",
            "exec",
            "-i",
            ORCHESTRATOR_CONTAINER,
            "python",
            CONTAINER_OPS,
            *args,
        ],
        timeout=timeout,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Container helper returned invalid JSON: {completed.stdout[-2000:]}"
        ) from exc


def _json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _probe(path: Path) -> dict[str, Any]:
    return _container_json(["probe", "--media", _container_path(path)], timeout=60)


def _attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: attempt.get(key)
        for key in (
            "index",
            "status",
            "error",
            "error_type",
            "variant",
            "video_path",
            "video_fit",
            "pan_start",
            "pan_end",
            "zoom_end",
            "zoom_focus_x",
            "zoom_focus_y",
            "temporal_quality",
            "video",
        )
        if attempt.get(key) is not None
    }


def analyze_case(artifact_path: str) -> dict[str, Any]:
    """Inspect one case directory or artifact without changing pipeline state."""

    artifact = _workspace_path(artifact_path)
    case_dir = artifact if artifact.is_dir() else artifact.parent
    debug_files = sorted(
        case_dir.glob("debug_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    result_files = sorted(
        case_dir.glob("*.result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    media_files = sorted(
        case_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True
    )

    response: dict[str, Any] = {
        "case_dir": str(case_dir.relative_to(REPO_ROOT)),
        "debug_records": [],
        "candidate_records": [],
        "media": [],
    }
    for path in debug_files[:10]:
        payload = _json_file(path)
        image2json = (
            payload.get("image2json")
            if isinstance(payload.get("image2json"), dict)
            else {}
        )
        response["debug_records"].append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "input_key": payload.get("input_key"),
                "job_id": payload.get("job_id"),
                "status": payload.get("status"),
                "error": payload.get("error"),
                "decision": payload.get("decision"),
                "analysis": image2json.get("analysis"),
                "semantic_plan": payload.get("semantic_plan"),
                "attempts": [
                    _attempt_summary(item)
                    for item in payload.get("attempts", [])
                    if isinstance(item, dict)
                ],
                "final_outputs": payload.get("final_outputs"),
            }
        )
    for path in result_files[:20]:
        payload = _json_file(path)
        response["candidate_records"].append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "state": payload.get("state"),
                "candidate_id": payload.get("candidate_id"),
                "source": payload.get("source"),
                "technical_plan": payload.get("technical_plan"),
                "generation": _attempt_summary(payload.get("generation") or {}),
                "presentation": payload.get("presentation"),
                "artifacts": payload.get("artifacts"),
                "human_feedback": payload.get("human_feedback"),
            }
        )
    for path in media_files[:20]:
        response["media"].append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "probe": _probe(path),
            }
        )
    return response


def quality_check(
    source_image: str, raw_video: str, *, fps: int, expected_frames: int
) -> dict[str, Any]:
    """Run the production SSIM continuity measurement without altering media."""

    source = _workspace_path(source_image)
    raw = _workspace_path(raw_video)
    if source.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("source_image must be a supported image")
    if raw.suffix.lower() != ".mp4":
        raise ValueError("raw_video must be an MP4")
    if not 1 <= int(fps) <= 120:
        raise ValueError("fps must be between 1 and 120")
    if not 3 <= int(expected_frames) <= 10000:
        raise ValueError("expected_frames must be between 3 and 10000")
    result = _container_json(
        [
            "quality",
            "--source",
            _container_path(source),
            "--video",
            _container_path(raw),
            "--fps",
            str(int(fps)),
            "--expected-frames",
            str(int(expected_frames)),
        ],
        timeout=180,
    )
    return {
        "source_image": str(source.relative_to(REPO_ROOT)),
        "raw_video": str(raw.relative_to(REPO_ROOT)),
        **result,
    }


def render_with_overrides(
    input_file: str,
    job_id: str,
    *,
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
    """Render one input with explicit validated overrides through the orchestrator."""

    source = _workspace_path(INPUT_ROOT / input_file)
    try:
        relative_input = source.relative_to(INPUT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("input_file must resolve below video_input") from exc
    if source.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("input_file must be a supported image")
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError(
            "job_id must contain only letters, numbers, dot, underscore, or dash"
        )
    if (
        not _SAFE_PREFIX_RE.fullmatch(output_prefix)
        or ".." in Path(output_prefix).parts
    ):
        raise ValueError("output_prefix must be a safe relative path")
    if preset is not None and preset not in _PRESETS:
        raise ValueError(f"Unsupported preset: {preset}")
    if render_variants not in _RENDER_VARIANTS:
        raise ValueError(f"Unsupported render_variants: {render_variants}")
    if output_aspect is not None and output_aspect not in _ASPECTS:
        raise ValueError(f"Unsupported output_aspect: {output_aspect}")
    if final_crop_motion is not None and final_crop_motion not in _MOTIONS:
        raise ValueError(f"Unsupported final_crop_motion: {final_crop_motion}")
    if not 60 <= int(timeout_seconds) <= 21600:
        raise ValueError("timeout_seconds must be between 60 and 21600")

    overrides: dict[str, Any] = {"render_variants": render_variants}
    optional = {
        "preset": preset,
        "output_aspect": output_aspect,
        "target_aspect": output_aspect,
        "final_crop_motion": final_crop_motion,
        "pan_start": pan_start,
        "pan_end": pan_end,
        "pan_max_span": pan_max_span,
        "zoom_end": zoom_end,
        "zoom_focus_x": zoom_focus_x,
        "zoom_focus_y": zoom_focus_y,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "use_original_input_for_video": use_original_input_for_video,
    }
    overrides.update(
        {key: value for key, value in optional.items() if value is not None}
    )

    _require_container()
    case_dir = OUTPUT_ROOT / output_prefix / job_id / source.stem
    before = (
        {path.resolve() for path in case_dir.glob("*")} if case_dir.exists() else set()
    )
    command = [
        "docker",
        "exec",
        "-i",
        ORCHESTRATOR_CONTAINER,
        "python",
        "/app/services/orchestrator/run_batch.py",
        "--job-id",
        job_id,
        "--input-file",
        relative_input,
        "--output-prefix",
        output_prefix,
        "--local-input-dir",
        "/data/local_inputs",
        "--local-output-dir",
        "/data/local_outputs",
        "--video-params-json",
        json.dumps(overrides, separators=(",", ":")),
    ]
    if animation_directions.strip():
        command.extend(["--animation-directions", animation_directions.strip()])
    if debug:
        command.append("--debug")
    completed = _run(command, timeout=int(timeout_seconds), check=False)
    after = (
        {path.resolve() for path in case_dir.glob("*")} if case_dir.exists() else set()
    )
    created = sorted(str(path.relative_to(REPO_ROOT)) for path in after - before)
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "input_file": relative_input,
        "job_id": job_id,
        "overrides": overrides,
        "created_artifacts": created,
        "log_tail": (completed.stdout + "\n" + completed.stderr).splitlines()[-100:],
    }


def remux_existing_raw(
    raw_video: str,
    audio_source: str,
    output_file: str,
    *,
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
    """Create a presentation variant from existing raw video and audio."""

    raw = _workspace_path(raw_video)
    audio = _workspace_path(audio_source)
    output = _output_path(output_file)
    if raw.suffix.lower() != ".mp4":
        raise ValueError("raw_video must be an MP4")
    if video_fit not in _VIDEO_FITS:
        raise ValueError(f"Unsupported video_fit: {video_fit}")
    if output_aspect not in _ASPECTS:
        raise ValueError(f"Unsupported output_aspect: {output_aspect}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_file must end in .mp4")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    args = [
        "remux",
        "--video",
        _container_path(raw),
        "--audio",
        _container_path(audio),
        "--output",
        _container_path(output),
        "--video-fit",
        video_fit,
        "--output-aspect",
        output_aspect,
        "--pan-start",
        str(float(pan_start)),
        "--pan-end",
        str(float(pan_end)),
        "--zoom-end",
        str(float(zoom_end)),
        "--zoom-focus-x",
        str(float(zoom_focus_x)),
        "--zoom-focus-y",
        str(float(zoom_focus_y)),
        "--mix-db",
        str(float(mix_db)),
        "--target-duration-s",
        str(float(target_duration_s)),
    ]
    if export_still:
        args.append("--export-still")
    result = _container_json(args, timeout=900)
    return {
        "raw_video": str(raw.relative_to(REPO_ROOT)),
        "audio_source": str(audio.relative_to(REPO_ROOT)),
        "output_file": str(output.relative_to(REPO_ROOT)),
        **result,
    }


def record_review(
    result_file: str,
    status: str,
    *,
    rating: int | None = None,
    issues: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Persist explicit human feedback in one candidate result record."""

    result_path = _workspace_path(result_file)
    try:
        result_path.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("result_file must be inside video_output") from exc
    if not result_path.name.endswith(".result.json"):
        raise ValueError("result_file must end in .result.json")
    normalized_status = status.strip().lower()
    if normalized_status not in {"accepted", "rejected", "pending"}:
        raise ValueError("status must be accepted, rejected, or pending")
    args = [
        "review",
        "--result",
        _container_path(result_path),
        "--status",
        normalized_status,
        "--issues-json",
        json.dumps(issues or []),
        "--notes",
        notes,
    ]
    if rating is not None:
        args.extend(["--rating", str(int(rating))])
    updated = _container_json(args, timeout=60)
    return {
        "result_file": str(result_path.relative_to(REPO_ROOT)),
        **updated,
    }


def gate_eligibility(analysis: dict[str, Any], video_config: dict[str, Any]) -> bool:
    """Expose the exact production eligibility predicate for tests and audits."""

    return needs_temporal_stability_gate(analysis, video_config)
