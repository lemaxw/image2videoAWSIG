"""Lightweight raw-render quality checks used before audio and final muxing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Dict

_SSIM_ALL_RE = re.compile(r"\bAll:([0-9]+(?:\.[0-9]+)?)")
_MEAN_SIMILARITY_STEP_LIMIT = 0.025
_MAX_SIMILARITY_STEP_LIMIT = 0.10


def _summary_from_ssim_output(output: str) -> Dict[str, Any]:
    # FFmpeg's ssim filter prints one aggregate `All:<score>` value for every
    # compared frame. This score is each generated frame's similarity to the
    # original still, not a comparison with the preceding generated frame.
    scores = [float(match.group(1)) for match in _SSIM_ALL_RE.finditer(output)]
    if len(scores) < 3:
        return {
            "status": "unavailable",
            "reason": "fewer than three SSIM samples",
            "sample_count": len(scores),
        }

    # Absolute similarity may fall gradually during legitimate animation, so it
    # is not used as a pass/fail threshold. Instead, trace how much the
    # frame-to-source score changes from one generated frame to the next. A
    # sudden geometry warp or visual jump appears as a sharp step in this
    # sequence; unstable stretches also raise the mean step.
    steps = [abs(current - previous) for previous, current in zip(scores, scores[1:])]
    return {
        "status": "measured",
        "sample_count": len(scores),
        "source_similarity_mean": round(mean(scores), 6),
        "source_similarity_min": round(min(scores), 6),
        "source_similarity_max": round(max(scores), 6),
        "mean_similarity_step": round(mean(steps), 6),
        "max_similarity_step": round(max(steps), 6),
    }


def needs_temporal_stability_gate(
    analysis: Dict[str, Any], video_cfg: Dict[str, Any]
) -> bool:
    preset = str(video_cfg.get("preset", ""))
    if preset != "WAN22_NATURAL":
        return False
    dynamic = (
        analysis.get("dynamic_potential")
        if isinstance(analysis.get("dynamic_potential"), dict)
        else {}
    )
    complexity = (
        analysis.get("content_complexity")
        if isinstance(analysis.get("content_complexity"), dict)
        else {}
    )
    params = (
        video_cfg.get("params") if isinstance(video_cfg.get("params"), dict) else {}
    )
    fragile_detail = any(
        bool(complexity.get(key))
        for key in ("dense_details", "fine_geometry", "repeating_patterns")
    )

    # Large SSIM steps can be valid in a genuinely dynamic scene. Restrict this
    # guard to image2json's low-motion, fragile-detail class where the raw video
    # should stay close to the source and the final FFmpeg push/pull supplies
    # most of the intended camera motion.
    return (
        str(dynamic.get("level", "")).strip().lower() == "low"
        and fragile_detail
        and str(params.get("final_crop_motion", "")).strip().lower()
        in {"push_in", "pull_out"}
    )


def assess_temporal_stability(
    video_path: Path,
    source_image: Path,
    *,
    fps: int,
    expected_frames: int,
) -> Dict[str, Any]:
    """Measure frame-to-source continuity and reject abrupt low-motion renders."""

    # Equivalent diagnostic command:
    #
    #   ffmpeg -i RAW.mp4 -loop 1 -framerate FPS -i SOURCE.jpg \
    #     -filter_complex \
    #     "[1:v][0:v]scale2ref[ref][vid];[vid][ref]ssim=stats_file=/dev/stdout" \
    #     -frames:v EXPECTED_FRAMES -f null -
    #
    # Stream 0 is the raw generated clip. Stream 1 loops the original image for
    # the same number of frames. `scale2ref` resizes only the still to the raw
    # video's native dimensions, then `ssim` emits one `All:` score per frame.
    # Writing the trace to stdout lets us retain the values without a temporary
    # stats file. This runs before presentation zoom/pan, audio, or final mux,
    # which is how it distinguishes a renderer jump from a post-process issue.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-loop",
        "1",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        str(source_image),
        "-filter_complex",
        "[1:v][0:v]scale2ref[ref][vid];[vid][ref]ssim=stats_file=/dev/stdout",
        "-frames:v",
        str(max(3, int(expected_frames))),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Quality diagnostics must not turn a missing FFmpeg feature or a
        # transient analysis timeout into loss of an otherwise usable render.
        # Fail open, but return the reason so debug.json preserves the gap.
        return {
            "status": "unavailable",
            "reason": str(exc),
            "error_type": exc.__class__.__name__,
            "passed": True,
        }

    result = _summary_from_ssim_output(completed.stdout)
    if result["status"] != "measured":
        result["passed"] = True
        return result

    # Require both conditions: the mean limit rejects a sustained unstable
    # stretch, while the maximum limit confirms at least one clearly abrupt
    # discontinuity. Using AND avoids rejecting a stable clip for one harmless
    # local-motion spike, and avoids rejecting smooth continuous motion merely
    # because its average similarity drifts.
    result["thresholds"] = {
        "mean_similarity_step": _MEAN_SIMILARITY_STEP_LIMIT,
        "max_similarity_step": _MAX_SIMILARITY_STEP_LIMIT,
    }
    result["passed"] = not (
        result["mean_similarity_step"] > result["thresholds"]["mean_similarity_step"]
        and result["max_similarity_step"] > result["thresholds"]["max_similarity_step"]
    )
    if not result["passed"]:
        result["reason"] = (
            "abrupt raw-frame continuity changes in a low-motion, detail-dense scene"
        )
    return result
