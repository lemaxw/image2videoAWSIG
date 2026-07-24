"""Container-side FFmpeg helpers for the pipeline MCP server."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from services.orchestrator.mux import mux_video_audio
from services.orchestrator.quality import assess_temporal_stability
from services.orchestrator.review import update_feedback


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _export_still(video_path: Path, image_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            "0.1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(image_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--media", type=Path, required=True)

    quality = subparsers.add_parser("quality")
    quality.add_argument("--source", type=Path, required=True)
    quality.add_argument("--video", type=Path, required=True)
    quality.add_argument("--fps", type=int, required=True)
    quality.add_argument("--expected-frames", type=int, required=True)

    remux = subparsers.add_parser("remux")
    remux.add_argument("--video", type=Path, required=True)
    remux.add_argument("--audio", type=Path, required=True)
    remux.add_argument("--output", type=Path, required=True)
    remux.add_argument("--video-fit", required=True)
    remux.add_argument("--output-aspect", required=True)
    remux.add_argument("--pan-start", type=float, required=True)
    remux.add_argument("--pan-end", type=float, required=True)
    remux.add_argument("--zoom-end", type=float, required=True)
    remux.add_argument("--zoom-focus-x", type=float, required=True)
    remux.add_argument("--zoom-focus-y", type=float, required=True)
    remux.add_argument("--mix-db", type=float, required=True)
    remux.add_argument("--target-duration-s", type=float, required=True)
    remux.add_argument("--export-still", action="store_true")

    review = subparsers.add_parser("review")
    review.add_argument("--result", type=Path, required=True)
    review.add_argument(
        "--status", choices=("accepted", "rejected", "pending"), required=True
    )
    review.add_argument("--rating", type=int)
    review.add_argument("--issues-json", default="[]")
    review.add_argument("--notes", default="")

    args = parser.parse_args()
    if args.operation == "probe":
        result = _probe(args.media)
    elif args.operation == "quality":
        result = assess_temporal_stability(
            args.video,
            args.source,
            fps=args.fps,
            expected_frames=args.expected_frames,
        )
    elif args.operation == "review":
        issues = json.loads(args.issues_json)
        if not isinstance(issues, list):
            raise ValueError("--issues-json must be a JSON array")
        updated = update_feedback(
            args.result,
            status=args.status,
            rating=args.rating,
            issues=issues,
            notes=args.notes,
        )
        result = {
            "state": updated.get("state"),
            "human_feedback": updated.get("human_feedback"),
        }
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = mux_video_audio(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output,
            mix_db=args.mix_db,
            video_fit=args.video_fit,
            pan_start=args.pan_start,
            pan_end=args.pan_end,
            output_aspect=args.output_aspect,
            target_duration_s=args.target_duration_s,
            zoom_end=args.zoom_end,
            zoom_focus_x=args.zoom_focus_x,
            zoom_focus_y=args.zoom_focus_y,
        )
        result["probe"] = _probe(args.output)
        if args.export_still:
            still_path = args.output.with_suffix(".jpg")
            _export_still(args.output, still_path)
            result["still"] = str(still_path)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
