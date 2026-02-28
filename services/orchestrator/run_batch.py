#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from services.decision.decision_service import decide_for_image
from services.orchestrator.comfy_client import ComfyClient, find_latest_mp4
from services.orchestrator.mux import mux_video_audio
from services.orchestrator.s3_io import S3IO
from services.orchestrator.validate import validate_and_clamp_decision


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "time": int(time.time()),
        }
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=True)


logger = logging.getLogger("run_batch")
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _audio_generate(audio_url: str, prompt: str, duration_s: int, output_dir: Path) -> Path:
    payload = {
        "prompt": prompt,
        "duration_s": duration_s,
        "output_dir": str(output_dir),
    }
    resp = requests.post(f"{audio_url}/generate", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return Path(data["wav_path"])


def _build_workflow(comfy: ComfyClient, templates_root: Path, local_input: Path, output_prefix: str, video_cfg: Dict[str, Any]) -> Dict[str, Any]:
    preset = video_cfg["preset"]
    if preset.startswith("SVD") or preset == "FAILSAFE_LOW_MEM":
        return comfy.build_svd_workflow(
            templates_root / "svd_workflow.json",
            input_image=str(local_input),
            output_prefix=output_prefix,
            decision_video=video_cfg,
        )
    return comfy.build_animatediff_workflow(
        templates_root / "animatediff_workflow.json",
        output_prefix=output_prefix,
        decision_video=video_cfg,
    )


def process_one_image(
    s3: S3IO,
    comfy: ComfyClient,
    audio_url: str,
    input_bucket: str,
    input_key: str,
    output_bucket: str,
    output_prefix: str,
    job_id: str,
    work_root: Path,
) -> bool:
    start_t = time.time()
    image_name = Path(input_key).stem
    local_input = work_root / "inputs" / Path(input_key).name
    local_case_dir = work_root / "cases" / image_name
    local_case_dir.mkdir(parents=True, exist_ok=True)

    debug: Dict[str, Any] = {
        "input_key": input_key,
        "job_id": job_id,
        "attempts": [],
        "timings": {},
        "status": "failed",
        "error": None,
    }

    try:
        s3.download_file(input_bucket, input_key, local_input)
        t_decision_start = time.time()
        decision = decide_for_image(local_input, metadata={"job_id": job_id, "input_key": input_key})
        decision = validate_and_clamp_decision(decision)
        debug["decision"] = decision
        debug["timings"]["decision_s"] = round(time.time() - t_decision_start, 3)

        video_candidates: List[Dict[str, Any]] = [decision["video"], *decision["fallbacks"]]
        final_mux: Path | None = None
        workflow_used: Dict[str, Any] | None = None

        for idx, video_cfg in enumerate(video_candidates):
            attempt = {"index": idx, "video": video_cfg, "status": "started"}
            t_render_start = time.time()
            try:
                workflow = _build_workflow(
                    comfy,
                    templates_root=Path("/app/services/comfy/workflow_templates"),
                    local_input=local_input,
                    output_prefix=f"{job_id}-{image_name}-{idx}",
                    video_cfg=video_cfg,
                )
                prompt_id = comfy.submit_workflow(workflow)
                hist = comfy.wait_for_prompt(prompt_id)
                video_path = find_latest_mp4(hist, Path(os.environ.get("OUTPUT_DIR", "/data/outputs")))

                audio_cfg = decision["audio"]
                audio_path = _audio_generate(
                    audio_url=audio_url,
                    prompt=audio_cfg["prompt"],
                    duration_s=audio_cfg["duration_s"],
                    output_dir=local_case_dir,
                )

                final_mux = local_case_dir / "final.mp4"
                mux_video_audio(video_path=video_path, audio_path=audio_path, output_path=final_mux, mix_db=audio_cfg["mix_db"])

                workflow_used = workflow
                attempt["status"] = "success"
                attempt["prompt_id"] = prompt_id
                attempt["video_path"] = str(video_path)
                attempt["audio_path"] = str(audio_path)
                attempt["render_s"] = round(time.time() - t_render_start, 3)
                debug["attempts"].append(attempt)
                break
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["error"] = str(exc)
                attempt["render_s"] = round(time.time() - t_render_start, 3)
                debug["attempts"].append(attempt)

        if final_mux is None or workflow_used is None:
            raise RuntimeError("All render attempts failed")

        debug["workflow_used"] = workflow_used
        debug["status"] = "success"
        debug["timings"]["total_s"] = round(time.time() - start_t, 3)

        final_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/final.mp4"
        debug_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/debug.json"

        s3.upload_file(final_mux, output_bucket, final_key, content_type="video/mp4")
        debug_path = local_case_dir / "debug.json"
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        s3.upload_file(debug_path, output_bucket, debug_key, content_type="application/json")

        logger.info("processed", extra={"extra": {"image": input_key, "status": "success", "output_key": final_key}})
        return True
    except Exception as exc:
        debug["error"] = str(exc)
        debug["timings"]["total_s"] = round(time.time() - start_t, 3)

        debug_path = local_case_dir / "debug.json"
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        debug_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/debug.json"
        s3.upload_file(debug_path, output_bucket, debug_key, content_type="application/json")
        logger.error("processed", extra={"extra": {"image": input_key, "status": "failed", "error": str(exc)}})
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run image->video batch orchestrator")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input-bucket", default=os.environ.get("DEFAULT_INPUT_BUCKET", ""))
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-bucket", default=os.environ.get("DEFAULT_OUTPUT_BUCKET", ""))
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-fail-ratio", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-input-dir", default="")
    parser.add_argument("--local-output-dir", default="")
    args = parser.parse_args()

    if not args.dry_run and (not args.input_bucket or not args.output_bucket):
        raise ValueError("input/output bucket must be provided unless --dry-run is set")

    work_root = Path(os.environ.get("WORK_ROOT", "/tmp/orchestrator")) / args.job_id
    work_root.mkdir(parents=True, exist_ok=True)

    s3 = S3IO(dry_run=args.dry_run, local_input_dir=args.local_input_dir, local_output_dir=args.local_output_dir)
    comfy = ComfyClient(os.environ.get("COMFY_URL", "http://localhost:8188"))
    audio_url = os.environ.get("AUDIO_URL", "http://localhost:8000")

    image_keys = s3.list_images(args.input_bucket, args.input_prefix)
    if not image_keys:
        logger.warning("no images found", extra={"extra": {"prefix": args.input_prefix}})
        return 0

    total = len(image_keys)
    failures = 0
    for key in image_keys:
        ok = process_one_image(
            s3=s3,
            comfy=comfy,
            audio_url=audio_url,
            input_bucket=args.input_bucket,
            input_key=key,
            output_bucket=args.output_bucket,
            output_prefix=args.output_prefix,
            job_id=args.job_id,
            work_root=work_root,
        )
        failures += 0 if ok else 1

    fail_ratio = failures / total
    logger.info("batch done", extra={"extra": {"total": total, "failures": failures, "fail_ratio": fail_ratio}})
    return 1 if fail_ratio > args.max_fail_ratio else 0


if __name__ == "__main__":
    sys.exit(main())
