#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image

# Allow `python services/orchestrator/run_batch.py ...` from repo root.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from services.decision.decision_service import decide_for_image_detailed
from services.orchestrator.comfy_client import ComfyClient, find_latest_mp4
from services.orchestrator.mux import mux_video_audio
from services.orchestrator.validate import validate_and_clamp_decision


DEFAULT_VIDEO_OVERRIDES: Dict[str, Any] = {
    "fps": 5,
    "frames": 25,
    "resolution_width": 768,
    "steps": 24,
    "motion_bucket_id": 24,
    "seed": 123,
}


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


def _log(level: str, msg: str, **fields: Any) -> None:
    fn = getattr(logger, level)
    fn(msg, extra={"extra": fields})


class LocalIO:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_images(self, prefix: str) -> List[str]:
        root = self.input_dir / prefix if prefix and prefix != "." else self.input_dir
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(self.input_dir))
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    def copy_input(self, rel_key: str, local_path: Path) -> None:
        src = self.input_dir / rel_key
        if not src.exists():
            raise FileNotFoundError(f"input not found: {src}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def write_output(self, local_path: Path, key: str) -> None:
        dst = self.output_dir / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)


def _audio_generate(audio_url: str, prompt: str, duration_s: int, output_dir: Path) -> Dict[str, str]:
    payload = {
        "prompt": prompt,
        "duration_s": duration_s,
        "output_dir": str(output_dir),
    }
    resp = requests.post(f"{audio_url}/generate", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return {"wav_path": data["wav_path"], "backend": data.get("backend", "unknown")}


def _wait_http_ok(url: str, timeout_s: int, label: str) -> None:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 400:
                _log("info", "service.ready", service=label, url=url, status_code=resp.status_code)
                return
            last_err = f"http_{resp.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(2)
    raise RuntimeError(f"{label} not ready within {timeout_s}s: {last_err}")


def _resolve_comfy_url(initial_url: str) -> str:
    candidates = [initial_url]
    if "127.0.0.1:8188" in initial_url:
        candidates.append(initial_url.replace("127.0.0.1:8188", "127.0.0.1:18188"))
    if "localhost:8188" in initial_url:
        candidates.append(initial_url.replace("localhost:8188", "localhost:18188"))
    if "127.0.0.1:18188" in initial_url:
        candidates.append(initial_url.replace("127.0.0.1:18188", "127.0.0.1:8188"))
    if "localhost:18188" in initial_url:
        candidates.append(initial_url.replace("localhost:18188", "localhost:8188"))

    last_err = None
    for url in candidates:
        try:
            _wait_http_ok(f"{url}/system_stats", timeout_s=35, label="comfy")
            return url
        except Exception as exc:
            last_err = exc
            _log("warning", "service.probe.failed", service="comfy", url=url, error=str(exc))
    raise RuntimeError(f"comfy endpoint resolution failed: {last_err}")


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


def _resolve_templates_root() -> Path:
    env_path = os.environ.get("WORKFLOW_TEMPLATES_DIR", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    in_container = Path("/app/services/comfy/workflow_templates")
    if in_container.exists():
        return in_container

    local_repo = Path(__file__).resolve().parents[1] / "comfy" / "workflow_templates"
    if local_repo.exists():
        return local_repo

    raise FileNotFoundError("Workflow templates directory not found")


def _apply_video_overrides(decision: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(decision))
    target_keys = {"preset", "duration_s", "fps", "frames", "resolution_width", "seed"}
    framing_keys = {"crop_anchor", "target_aspect"}

    def apply_to_video(video_cfg: Dict[str, Any]) -> None:
        params = video_cfg.get("params") or {}
        for key, value in overrides.items():
            if key in target_keys:
                video_cfg[key] = value
                if key != "preset":
                    params[key] = value
            else:
                params[key] = value
        video_cfg["params"] = params

    apply_to_video(out["video"])
    for fb in out.get("fallbacks", []):
        apply_to_video(fb)
    framing = out.get("framing") or {}
    if isinstance(framing, dict):
        for key in framing_keys:
            if key in overrides:
                framing[key] = overrides[key]
        out["framing"] = framing
    return out


def _propagate_target_aspect(decision: Dict[str, Any]) -> None:
    target_aspect = str((decision.get("framing") or {}).get("target_aspect", ""))
    if not target_aspect:
        return
    for cfg in [decision.get("video"), *(decision.get("fallbacks") or [])]:
        if isinstance(cfg, dict):
            params = cfg.get("params") or {}
            params["target_aspect"] = target_aspect
            cfg["params"] = params


def _crop_anchor_offsets(anchor: str) -> tuple[float, float]:
    mapping = {
        "left_top": (0.0, 0.0),
        "center_top": (0.5, 0.0),
        "right_top": (1.0, 0.0),
        "left_center": (0.0, 0.5),
        "center_center": (0.5, 0.5),
        "right_center": (1.0, 0.5),
        "left_bottom": (0.0, 1.0),
        "center_bottom": (0.5, 1.0),
        "right_bottom": (1.0, 1.0),
    }
    return mapping.get(anchor, (0.5, 0.5))


def _prepare_instagram_input_image(source_path: Path, out_dir: Path, framing: Dict[str, Any]) -> Dict[str, Any]:
    target_aspect = str(framing.get("target_aspect", "instagram_reel_9_16"))
    crop_anchor = str(framing.get("crop_anchor", "center_center"))

    out_dir.mkdir(parents=True, exist_ok=True)
    staged_path = out_dir / f"{source_path.stem}.ig{source_path.suffix.lower() or '.jpg'}"

    with Image.open(source_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        target_ratio = 9.0 / 16.0
        current_ratio = width / height if height > 0 else target_ratio

        if current_ratio > target_ratio:
            crop_h = height
            crop_w = int(round(crop_h * target_ratio))
        else:
            crop_w = width
            crop_h = int(round(crop_w / target_ratio))

        crop_w = max(1, min(width, crop_w))
        crop_h = max(1, min(height, crop_h))

        x_bias, y_bias = _crop_anchor_offsets(crop_anchor)
        left = int(round((width - crop_w) * x_bias))
        top = int(round((height - crop_h) * y_bias))
        left = max(0, min(left, width - crop_w))
        top = max(0, min(top, height - crop_h))
        right = left + crop_w
        bottom = top + crop_h

        cropped = img.crop((left, top, right, bottom))
        cropped.save(staged_path, quality=95)

    return {
        "path": str(staged_path),
        "target_aspect": target_aspect,
        "crop_anchor": crop_anchor,
        "source_size": {"width": width, "height": height},
        "crop_box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "output_size": {"width": crop_w, "height": crop_h},
    }


def _cleanup_intermediates(local_case_dir: Path, render_input: Path, source_input: Path, attempt: Dict[str, Any] | None) -> None:
    if attempt:
        for k in ("video_path", "audio_path"):
            p = attempt.get(k)
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
    if render_input != source_input:
        try:
            render_input.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        shutil.rmtree(local_case_dir, ignore_errors=True)
    except Exception:
        pass


def process_one_image(
    io: LocalIO,
    comfy: ComfyClient,
    audio_url: str,
    templates_root: Path,
    input_key: str,
    output_prefix: str,
    job_id: str,
    work_root: Path,
    video_overrides: Dict[str, Any],
    debug_enabled: bool,
) -> bool:
    start_t = time.time()
    image_name = Path(input_key).stem
    input_root = Path(os.environ.get("INPUT_DIR", str(work_root / "inputs")))
    input_root.mkdir(parents=True, exist_ok=True)
    local_input = input_root / Path(input_key).name
    local_case_dir = work_root / "cases" / image_name
    local_case_dir.mkdir(parents=True, exist_ok=True)

    debug: Dict[str, Any] = {"input_key": input_key, "job_id": job_id, "attempts": [], "timings": {}, "status": "failed", "error": None}
    last_attempt: Dict[str, Any] | None = None
    render_input = local_input

    try:
        _log("info", "image.start", job_id=job_id, input_key=input_key)

        t_download = time.time()
        io.copy_input(input_key, local_input)
        debug["timings"]["download_s"] = round(time.time() - t_download, 3)
        _log("info", "image.download.done", job_id=job_id, input_key=input_key, duration_s=debug["timings"]["download_s"])

        t_decision_start = time.time()
        decision_result = decide_for_image_detailed(local_input, metadata={"job_id": job_id, "input_key": input_key})
        decision = decision_result["decision"]
        openai_meta = decision_result.get("openai", {})
        decision = _apply_video_overrides(decision, video_overrides)
        decision = validate_and_clamp_decision(decision)
        _propagate_target_aspect(decision)
        crop_info = _prepare_instagram_input_image(local_input, input_root, decision.get("framing", {}))
        render_input = Path(crop_info["path"])
        debug["decision"] = decision
        debug["framing"] = crop_info
        debug["openai"] = openai_meta
        debug["timings"]["decision_s"] = round(time.time() - t_decision_start, 3)
        _log(
            "info",
            "image.decision.done",
            job_id=job_id,
            input_key=input_key,
            duration_s=debug["timings"]["decision_s"],
            openai_model=openai_meta.get("model"),
            openai_status=openai_meta.get("status", "unknown"),
            openai_attempts=openai_meta.get("attempts", 0),
            openai_input_tokens=openai_meta.get("usage", {}).get("input_tokens", 0),
            openai_output_tokens=openai_meta.get("usage", {}).get("output_tokens", 0),
            openai_total_tokens=openai_meta.get("usage", {}).get("total_tokens", 0),
            crop_anchor=crop_info.get("crop_anchor"),
            cropped_input=render_input.as_posix(),
        )

        video_candidates: List[Dict[str, Any]] = [decision["video"], *decision["fallbacks"]]
        final_mux: Path | None = None
        workflow_used: Dict[str, Any] | None = None

        for idx, video_cfg in enumerate(video_candidates):
            attempt = {"index": idx, "video": video_cfg, "status": "started"}
            last_attempt = attempt
            t_render_start = time.time()
            _log("info", "image.render.attempt.start", job_id=job_id, input_key=input_key, attempt_index=idx, preset=video_cfg.get("preset"))
            try:
                t_workflow_build = time.time()
                workflow = _build_workflow(
                    comfy,
                    templates_root=templates_root,
                    local_input=render_input,
                    output_prefix=f"{job_id}-{image_name}-{idx}",
                    video_cfg=video_cfg,
                )
                attempt["workflow_build_s"] = round(time.time() - t_workflow_build, 3)
                t_submit = time.time()
                prompt_id = comfy.submit_workflow(workflow)
                attempt["submit_s"] = round(time.time() - t_submit, 3)
                t_wait = time.time()
                hist = comfy.wait_for_prompt(prompt_id)
                attempt["comfy_wait_s"] = round(time.time() - t_wait, 3)
                video_path = find_latest_mp4(hist, Path(os.environ.get("OUTPUT_DIR", "/data/outputs")))

                audio_cfg = decision["audio"]
                shared_output_root = Path(os.environ.get("OUTPUT_DIR", "/data/outputs"))
                shared_audio_dir = shared_output_root / "audio" / job_id / image_name
                t_audio = time.time()
                _log("info", "image.audio.start", job_id=job_id, input_key=input_key, attempt_index=idx, duration_s=audio_cfg["duration_s"], output_dir=str(shared_audio_dir))
                audio_info = _audio_generate(audio_url=audio_url, prompt=audio_cfg["prompt"], duration_s=audio_cfg["duration_s"], output_dir=shared_audio_dir)
                attempt["audio_generate_s"] = round(time.time() - t_audio, 3)
                attempt["audio_backend"] = audio_info.get("backend", "unknown")
                audio_path = Path(audio_info["wav_path"])
                _log("info", "image.audio.done", job_id=job_id, input_key=input_key, attempt_index=idx, backend=attempt["audio_backend"], wav_path=str(audio_path), duration_s=attempt["audio_generate_s"])

                final_mux = local_case_dir / "final.mp4"
                t_mux = time.time()
                mux_video_audio(video_path=video_path, audio_path=audio_path, output_path=final_mux, mix_db=audio_cfg["mix_db"])
                attempt["mux_s"] = round(time.time() - t_mux, 3)
                attempt["prompt_id"] = prompt_id
                attempt["video_path"] = str(video_path)
                attempt["audio_path"] = str(audio_path)
                attempt["render_s"] = round(time.time() - t_render_start, 3)
                attempt["status"] = "success"
                debug["attempts"].append(attempt)
                workflow_used = workflow
                _log("info", "image.render.attempt.done", job_id=job_id, input_key=input_key, attempt_index=idx, status="success", duration_s=attempt["render_s"], preset=video_cfg.get("preset"), prompt_id=prompt_id)
                break
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["error"] = str(exc)
                attempt["error_type"] = exc.__class__.__name__
                attempt["render_s"] = round(time.time() - t_render_start, 3)
                debug["attempts"].append(attempt)
                _log("error", "image.render.attempt.failed", job_id=job_id, input_key=input_key, attempt_index=idx, status="failed", duration_s=attempt["render_s"], preset=video_cfg.get("preset"), error=str(exc), error_type=exc.__class__.__name__)

        if final_mux is None or not final_mux.exists():
            raise RuntimeError("All render attempts failed")

        debug["workflow_used"] = workflow_used
        debug["status"] = "success"
        debug["timings"]["total_s"] = round(time.time() - start_t, 3)

        final_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/final.mp4"
        debug_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/debug.json"
        io.write_output(final_mux, final_key)
        debug["timings"]["upload_video_s"] = 0.0

        debug_path = local_case_dir / "debug.json"
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        io.write_output(debug_path, debug_key)
        debug["timings"]["upload_debug_s"] = 0.0

        _log("info", "image.done", job_id=job_id, input_key=input_key, status="success", output_key=final_key, total_s=debug["timings"]["total_s"], openai_total_tokens=debug.get("openai", {}).get("usage", {}).get("total_tokens", 0))

        if not debug_enabled:
            _cleanup_intermediates(local_case_dir=local_case_dir, render_input=render_input, source_input=local_input, attempt=last_attempt)
        return True

    except Exception as exc:
        debug["error"] = str(exc)
        debug["error_type"] = exc.__class__.__name__
        debug["timings"]["total_s"] = round(time.time() - start_t, 3)
        debug_path = local_case_dir / "debug.json"
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        debug_key = f"{output_prefix.rstrip('/')}/{job_id}/{image_name}/debug.json"
        io.write_output(debug_path, debug_key)
        _log("error", "image.done", job_id=job_id, input_key=input_key, status="failed", total_s=debug["timings"]["total_s"], error=str(exc), error_type=exc.__class__.__name__)
        if not debug_enabled:
            _cleanup_intermediates(local_case_dir=local_case_dir, render_input=render_input, source_input=local_input, attempt=last_attempt)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run image->video batch orchestrator (local-only)")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-fail-ratio", type=float, default=0.3)
    parser.add_argument("--local-input-dir", required=True)
    parser.add_argument("--local-output-dir", required=True)
    parser.add_argument("--debug", action="store_true", help="Keep intermediate artifacts (cropped image, temp case dir, comfy/audio intermediates).")
    parser.add_argument("--video-params-json", default="", help="JSON object of runtime video overrides.")
    args = parser.parse_args()

    input_dir = Path(args.local_input_dir).resolve()
    output_dir = Path(args.local_output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("--local-input-dir and --local-output-dir must be different directories.")

    work_root = Path(os.environ.get("WORK_ROOT", "/tmp/orchestrator")) / args.job_id
    work_root.mkdir(parents=True, exist_ok=True)

    video_overrides = dict(DEFAULT_VIDEO_OVERRIDES)
    if args.video_params_json:
        parsed = json.loads(args.video_params_json)
        if not isinstance(parsed, dict):
            raise ValueError("--video-params-json must be a JSON object")
        video_overrides.update(parsed)
    _log("info", "video.overrides", job_id=args.job_id, overrides=video_overrides)

    io = LocalIO(input_dir=input_dir, output_dir=output_dir)
    comfy_url = _resolve_comfy_url(os.environ.get("COMFY_URL", "http://localhost:18188"))
    comfy = ComfyClient(comfy_url)
    audio_url = os.environ.get("AUDIO_URL", "http://localhost:8000")
    templates_root = _resolve_templates_root()

    _log("info", "service.endpoint.selected", service="comfy", url=comfy.base_url)
    _wait_http_ok(f"{audio_url}/health", timeout_s=30, label="audio")

    image_keys = io.list_images(args.input_prefix)
    if not image_keys:
        logger.warning("no images found", extra={"extra": {"prefix": args.input_prefix}})
        return 0

    failures = 0
    for key in image_keys:
        ok = process_one_image(
            io=io,
            comfy=comfy,
            audio_url=audio_url,
            templates_root=templates_root,
            input_key=key,
            output_prefix=args.output_prefix,
            job_id=args.job_id,
            work_root=work_root,
            video_overrides=video_overrides,
            debug_enabled=args.debug,
        )
        failures += 0 if ok else 1

    total = len(image_keys)
    fail_ratio = failures / total
    _log("info", "batch.done", job_id=args.job_id, total=total, failures=failures, fail_ratio=fail_ratio, max_fail_ratio=args.max_fail_ratio)
    return 1 if fail_ratio > args.max_fail_ratio else 0


if __name__ == "__main__":
    sys.exit(main())
