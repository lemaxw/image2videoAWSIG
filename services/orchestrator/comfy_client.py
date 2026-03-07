import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import requests


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._object_info_cache: Dict[str, Any] | None = None

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                resp = self.session.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
                if resp.status_code >= 400:
                    body = (resp.text or "")[:1200]
                    raise RuntimeError(f"Comfy {method} {path} failed ({resp.status_code}): {body}")
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt == 5:
                    break
                time.sleep(min(2 * attempt, 6))
        raise RuntimeError(f"Comfy request failed after retries: {method} {path}; error={last_exc}")

    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        data = self._request_json("POST", "/prompt", json={"prompt": workflow}, timeout=90)
        return data["prompt_id"]

    def get_object_info(self) -> Dict[str, Any]:
        if self._object_info_cache is None:
            self._object_info_cache = self._request_json("GET", "/object_info", timeout=60)
        return self._object_info_cache

    def resolve_svd_checkpoint(self, desired_ckpt: str = "svd_xt.safetensors") -> str:
        try:
            info = self.get_object_info()
            loader_info = info.get("ImageOnlyCheckpointLoader", {})
            input_cfg = loader_info.get("input", {})
            required = input_cfg.get("required", {})
            ckpt_meta = required.get("ckpt_name", [])
            available = ckpt_meta[0] if isinstance(ckpt_meta, list) and ckpt_meta else []
            if isinstance(available, list):
                if not available:
                    raise RuntimeError(
                        "No checkpoints available in Comfy ImageOnlyCheckpointLoader. "
                        "Place an SVD checkpoint under models/checkpoints."
                    )
                if desired_ckpt in available:
                    return desired_ckpt
                return available[0]
        except Exception:
            raise
        return desired_ckpt

    def wait_for_prompt(self, prompt_id: str, timeout_s: int = 900) -> Dict[str, Any]:
        # Poll /history until Comfy has materialized output metadata for this prompt.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            payload = self._request_json("GET", f"/history/{prompt_id}", timeout=30)
            if prompt_id in payload:
                return payload[prompt_id]
            time.sleep(3)
        raise TimeoutError(f"Comfy prompt timed out: {prompt_id}")

    @staticmethod
    def _render_template(template_path: Path, substitutions: Dict[str, Any]) -> Dict[str, Any]:
        # Workflow JSON templates use placeholder tokens for runtime injection.
        text = template_path.read_text(encoding="utf-8")
        for key, value in substitutions.items():
            text = text.replace(key, str(value))
        return json.loads(text)

    @staticmethod
    def _resolve_dimensions(decision_video: Dict[str, Any]) -> tuple[int, int]:
        params = decision_video.get("params", {})
        resolution = int(params.get("resolution_width", decision_video["resolution_width"]))
        target_aspect = str(params.get("target_aspect", ""))

        # Instagram Reel mode: portrait 9:16.
        if target_aspect == "instagram_reel_9_16":
            height = max(384, min(768, int(round(resolution / 64) * 64)))
            width = max(384, min(768, int(round((height * 9 / 16) / 64) * 64)))
            return width, height

        # Default mode: landscape 16:9.
        width = resolution
        height = int(width * 9 / 16)
        height = max(384, min(768, int(round(height / 64) * 64)))
        return width, height

    def build_svd_workflow(self, template_path: Path, input_image: str, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        ckpt_name = self.resolve_svd_checkpoint()
        width, height = self._resolve_dimensions(decision_video)
        frames = int(params.get("frames", decision_video["frames"]))
        fps = int(params.get("fps", decision_video["fps"]))
        seed = int(params.get("seed", decision_video["seed"]))
        return self._render_template(
            template_path,
            {
                "__INPUT_IMAGE__": input_image,
                "__CKPT_NAME__": ckpt_name,
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": frames,
                "__FPS__": fps,
                "__SEED__": seed,
                "__STEPS__": int(params.get("steps", 16)),
                "__MOTION_BUCKET_ID__": int(params.get("motion_bucket_id", 30)),
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )

    def build_animatediff_workflow(self, template_path: Path, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        width, height = self._resolve_dimensions(decision_video)
        motion_strength = float(params.get("motion_strength", 35))
        denoise = max(0.2, min(0.95, motion_strength / 100.0))
        frames = int(params.get("frames", decision_video["frames"]))
        fps = int(params.get("fps", decision_video["fps"]))
        seed = int(params.get("seed", decision_video["seed"]))
        return self._render_template(
            template_path,
            {
                "__PROMPT__": params.get("prompt", "cinematic scene with subtle movement"),
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": frames,
                "__FPS__": fps,
                "__SEED__": seed,
                "__STEPS__": int(params.get("steps", 18)),
                "__CFG__": float(params.get("cfg", 3.5)),
                "__DENOISE__": denoise,
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )


def find_latest_mp4(history_payload: Dict[str, Any], output_root: Path) -> Path:
    # Prefer explicit history references; fallback to latest file for custom nodes.
    candidates = []
    outputs = history_payload.get("outputs", {})
    for node_out in outputs.values():
        for item in node_out.get("gifs", []) + node_out.get("images", []) + node_out.get("videos", []):
            filename = item.get("filename", "")
            if filename.endswith(".mp4"):
                path = output_root / "comfy" / filename
                if path.exists():
                    candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    fallback = sorted((output_root / "comfy").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]

    raise FileNotFoundError("No mp4 output found from ComfyUI history")
