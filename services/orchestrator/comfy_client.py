import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import requests


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        resp = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["prompt_id"]

    def wait_for_prompt(self, prompt_id: str, timeout_s: int = 900) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                if prompt_id in payload:
                    return payload[prompt_id]
            time.sleep(3)
        raise TimeoutError(f"Comfy prompt timed out: {prompt_id}")

    @staticmethod
    def _render_template(template_path: Path, substitutions: Dict[str, Any]) -> Dict[str, Any]:
        text = template_path.read_text(encoding="utf-8")
        for key, value in substitutions.items():
            text = text.replace(key, str(value))
        return json.loads(text)

    def build_svd_workflow(self, template_path: Path, input_image: str, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        width = int(decision_video["resolution_width"])
        height = int(width * 9 / 16)
        height = max(384, min(768, int(round(height / 64) * 64)))
        return self._render_template(
            template_path,
            {
                "__INPUT_IMAGE__": input_image,
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": int(decision_video["frames"]),
                "__FPS__": int(decision_video["fps"]),
                "__SEED__": int(decision_video["seed"]),
                "__STEPS__": int(params.get("steps", 16)),
                "__MOTION_BUCKET_ID__": int(params.get("motion_bucket_id", 30)),
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )

    def build_animatediff_workflow(self, template_path: Path, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        width = int(decision_video["resolution_width"])
        height = int(width * 9 / 16)
        height = max(384, min(768, int(round(height / 64) * 64)))
        motion_strength = float(params.get("motion_strength", 35))
        denoise = max(0.2, min(0.95, motion_strength / 100.0))
        return self._render_template(
            template_path,
            {
                "__PROMPT__": params.get("prompt", "cinematic scene with subtle movement"),
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": int(decision_video["frames"]),
                "__FPS__": int(decision_video["fps"]),
                "__SEED__": int(decision_video["seed"]),
                "__STEPS__": int(params.get("steps", 18)),
                "__CFG__": float(params.get("cfg", 3.5)),
                "__DENOISE__": denoise,
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )


def find_latest_mp4(history_payload: Dict[str, Any], output_root: Path) -> Path:
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
