import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import requests
from PIL import Image


TILED_VAE_TILE_SIZE = 512
TILED_VAE_OVERLAP = 64
TILED_VAE_TEMPORAL_SIZE = 64
TILED_VAE_TEMPORAL_OVERLAP = 8


def tiled_vae_decode_node(samples_node: str, vae_node: str) -> Dict[str, Any]:
    """Build a spatially tiled decode without short temporal seam artifacts."""
    return {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": [samples_node, 0],
            "vae": [vae_node, 0],
            "tile_size": TILED_VAE_TILE_SIZE,
            "overlap": TILED_VAE_OVERLAP,
            "temporal_size": TILED_VAE_TEMPORAL_SIZE,
            "temporal_overlap": TILED_VAE_TEMPORAL_OVERLAP,
        },
    }


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
        if "prompt_id" not in data:
            raise RuntimeError(f"Comfy rejected workflow: {json.dumps(data, ensure_ascii=True)[:1600]}")
        return data["prompt_id"]

    def get_object_info(self) -> Dict[str, Any]:
        if self._object_info_cache is None:
            self._object_info_cache = self._request_json("GET", "/object_info", timeout=60)
        return self._object_info_cache

    def _resolve_model_combo(self, node_name: str, input_name: str, desired_name: str, empty_hint: str) -> str:
        info = self.get_object_info()
        loader_info = info.get(node_name, {})
        input_cfg = loader_info.get("input", {})
        required = input_cfg.get("required", {})
        model_meta = required.get(input_name, [])
        available = model_meta[0] if isinstance(model_meta, list) and model_meta else []
        if isinstance(available, list):
            if not available:
                raise RuntimeError(empty_hint)
            if desired_name in available:
                return desired_name
            return available[0]
        return desired_name

    def resolve_diffusion_model(self, desired_name: str) -> str:
        return self._resolve_model_combo(
            "UNETLoader",
            "unet_name",
            desired_name,
            "No diffusion models found in Comfy. Place model files under models/diffusion_models.",
        )

    def resolve_vae(self, desired_name: str) -> str:
        return self._resolve_model_combo(
            "VAELoader",
            "vae_name",
            desired_name,
            "No VAE models found in Comfy. Place VAE files under models/vae.",
        )

    def resolve_clip_vision(self, desired_name: str) -> str:
        return self._resolve_model_combo(
            "CLIPVisionLoader",
            "clip_name",
            desired_name,
            "No CLIP vision models found in Comfy. Place CLIP vision files under models/clip_vision.",
        )

    def resolve_text_encoder(self, input_name: str, desired_name: str) -> str:
        return self._resolve_model_combo(
            "DualCLIPLoader",
            input_name,
            desired_name,
            "No text encoders found in Comfy. Place text encoder files under models/text_encoders.",
        )

    def resolve_single_text_encoder(self, desired_name: str) -> str:
        return self._resolve_model_combo(
            "CLIPLoader",
            "clip_name",
            desired_name,
            "No text encoders found in Comfy. Place model files under models/text_encoders.",
        )

    def wait_for_prompt(self, prompt_id: str, timeout_s: int | None = None) -> Dict[str, Any]:
        # Poll /history until Comfy has completed execution for this prompt.
        if timeout_s is None:
            timeout_s = int(os.environ.get("COMFY_PROMPT_TIMEOUT_S", "3600"))
        deadline = time.time() + timeout_s
        missing_since: float | None = None
        while time.time() < deadline:
            payload = self._request_json("GET", f"/history/{prompt_id}", timeout=30)
            if prompt_id in payload:
                item = payload[prompt_id]
                status = item.get("status", {}) if isinstance(item, dict) else {}
                status_str = str(status.get("status_str", ""))
                if status_str in {"error", "failed"}:
                    messages = status.get("messages", [])
                    raise RuntimeError(
                        f"Comfy prompt failed: prompt_id={prompt_id} status={status_str} messages_tail={str(messages[-1:])[:1200]}"
                    )
                completed = bool(status.get("completed", False))
                if not completed:
                    time.sleep(3)
                    continue
                if status_str and status_str != "success":
                    messages = status.get("messages", [])
                    raise RuntimeError(
                        f"Comfy prompt failed: prompt_id={prompt_id} status={status_str} messages_tail={str(messages[-1:])[:1200]}"
                    )
                return item
            queue = self._request_json("GET", "/queue", timeout=30)
            queued = False
            for bucket in ("queue_running", "queue_pending"):
                for entry in queue.get(bucket, []):
                    if isinstance(entry, list) and len(entry) > 1 and entry[1] == prompt_id:
                        queued = True
                        break
                if queued:
                    break
            if queued:
                missing_since = None
            elif missing_since is None:
                missing_since = time.time()
            elif time.time() - missing_since > 30:
                raise RuntimeError(f"Comfy prompt disappeared from queue/history: prompt_id={prompt_id}")
            time.sleep(3)
        raise TimeoutError(f"Comfy prompt timed out: {prompt_id}")

    def diagnostics(self, prompt_id: str | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        checks = [("queue", "/queue"), ("system_stats", "/system_stats")]
        if prompt_id:
            checks.insert(1, ("history", f"/history/{prompt_id}"))
        for key, path in checks:
            try:
                out[key] = self._request_json("GET", path, timeout=10)
            except Exception as exc:
                out[key] = {"error": str(exc), "error_type": exc.__class__.__name__}
        return out

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> Dict[str, Any]:
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        url = f"{self.base_url}/free"
        try:
            resp = self.session.post(url, json=payload, timeout=30)
            if resp.status_code >= 400:
                return {"error": resp.text[:1200], "status_code": resp.status_code}
            if not resp.text.strip():
                return {"status_code": resp.status_code, "payload": payload}
            try:
                return resp.json()
            except ValueError:
                return {"status_code": resp.status_code, "text": resp.text[:1200], "payload": payload}
        except Exception as exc:
            return {"error": str(exc), "error_type": exc.__class__.__name__, "payload": payload}

    def clear_queue(self) -> Dict[str, Any]:
        """Interrupt orphaned work and remove pending prompts before a new batch."""
        results: Dict[str, Any] = {}
        for name, path, payload in (
            ("interrupt", "/interrupt", {}),
            ("clear", "/queue", {"clear": True}),
        ):
            try:
                response = self.session.post(f"{self.base_url}{path}", json=payload, timeout=30)
                results[name] = {"status_code": response.status_code, "ok": response.status_code < 400}
            except Exception as exc:
                results[name] = {"error": str(exc), "error_type": exc.__class__.__name__}
        return results

    @staticmethod
    def _render_template(template_path: Path, substitutions: Dict[str, Any]) -> Dict[str, Any]:
        # Workflow JSON templates use placeholder tokens for runtime injection.
        text = template_path.read_text(encoding="utf-8")
        for key, value in substitutions.items():
            text = text.replace(key, str(value))
        return json.loads(text)

    @staticmethod
    def _resolve_dimensions(decision_video: Dict[str, Any], input_image: str | None = None) -> tuple[int, int]:
        params = decision_video.get("params", {})
        resolution = int(decision_video["resolution_width"])
        target_aspect = str(params.get("target_aspect", ""))
        max_dimension = int(params.get("max_dimension", 1280 if str(decision_video.get("preset", "")).startswith("HUNYUAN15_") else 768))

        if bool(params.get("preserve_source_aspect")) and input_image:
            with Image.open(input_image) as image:
                source_width, source_height = image.size
            long_edge = max(384, min(max_dimension, resolution))
            scale = long_edge / max(source_width, source_height)
            width = max(192, int(round((source_width * scale) / 16.0) * 16))
            height = max(192, int(round((source_height * scale) / 16.0) * 16))
            return min(max_dimension, width), min(max_dimension, height)

        # Instagram Reel mode: portrait 9:16.
        if target_aspect == "instagram_reel_9_16":
            height = max(384, min(max_dimension, int(round(resolution / 64) * 64)))
            # Keep the internal render genuinely portrait. A 384px minimum width
            # turns small 9:16 Hunyuan comparison renders into square videos.
            width = max(192, min(max_dimension, int(round((height * 9 / 16) / 64) * 64)))
            return width, height

        # Default mode: landscape 16:9.
        width = resolution
        height = int(width * 9 / 16)
        width = max(384, min(max_dimension, int(round(width / 64) * 64)))
        height = max(384, min(max_dimension, int(round(height / 64) * 64)))
        return width, height

    def build_hunyuan15_i2v_workflow(self, template_path: Path, input_image: str, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        width, height = self._resolve_dimensions(decision_video, input_image)
        frames = int(decision_video["frames"])
        fps = int(decision_video["fps"])
        seed = int(decision_video["seed"])
        prompt = str(
            params.get(
                "prompt",
                params.get(
                    "video_prompt",
                    "cinematic image-to-video motion, preserve subject identity and composition, natural camera movement",
                ),
            )
        )
        negative_prompt = str(
            params.get(
                "negative_prompt",
                "low quality, blurry, distorted, deformed, text, watermark, flicker, jitter",
            )
        )
        diffusion_model = self.resolve_diffusion_model(
            str(params.get("diffusion_model", "hunyuanvideo1.5_720p_i2v_fp16.safetensors"))
        )
        workflow = self._render_template(
            template_path,
            {
                "__INPUT_IMAGE__": input_image,
                "__CLIP_VISION_NAME__": self.resolve_clip_vision(str(params.get("clip_vision_name", "sigclip_vision_patch14_384.safetensors"))),
                "__TEXT_ENCODER_1__": self.resolve_text_encoder("clip_name1", str(params.get("text_encoder_1", "qwen_2.5_vl_7b_fp8_scaled.safetensors"))),
                "__TEXT_ENCODER_2__": self.resolve_text_encoder("clip_name2", str(params.get("text_encoder_2", "byt5_small_glyphxl_fp16.safetensors"))),
                "__VAE_NAME__": self.resolve_vae(str(params.get("vae_name", "hunyuanvideo15_vae_fp16.safetensors"))),
                "__DIFFUSION_MODEL__": diffusion_model,
                "__WEIGHT_DTYPE__": str(params.get("weight_dtype", "default")),
                "__PROMPT_JSON__": json.dumps(prompt),
                "__NEGATIVE_PROMPT_JSON__": json.dumps(negative_prompt),
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": frames,
                "__FPS__": fps,
                "__SEED__": seed,
                "__STEPS__": int(params.get("steps", 20)),
                "__CFG__": float(params.get("cfg", 6.0)),
                "__SHIFT__": float(params.get("shift", 7.0)),
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )
        if bool(params.get("tiled_vae")):
            workflow["16"] = tiled_vae_decode_node("15", "3")
        return workflow

    def build_wan22_i2v_workflow(self, template_path: Path, input_image: str, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        params = decision_video.get("params", {})
        width, height = self._resolve_dimensions(decision_video, input_image)
        prompt = str(params.get("prompt", "Natural environmental motion. Preserve the original composition and viewpoint."))
        negative_prompt = str(params.get("negative_prompt", "flicker, jitter, unstable geometry, inconsistent appearance, scene transition, low quality"))
        workflow = self._render_template(
            template_path,
            {
                "__INPUT_IMAGE__": input_image,
                "__DIFFUSION_MODEL__": self.resolve_diffusion_model(str(params.get("diffusion_model", "wan2.2_ti2v_5B_fp16.safetensors"))),
                "__WEIGHT_DTYPE__": str(params.get("weight_dtype", "default")),
                "__TEXT_ENCODER__": self.resolve_single_text_encoder(str(params.get("text_encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))),
                "__VAE_NAME__": self.resolve_vae(str(params.get("vae_name", "wan2.2_vae.safetensors"))),
                "__PROMPT_JSON__": json.dumps(prompt),
                "__NEGATIVE_PROMPT_JSON__": json.dumps(negative_prompt),
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": int(decision_video["frames"]),
                "__FPS__": int(decision_video["fps"]),
                "__SEED__": int(decision_video["seed"]),
                "__STEPS__": int(params.get("steps", 20)),
                "__CFG__": float(params.get("cfg", 5.0)),
                "__SHIFT__": float(params.get("shift", 8.0)),
                "__SAMPLER__": str(params.get("sampler_name", "uni_pc")),
                "__SCHEDULER__": str(params.get("scheduler", "simple")),
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )
        if bool(params.get("tiled_vae", False)):
            workflow["10"] = tiled_vae_decode_node("9", "3")
        return workflow

    def build_deterministic_workflow(self, template_path: Path, input_image: str, output_prefix: str, decision_video: Dict[str, Any]) -> Dict[str, Any]:
        width, height = self._resolve_dimensions(decision_video, input_image)
        return self._render_template(
            template_path,
            {
                "__INPUT_IMAGE__": input_image,
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__FRAMES__": int(decision_video.get("frames", 150)),
                "__FPS__": int(decision_video.get("fps", 30)),
                "__OUTPUT_PREFIX__": f"{output_prefix}-{uuid.uuid4().hex[:6]}",
            },
        )


def find_latest_mp4(history_payload: Dict[str, Any], output_root: Path, expected_prefix: str = "") -> Path:
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

    pattern = "*.mp4"
    if expected_prefix:
        pattern = f"{expected_prefix}*.mp4"
    fallback = sorted((output_root / "comfy").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"No mp4 output found from ComfyUI history (expected_prefix={expected_prefix or '<any>'})"
    )
