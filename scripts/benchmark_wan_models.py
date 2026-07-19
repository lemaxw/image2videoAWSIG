#!/usr/bin/env python3
"""Run isolated Wan 2.2 benchmarks against ComfyUI."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from services.orchestrator.comfy_client import ComfyClient, find_latest_mp4


NEGATIVE = (
    "static image, frozen water, frozen clouds, camera shake, warped terrain, "
    "deformed buildings, flicker, jitter, added objects, disappearing objects, "
    "text, watermark, low quality"
)


def loader_nodes(model: str, vae: str) -> dict[str, dict]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "__IMAGE__"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "__PROMPT__", "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["1", 0]}},
    }


def output_nodes(
    fps: int,
    prefix: str,
    conditioning_node: str,
    latent_output: int = 0,
    *,
    cfg: float = 5.0,
    shift: float = 8.0,
    sampler_name: str = "uni_pc",
) -> dict[str, dict]:
    return {
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["7", 0],
                "seed": 42,
                "steps": 20,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": "simple",
                "positive": [conditioning_node, 0] if conditioning_node == "8" else ["5", 0],
                "negative": [conditioning_node, 1] if conditioning_node == "8" else ["6", 0],
                "latent_image": [conditioning_node, latent_output],
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": float(fps),
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def build_workflow(args: argparse.Namespace) -> dict[str, dict]:
    workflow = loader_nodes("wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_vae.safetensors")
    workflow["8"] = {
        "class_type": "Wan22ImageToVideoLatent",
        "inputs": {
            "vae": ["3", 0],
            "width": args.width,
            "height": args.height,
            "length": args.frames,
            "batch_size": 1,
            "start_image": ["4", 0],
        },
    }
    workflow.update(
        output_nodes(
            args.fps,
            f"wan-benchmark/{args.name}",
            "wan22",
            0,
            cfg=args.cfg,
            shift=args.shift,
            sampler_name=args.sampler_name,
        )
    )
    workflow["7"]["inputs"]["shift"] = args.shift
    workflow["9"]["inputs"]["latent_image"] = ["8", 0]

    workflow["4"]["inputs"]["image"] = args.input_image
    workflow["5"]["inputs"]["text"] = args.prompt
    workflow["6"]["inputs"]["text"] = args.negative_prompt
    workflow["9"]["inputs"]["seed"] = args.seed
    workflow["9"]["inputs"]["steps"] = args.steps
    return workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=NEGATIVE)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--sampler-name", default="uni_pc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comfy-url", default="http://comfyui:8188")
    parser.add_argument("--output-dir", default="/data/outputs/wan-benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(args.comfy_url)
    started = time.time()
    prompt_id = None
    try:
        workflow = build_workflow(args)
        prompt_id = client.submit_workflow(workflow)
        history = client.wait_for_prompt(prompt_id, timeout_s=5400)
        raw_path = find_latest_mp4(history, Path("/data/outputs"), f"wan-benchmark/{args.name}")
        final_path = output_dir / f"{args.name}.mp4"
        shutil.copy2(raw_path, final_path)
        result = {
            "name": args.name,
            "model": "wan22",
            "status": "success",
            "prompt_id": prompt_id,
            "input_image": args.input_image,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "cfg": args.cfg,
            "shift": args.shift,
            "sampler_name": args.sampler_name,
            "seed": args.seed,
            "prompt": args.prompt,
            "path": str(final_path),
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception as exc:
        result = {
            "name": args.name,
            "model": args.model,
            "status": "failed",
            "prompt_id": prompt_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "elapsed_s": round(time.time() - started, 3),
            "diagnostics": client.diagnostics(prompt_id),
        }
    with (output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    client.free_memory()
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
