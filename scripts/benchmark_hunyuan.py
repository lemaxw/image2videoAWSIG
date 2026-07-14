#!/usr/bin/env python3
"""Submit one unconstrained Hunyuan 1.5 benchmark render to ComfyUI.

This intentionally bypasses production decision validation so temporal lengths
supported by the live Comfy node can be tested without changing defaults.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from services.orchestrator.comfy_client import ComfyClient, find_latest_mp4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-image", required=True, help="Comfy-visible image path/name")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="flicker, jitter, warped geometry, disappearing objects, added objects")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution-width", type=int, default=704)
    parser.add_argument("--width", type=int, default=0, help="Explicit native render width; overrides aspect resolution")
    parser.add_argument("--height", type=int, default=0, help="Explicit native render height; overrides aspect resolution")
    parser.add_argument("--tiled-vae", action="store_true", help="Decode long videos in spatial/temporal tiles")
    parser.add_argument("--target-aspect", choices=("landscape", "instagram_reel_9_16"), default="landscape")
    parser.add_argument("--comfy-url", default="http://comfyui:8188")
    parser.add_argument("--output-dir", default="/data/outputs/hunyuan-benchmark")
    parser.add_argument("--template", default="/app/services/comfy/workflow_templates/hunyuan15_i2v_workflow.json")
    args = parser.parse_args()

    if args.frames < 1 or (args.frames != 30 and (args.frames - 1) % 4):
        raise SystemExit("Use the production baseline of 30 or a native 4n+1 length such as 49 or 61")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(args.comfy_url)
    video = {
        "preset": "HUNYUAN15_I2V_720P",
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "resolution_width": args.resolution_width,
        "params": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "steps": args.steps,
            "cfg": 5.8,
            "shift": 7.0,
            "weight_dtype": "fp8_e4m3fn",
            "diffusion_model": "hunyuanvideo1.5_720p_i2v_fp16.safetensors",
            "text_encoder_1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "text_encoder_2": "byt5_small_glyphxl_fp16.safetensors",
            "clip_vision_name": "sigclip_vision_patch14_384.safetensors",
            "vae_name": "hunyuanvideo15_vae_fp16.safetensors",
        },
    }
    if args.target_aspect == "instagram_reel_9_16":
        video["params"]["target_aspect"] = "instagram_reel_9_16"

    started = time.time()
    workflow = client.build_hunyuan15_i2v_workflow(
        Path(args.template), args.input_image, f"hunyuan-benchmark/{args.name}", video
    )
    if args.width and args.height:
        if args.width % 16 or args.height % 16:
            raise SystemExit("Explicit Hunyuan dimensions must be divisible by 16")
        workflow["9"]["inputs"]["width"] = args.width
        workflow["9"]["inputs"]["height"] = args.height
    if args.tiled_vae:
        workflow["16"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["15", 0],
                "vae": ["3", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 16,
                "temporal_overlap": 4,
            },
        }
    prompt_id = client.submit_workflow(workflow)
    try:
        history = client.wait_for_prompt(prompt_id)
        raw_path = find_latest_mp4(history, Path("/data/outputs"), expected_prefix=f"hunyuan-benchmark/{args.name}")
        final_path = output_dir / f"{args.name}.mp4"
        shutil.copy2(raw_path, final_path)
        result = {
            "name": args.name,
            "status": "success",
            "prompt_id": prompt_id,
            "input_image": args.input_image,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "width": args.width or workflow["9"]["inputs"]["width"],
            "height": args.height or workflow["9"]["inputs"]["height"],
            "tiled_vae": args.tiled_vae,
            "seed": args.seed,
            "target_aspect": args.target_aspect,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "raw_path": str(raw_path),
            "path": str(final_path),
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception as exc:
        result = {
            "name": args.name,
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
