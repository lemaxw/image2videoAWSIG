#!/usr/bin/env python3
"""Smoke-test the mounted SAM 2.1 and Depth Anything V2 checkpoints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


INPUT = Path("/data/inputs/1_20260627_130019.jpg")
OUTPUT = Path("/data/outputs/wan-benchmark")
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_sam(image_rgb: np.ndarray) -> dict:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    started = time.time()
    checkpoint = "/opt/ComfyUI/models/sam2/sam2.1_hiera_base_plus.pt"
    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image_rgb)
    height, width = image_rgb.shape[:2]
    # Loose box around the visible stream; the pipeline will obtain this from image2json.
    box = np.array([0.39 * width, 0.46 * height, 0.79 * width, 0.99 * height], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    mask = masks[int(np.argmax(scores))].astype(np.uint8) * 255
    cv2.imwrite(str(OUTPUT / "valley_sam21_stream_mask.png"), mask)
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([0, 220, 255])).astype(np.uint8)
    cv2.imwrite(str(OUTPUT / "valley_sam21_stream_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Point-guided sky/cloud mask. Negative points keep the mountain slopes and
    # tree line out of the result; image2json can supply equivalent regions.
    cloud_points = np.array(
        [
            [0.18 * width, 0.10 * height],
            [0.48 * width, 0.12 * height],
            [0.72 * width, 0.11 * height],
            [0.48 * width, 0.30 * height],
            [0.10 * width, 0.42 * height],
            [0.52 * width, 0.43 * height],
            [0.86 * width, 0.30 * height],
        ],
        dtype=np.float32,
    )
    cloud_labels = np.array([1, 1, 1, 1, 0, 0, 0], dtype=np.int32)
    cloud_masks, cloud_scores, _ = predictor.predict(
        point_coords=cloud_points,
        point_labels=cloud_labels,
        multimask_output=True,
    )
    cloud_mask = cloud_masks[int(np.argmax(cloud_scores))].astype(np.uint8) * 255
    cv2.imwrite(str(OUTPUT / "valley_sam21_cloud_mask.png"), cloud_mask)

    # Limit cloud motion to the upper scene even if SAM leaks into vegetation.
    cloud_mask[int(0.48 * height) :, :] = 0
    combined = cv2.bitwise_or(mask, cloud_mask)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(str(OUTPUT / "valley_sam21_cloud_stream_mask.png"), combined)
    combined_overlay = image_rgb.copy()
    combined_overlay[combined > 0] = (
        0.45 * combined_overlay[combined > 0] + 0.55 * np.array([255, 80, 210])
    ).astype(np.uint8)
    cv2.imwrite(
        str(OUTPUT / "valley_sam21_cloud_stream_overlay.jpg"),
        cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR),
    )
    elapsed = time.time() - started
    del predictor, model
    torch.cuda.empty_cache()
    return {
        "elapsed_s": round(elapsed, 3),
        "stream_score": float(np.max(scores)),
        "cloud_score": float(np.max(cloud_scores)),
        "stream_mask_pixels": int((mask > 0).sum()),
        "cloud_mask_pixels": int((cloud_mask > 0).sum()),
        "combined_mask_pixels": int((combined > 0).sum()),
    }


def run_depth(image_bgr: np.ndarray) -> dict:
    sys.path.insert(0, "/opt/Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    started = time.time()
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(
        "/opt/ComfyUI/models/depth_anything/depth_anything_v2_vits.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model = model.to("cuda").eval()
    depth = model.infer_image(image_bgr, input_size=518)
    normalized = ((depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT / "valley_depth_anything_v2_small.png"), normalized)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(OUTPUT / "valley_depth_anything_v2_small_color.jpg"), colored)
    elapsed = time.time() - started
    del model
    torch.cuda.empty_cache()
    return {
        "elapsed_s": round(elapsed, 3),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
    }


def main() -> None:
    image_bgr = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(INPUT)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = {"sam2": run_sam(image_rgb), "depth_anything_v2": run_depth(image_bgr)}
    (OUTPUT / "vision_models_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
