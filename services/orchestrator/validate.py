import copy
from typing import Any, Dict, List

ALLOWED_PRESETS = {
    "SVD_SUBTLE",
    "SVD_STRONG",
    "ANIMATEDIFF_GRASS_WIND",
    "ANIMATEDIFF_CITY_PULSE",
    "FAILSAFE_LOW_MEM",
}

ALLOWED_CROP_ANCHORS = {
    "left_top",
    "center_top",
    "right_top",
    "left_center",
    "center_center",
    "right_center",
    "left_bottom",
    "center_bottom",
    "right_bottom",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        ivalue = low
    return int(clamp(ivalue, low, high))


def _nearest_mult_64(width: int) -> int:
    # Comfy workflows and many diffusion checkpoints are more stable on /64 widths.
    rounded = int(round(width / 64.0) * 64)
    return clamp_int(rounded, 384, 768)


def default_video_for_preset(preset: str) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {
        "SVD_SUBTLE": {
            "fps": 5,
            "frames": 25,
            "resolution_width": 768,
            "seed": 123,
            "params": {"motion_bucket_id": 20, "steps": 22},
        },
        "SVD_STRONG": {
            "fps": 6,
            "frames": 25,
            "resolution_width": 768,
            "seed": 123,
            "params": {"motion_bucket_id": 38, "steps": 24},
        },
        "ANIMATEDIFF_GRASS_WIND": {
            "fps": 5,
            "frames": 25,
            "resolution_width": 704,
            "seed": 123,
            "params": {"motion_bucket_id": 26, "steps": 22},
        },
        "ANIMATEDIFF_CITY_PULSE": {
            "fps": 6,
            "frames": 25,
            "resolution_width": 768,
            "seed": 123,
            "params": {"motion_bucket_id": 42, "steps": 24},
        },
        "FAILSAFE_LOW_MEM": {
            "fps": 4,
            "frames": 16,
            "resolution_width": 512,
            "seed": 123,
            "params": {"motion_bucket_id": 16, "steps": 14},
        },
    }
    profile = profiles.get(preset, profiles["FAILSAFE_LOW_MEM"])
    return {
        "preset": preset,
        "duration_s": 5,
        "fps": profile["fps"],
        "frames": profile["frames"],
        "resolution_width": profile["resolution_width"],
        "seed": profile["seed"],
        "params": dict(profile["params"]),
    }


def _validate_video(video: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(video, dict):
        video = {}
    requested_preset = str(video.get("preset", "FAILSAFE_LOW_MEM"))
    preset = requested_preset
    if preset not in ALLOWED_PRESETS:
        preset = "FAILSAFE_LOW_MEM"

    # Current animatediff template is text-to-video (EmptyLatentImage), not image-conditioned.
    # To preserve "animate from input image" behavior, map animatediff intents to SVD families.
    if preset == "ANIMATEDIFF_GRASS_WIND":
        preset = "SVD_SUBTLE"
    elif preset == "ANIMATEDIFF_CITY_PULSE":
        preset = "SVD_STRONG"

    # Start from requested-preset defaults, then overlay model output.
    # This preserves semantic differences (e.g. FAILSAFE_LOW_MEM vs SVD_STRONG)
    # even when animatediff intents are mapped onto SVD execution.
    defaults_key = requested_preset if requested_preset in ALLOWED_PRESETS else preset
    v = copy.deepcopy(default_video_for_preset(defaults_key))
    v.update({k: video.get(k, v[k]) for k in ["duration_s", "fps", "frames", "resolution_width", "seed", "params"]})
    v["preset"] = preset

    params = v.get("params") or {}
    v["duration_s"] = 3 if int(video.get("duration_s", params.get("duration_s", v.get("duration_s", 5)))) == 3 else 5
    v["fps"] = clamp_int(video.get("fps", params.get("fps", v.get("fps", 6))), 3, 10)
    v["frames"] = clamp_int(video.get("frames", params.get("frames", v.get("frames", 20))), 10, 25)
    v["resolution_width"] = _nearest_mult_64(
        clamp_int(video.get("resolution_width", params.get("resolution_width", v.get("resolution_width", 576))), 384, 768)
    )
    v["seed"] = clamp_int(video.get("seed", params.get("seed", v.get("seed", 42))), 0, 2**31 - 1)

    # Clamp preset-specific parameters to safe execution ranges.
    if preset.startswith("SVD") or preset == "FAILSAFE_LOW_MEM":
        motion_strength = clamp_int(params.get("motion_strength", 35), 10, 80)
        mapped_motion = clamp_int(int(round(motion_strength * 0.75)), 10, 80)
        v["params"] = {
            "motion_bucket_id": clamp_int(params.get("motion_bucket_id", mapped_motion), 10, 80),
            "steps": clamp_int(params.get("steps", 16), 12, 25),
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
            "requested_preset": requested_preset,
        }
    else:
        v["params"] = {
            "steps": clamp_int(params.get("steps", 18), 12, 30),
            "cfg": float(clamp(float(params.get("cfg", 3.5)), 1.5, 8.0)),
            "motion_strength": clamp_int(params.get("motion_strength", 35), 10, 80),
            "prompt": str(params.get("prompt", "subtle cinematic movement"))[:240],
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
        }

    return v


def validate_and_clamp_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    scene = raw.get("scene") or {}
    if not isinstance(scene, dict):
        scene = {}

    video = _validate_video(raw.get("video") or {})

    framing = raw.get("framing") or {}
    if not isinstance(framing, dict):
        framing = {}
    anchor = str(framing.get("crop_anchor", "center_center"))
    if anchor not in ALLOWED_CROP_ANCHORS:
        anchor = "center_center"
    framing_out = {
        "target_aspect": "instagram_reel_9_16",
        "crop_anchor": anchor,
    }

    audio = raw.get("audio") or {}
    if not isinstance(audio, dict):
        audio = {}
    audio_out = {
        "prompt": " ".join(str(audio.get("prompt", "soft ambience")).split())[:96],
        "duration_s": 3 if int(audio.get("duration_s", video["duration_s"])) == 3 else 5,
        "mix_db": float(clamp(float(audio.get("mix_db", -8.0)), -24.0, 6.0)),
    }

    fallbacks_raw = raw.get("fallbacks") or []
    if not isinstance(fallbacks_raw, list):
        fallbacks_raw = []
    fallbacks_raw = [c for c in fallbacks_raw if isinstance(c, dict)]
    fallbacks: List[Dict[str, Any]] = []
    for candidate in fallbacks_raw[:2]:
        fallbacks.append(_validate_video(candidate))

    # Contract requires exactly two fallback candidates.
    while len(fallbacks) < 2:
        if len(fallbacks) == 0:
            fallbacks.append(default_video_for_preset("SVD_SUBTLE"))
        else:
            fallbacks.append(default_video_for_preset("FAILSAFE_LOW_MEM"))

    return {
        "scene": {
            "tags": [str(t)[:50] for t in (scene.get("tags") or [])[:8]],
            "has_people": bool(scene.get("has_people", False)),
            "confidence": float(clamp(float(scene.get("confidence", 0.5)), 0.0, 1.0)),
        },
        "framing": framing_out,
        "video": video,
        "audio": audio_out,
        "fallbacks": fallbacks,
    }
