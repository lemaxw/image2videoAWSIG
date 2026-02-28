import copy
from typing import Any, Dict, List

ALLOWED_PRESETS = {
    "SVD_SUBTLE",
    "SVD_STRONG",
    "ANIMATEDIFF_GRASS_WIND",
    "ANIMATEDIFF_CITY_PULSE",
    "FAILSAFE_LOW_MEM",
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
    rounded = int(round(width / 64.0) * 64)
    return clamp_int(rounded, 384, 768)


def default_video_for_preset(preset: str) -> Dict[str, Any]:
    base = {
        "preset": preset,
        "duration_s": 5,
        "fps": 6,
        "frames": 20,
        "resolution_width": 576,
        "seed": 42,
        "params": {},
    }
    if preset.startswith("SVD") or preset == "FAILSAFE_LOW_MEM":
        base["params"] = {"motion_bucket_id": 30, "steps": 16}
    else:
        base["params"] = {
            "steps": 18,
            "cfg": 3.5,
            "motion_strength": 35,
            "prompt": "subtle cinematic movement",
        }
    return base


def _validate_video(video: Dict[str, Any]) -> Dict[str, Any]:
    preset = video.get("preset", "FAILSAFE_LOW_MEM")
    if preset not in ALLOWED_PRESETS:
        preset = "FAILSAFE_LOW_MEM"

    v = copy.deepcopy(default_video_for_preset(preset))
    v.update({k: video.get(k, v[k]) for k in ["duration_s", "fps", "frames", "resolution_width", "seed", "params"]})
    v["preset"] = preset

    v["duration_s"] = 3 if int(v.get("duration_s", 5)) == 3 else 5
    v["fps"] = clamp_int(v.get("fps", 6), 3, 10)
    v["frames"] = clamp_int(v.get("frames", 20), 10, 25)
    v["resolution_width"] = _nearest_mult_64(clamp_int(v.get("resolution_width", 576), 384, 768))
    v["seed"] = clamp_int(v.get("seed", 42), 0, 2**31 - 1)

    params = v.get("params") or {}
    if preset.startswith("SVD") or preset == "FAILSAFE_LOW_MEM":
        v["params"] = {
            "motion_bucket_id": clamp_int(params.get("motion_bucket_id", 30), 10, 80),
            "steps": clamp_int(params.get("steps", 16), 12, 25),
        }
    else:
        v["params"] = {
            "steps": clamp_int(params.get("steps", 18), 12, 30),
            "cfg": float(clamp(float(params.get("cfg", 3.5)), 1.5, 8.0)),
            "motion_strength": clamp_int(params.get("motion_strength", 35), 10, 80),
            "prompt": str(params.get("prompt", "subtle cinematic movement"))[:240],
        }

    return v


def validate_and_clamp_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    scene = raw.get("scene") or {}
    video = _validate_video(raw.get("video") or {})

    audio = raw.get("audio") or {}
    audio_out = {
        "prompt": str(audio.get("prompt", "soft ambient whoosh"))[:300],
        "duration_s": 3 if int(audio.get("duration_s", video["duration_s"])) == 3 else 5,
        "mix_db": float(clamp(float(audio.get("mix_db", -8.0)), -24.0, 6.0)),
    }

    fallbacks_raw: List[Dict[str, Any]] = raw.get("fallbacks") or []
    fallbacks: List[Dict[str, Any]] = []
    for candidate in fallbacks_raw[:2]:
        fallbacks.append(_validate_video(candidate))

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
        "video": video,
        "audio": audio_out,
        "fallbacks": fallbacks,
    }
