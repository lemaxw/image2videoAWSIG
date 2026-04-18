import copy
import random
from typing import Any, Dict, List

ALLOWED_PRESETS = {
    "SVD_SUBTLE",
    "SVD_STRONG",
    "ANIMATEDIFF_GRASS_WIND",
    "ANIMATEDIFF_CITY_PULSE",
    "ANIMATEDIFF_LOW_MEM",
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


def _scene_terms(scene: Dict[str, Any]) -> str:
    tags = [str(t).strip().lower().replace("_", " ") for t in (scene.get("tags") or []) if str(t).strip()]
    terms = []
    for tag in tags[:4]:
        if tag not in terms:
            terms.append(tag)
    if bool(scene.get("has_people")) and "people" not in terms and "person" not in terms:
        terms.append("people")
    return ", ".join(terms[:4]) if terms else "original scene"


def _scene_tags_text(scene: Dict[str, Any]) -> str:
    return " ".join(str(t).strip().lower().replace("_", " ") for t in (scene.get("tags") or []))


def _anime_profile_for_preset(preset: str) -> Dict[str, str]:
    profiles = {
        "SVD_SUBTLE": {
            "style": "soft cel shading, clean lineart, gentle anime key visual",
            "audio": "soft, airy, gentle cinematic ambience",
        },
        "SVD_STRONG": {
            "style": "cinematic cel shading, stronger contrast, polished anime key visual",
            "audio": "atmospheric, warm, cinematic ambience",
        },
        "ANIMATEDIFF_GRASS_WIND": {
            "style": "soft cel shading, natural outdoor anime illustration, detailed hair movement cues",
            "audio": "airy natural ambience, soft wind texture, cinematic ambience",
        },
        "ANIMATEDIFF_CITY_PULSE": {
            "style": "urban anime illustration, cinematic cel shading, graphic contrast",
            "audio": "urban ambient texture, soft pulse, cinematic ambience",
        },
        "ANIMATEDIFF_LOW_MEM": {
            "style": "simple anime illustration, clean lineart, low-detail stable cel shading",
            "audio": "soft ambient texture, gentle atmosphere, cinematic ambience",
        },
        "FAILSAFE_LOW_MEM": {
            "style": "simple anime illustration, clean lineart, low-detail stable cel shading",
            "audio": "soft ambient texture, gentle atmosphere, cinematic ambience",
        },
    }
    return profiles.get(preset, profiles["FAILSAFE_LOW_MEM"])


def _stylization_cues(scene: Dict[str, Any], preset: str, hint: str) -> str:
    tags = _scene_tags_text(scene)
    cues: List[str] = []
    text = f"{tags} {hint.lower()} {preset.lower()}"
    if any(term in text for term in ["car", "cars", "traffic", "street", "road", "city", "urban"]):
        cues.append("headlights and taillights become soft streaks of anime light")
    if any(term in text for term in ["water", "sea", "ocean", "lake", "river"]):
        cues.append("water becomes luminous with colorful reflections")
    if any(term in text for term in ["sky", "cloud", "clouds"]):
        cues.append("sky and clouds become more colorful, layered, and dramatic")
    if any(term in text for term in ["sunset", "sunrise", "golden hour"]):
        cues.append("sunlight becomes painterly golden anime glow")
    if any(term in text for term in ["night", "neon", "lights", "city pulse"]):
        cues.append("night lighting becomes saturated cinematic anime glow")
    if any(term in text for term in ["tree", "trees", "grass", "field", "forest", "wind"]):
        cues.append("foliage simplifies into clean anime shapes with gentle motion-friendly detail")
    if bool(scene.get("has_people")):
        cues.append("faces stay readable and expressive")
    return ", ".join(cues[:4])


def _compose_anime_prompt(preset: str, scene: Dict[str, Any], current_prompt: str, hint: str = "") -> str:
    scene_text = _scene_terms(scene)
    profile = _anime_profile_for_preset(preset)
    base = "anime illustration, preserve original composition, preserve camera angle, preserve subject layout"
    people_guard = ", preserve face placement, preserve body framing" if bool(scene.get("has_people")) else ""
    detail = current_prompt.strip() if current_prompt and "preserve original composition" not in current_prompt.lower() else profile["style"]
    hint_text = hint.strip()
    stylization = _stylization_cues(scene, preset, hint_text)
    combined = ", ".join(part for part in [base, detail, hint_text, stylization, scene_text] if part)
    return f"{combined}{people_guard}"[:240]


def _merge_prompt_hints(*parts: Any) -> str:
    merged: List[str] = []
    for part in parts:
        text = " ".join(str(part or "").split()).strip()
        if text:
            merged.append(text)
    return ", ".join(merged)[:160]


def _motion_bucket_for_scene(scene: Dict[str, Any], preset: str, current_value: int) -> int:
    text = _scene_tags_text(scene)
    value = current_value
    if any(term in text for term in ["car", "cars", "traffic", "highway", "streetlights", "street lights", "road"]):
        value = max(value, 28 if preset in {"SVD_STRONG", "ANIMATEDIFF_CITY_PULSE"} else 24)
    if any(term in text for term in ["water", "sea", "ocean", "lake", "river", "cloud", "clouds", "sky"]):
        value = max(value, 20)
    return clamp_int(value, 10, 80)


def _compose_audio_prompt(preset: str, scene: Dict[str, Any], current_prompt: str) -> str:
    scene_text = _scene_terms(scene)
    profile = _anime_profile_for_preset(preset)
    prompt = " ".join(str(current_prompt or "").split()).strip()
    if not prompt:
        prompt = profile["audio"]
    if scene_text != "original scene" and scene_text not in prompt.lower():
        prompt = f"{scene_text}, {prompt}"
    return prompt[:96]


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


def _resolve_seed(video: Dict[str, Any], params: Dict[str, Any], default_seed: Any = None) -> int:
    for candidate in (video.get("seed"), params.get("seed"), default_seed):
        if candidate is None:
            continue
        try:
            return clamp_int(candidate, 0, 2**31 - 1)
        except Exception:
            continue
    return random.SystemRandom().randint(0, 2**31 - 1)


def default_video_for_preset(preset: str) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {
        "SVD_SUBTLE": {
            "fps": 5,
            "frames": 16,
            "resolution_width": 576,
            "params": {
                "motion_bucket_id": 18,
                "steps": 18,
                "anime_ckpt_name": "meinamix_v11.safetensors",
                "anime_steps": 20,
                "anime_cfg": 5.0,
                "anime_denoise": 0.30,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, soft cel shading, preserve original composition, preserve subject layout, low-motion anime still",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, noisy background",
            },
        },
        "SVD_STRONG": {
            "fps": 5,
            "frames": 16,
            "resolution_width": 576,
            "params": {
                "motion_bucket_id": 24,
                "steps": 20,
                "anime_ckpt_name": "counterfeit_v30.safetensors",
                "anime_steps": 20,
                "anime_cfg": 5.3,
                "anime_denoise": 0.32,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, cinematic cel shading, preserve original composition, preserve pose, preserve camera angle",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, oversaturated",
            },
        },
        "ANIMATEDIFF_GRASS_WIND": {
            "fps": 5,
            "frames": 16,
            "resolution_width": 576,
            "params": {
                "anime_ckpt_name": "meinamix_v11.safetensors",
                "anime_steps": 20,
                "anime_cfg": 5.0,
                "anime_denoise": 0.32,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, soft cel shading, detailed hair, preserve original composition, preserve subject layout, subtle natural wind mood",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, noisy background",
                "steps": 18,
                "motion_bucket_id": 18,
            },
        },
        "ANIMATEDIFF_CITY_PULSE": {
            "fps": 5,
            "frames": 16,
            "resolution_width": 576,
            "params": {
                "anime_ckpt_name": "counterfeit_v30.safetensors",
                "anime_steps": 20,
                "anime_cfg": 5.4,
                "anime_denoise": 0.30,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, cinematic cel shading, urban key visual, preserve original composition, preserve pose, preserve camera angle",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, oversaturated",
                "steps": 18,
                "motion_bucket_id": 22,
            },
        },
        "ANIMATEDIFF_LOW_MEM": {
            "fps": 4,
            "frames": 12,
            "resolution_width": 512,
            "params": {
                "anime_ckpt_name": "anything-v5-prt.safetensors",
                "anime_steps": 18,
                "anime_cfg": 4.8,
                "anime_denoise": 0.28,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, soft cel shading, preserve original composition, preserve subject placement, low motion-friendly anime still",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, messy lineart",
                "steps": 16,
                "motion_bucket_id": 16,
            },
        },
        "FAILSAFE_LOW_MEM": {
            "fps": 4,
            "frames": 12,
            "resolution_width": 512,
            "params": {
                "motion_bucket_id": 14,
                "steps": 14,
                "anime_ckpt_name": "anything-v5-prt.safetensors",
                "anime_steps": 16,
                "anime_cfg": 4.6,
                "anime_denoise": 0.26,
                "anime_sampler_name": "dpmpp_2m",
                "anime_scheduler": "karras",
                "anime_prompt": "anime illustration, clean lineart, soft cel shading, preserve original composition, preserve subject placement, simple anime still",
                "anime_negative_prompt": "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark, messy lineart",
            },
        },
    }
    profile = profiles.get(preset, profiles["FAILSAFE_LOW_MEM"])
    return {
        "preset": preset,
        "duration_s": 5,
        "fps": profile["fps"],
        "frames": profile["frames"],
        "resolution_width": profile["resolution_width"],
        "seed": None,
        "params": dict(profile["params"]),
    }


def _validate_video(video: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(video, dict):
        video = {}
    requested_preset = str(video.get("preset", "FAILSAFE_LOW_MEM"))
    preset = requested_preset
    if preset not in ALLOWED_PRESETS:
        preset = "FAILSAFE_LOW_MEM"

    # Start from preset defaults, then overlay model output.
    v = copy.deepcopy(default_video_for_preset(preset))
    v.update({k: video.get(k, v[k]) for k in ["duration_s", "fps", "frames", "resolution_width", "seed", "params"]})
    v["preset"] = preset

    params = v.get("params") or {}
    v["duration_s"] = 3 if int(video.get("duration_s", params.get("duration_s", v.get("duration_s", 5)))) == 3 else 5
    v["fps"] = clamp_int(video.get("fps", params.get("fps", v.get("fps", 6))), 3, 10)
    requested_frames = video.get("frames", params.get("frames", v.get("frames", 20)))
    min_frames_for_duration = v["duration_s"] * v["fps"]
    v["frames"] = clamp_int(max(int(requested_frames), min_frames_for_duration), 10, 35)
    v["resolution_width"] = _nearest_mult_64(
        clamp_int(video.get("resolution_width", params.get("resolution_width", v.get("resolution_width", 576))), 384, 768)
    )
    v["seed"] = _resolve_seed(video, params, v.get("seed"))

    # Clamp preset-specific parameters to safe execution ranges.
    if preset.startswith("SVD") or preset == "FAILSAFE_LOW_MEM":
        prompt_hint = _merge_prompt_hints(params.get("anime_prompt_hint", ""), params.get("animation_directions", ""))
        motion_strength = clamp_int(params.get("motion_strength", 35), 10, 80)
        mapped_motion = clamp_int(int(round(motion_strength * 0.75)), 10, 80)
        v["params"] = {
            "motion_bucket_id": clamp_int(params.get("motion_bucket_id", mapped_motion), 10, 80),
            "steps": clamp_int(params.get("steps", 16), 12, 25),
            "anime_ckpt_name": str(params.get("anime_ckpt_name", "meinamix_v11.safetensors"))[:160],
            "anime_steps": clamp_int(params.get("anime_steps", 20), 12, 28),
            "anime_cfg": float(clamp(float(params.get("anime_cfg", 5.0)), 2.0, 7.0)),
            "anime_denoise": float(clamp(float(params.get("anime_denoise", 0.30)), 0.15, 0.45)),
            "anime_sampler_name": str(params.get("anime_sampler_name", "dpmpp_2m"))[:40],
            "anime_scheduler": str(params.get("anime_scheduler", "karras"))[:40],
            "anime_prompt": str(
                params.get(
                    "anime_prompt",
                    "anime illustration, clean lineart, soft cel shading, preserve original composition",
                )
            )[:240],
            "anime_prompt_hint": prompt_hint,
            "anime_negative_prompt": str(
                params.get(
                    "anime_negative_prompt",
                    "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark",
                )
            )[:240],
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
            "requested_preset": requested_preset,
        }
    else:
        prompt_hint = _merge_prompt_hints(params.get("anime_prompt_hint", ""), params.get("animation_directions", ""))
        v["params"] = {
            "steps": clamp_int(params.get("steps", 18), 12, 24),
            "motion_bucket_id": clamp_int(params.get("motion_bucket_id", 18), 10, 32),
            "anime_ckpt_name": str(params.get("anime_ckpt_name", "meinamix_v11.safetensors"))[:160],
            "anime_steps": clamp_int(params.get("anime_steps", 20), 12, 28),
            "anime_cfg": float(clamp(float(params.get("anime_cfg", 5.2)), 2.0, 7.0)),
            "anime_denoise": float(clamp(float(params.get("anime_denoise", 0.32)), 0.15, 0.45)),
            "anime_sampler_name": str(params.get("anime_sampler_name", "dpmpp_2m"))[:40],
            "anime_scheduler": str(params.get("anime_scheduler", "karras"))[:40],
            "anime_prompt": str(
                params.get(
                    "anime_prompt",
                    "anime illustration, clean lineart, soft cel shading, preserve original composition",
                )
            )[:240],
            "anime_prompt_hint": prompt_hint,
            "anime_negative_prompt": str(
                params.get(
                    "anime_negative_prompt",
                    "photorealistic, realistic skin, 3d, blurry, lowres, deformed hands, bad anatomy, text, watermark",
                )
            )[:240],
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
        # Keep audio clearly audible in final mux by default.
        "mix_db": float(clamp(float(audio.get("mix_db", -2.0)), -3.0, 6.0)),
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

    scene_out = {
        "tags": [str(t)[:50] for t in (scene.get("tags") or [])[:8]],
        "has_people": bool(scene.get("has_people", False)),
        "confidence": float(clamp(float(scene.get("confidence", 0.5)), 0.0, 1.0)),
    }
    video["params"]["anime_prompt"] = _compose_anime_prompt(
        video["preset"],
        scene_out,
        str(video["params"].get("anime_prompt", "")),
        str(video["params"].get("anime_prompt_hint", "")),
    )
    if "motion_bucket_id" in video["params"]:
        video["params"]["motion_bucket_id"] = _motion_bucket_for_scene(
            scene_out,
            video["preset"],
            int(video["params"].get("motion_bucket_id", 18)),
        )
    audio_out["prompt"] = _compose_audio_prompt(video["preset"], scene_out, audio_out["prompt"])
    for fb in fallbacks:
        fb["params"]["anime_prompt"] = _compose_anime_prompt(
            fb["preset"],
            scene_out,
            str(fb["params"].get("anime_prompt", "")),
            str(fb["params"].get("anime_prompt_hint", "")),
        )
        if "motion_bucket_id" in fb["params"]:
            fb["params"]["motion_bucket_id"] = _motion_bucket_for_scene(
                scene_out,
                fb["preset"],
                int(fb["params"].get("motion_bucket_id", 18)),
            )

    return {
        "scene": scene_out,
        "framing": framing_out,
        "video": video,
        "audio": audio_out,
        "fallbacks": fallbacks,
    }
