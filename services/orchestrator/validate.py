import copy
import random
from typing import Any, Dict, List

ALLOWED_PRESETS = {
    "WAN22_NATURAL",
    "DETERMINISTIC_ORIGINAL",
    "HUNYUAN15_I2V_720P",
    "HUNYUAN15_I2V_FAST",
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


AUDIO_SOUND_WORDS = {
    "bird",
    "birds",
    "chirp",
    "chirping",
    "insect",
    "insects",
    "buzz",
    "buzzing",
    "leaf",
    "leaves",
    "rustle",
    "rustling",
    "foliage",
    "grass",
    "water",
    "ripples",
    "waves",
    "traffic",
    "train",
    "hum",
    "tone",
    "air",
    "room",
    "ambience",
    "soundscape",
    "gulls",
}

TEXTURE_WORDS = {
    "soft",
    "warm",
    "airy",
    "dreamy",
    "gentle",
    "cinematic",
    "atmospheric",
    "ambient",
    "texture",
    "textures",
}


def _audio_profile_for_preset(preset: str) -> Dict[str, str]:
    profiles = {
        "WAN22_NATURAL": {
            "audio": "quiet natural environmental ambience, no music",
        },
        "DETERMINISTIC_ORIGINAL": {
            "audio": "quiet environmental ambience, no music",
        },
        "HUNYUAN15_I2V_720P": {
            "audio": "realistic environmental ambience, no music",
        },
        "HUNYUAN15_I2V_FAST": {
            "audio": "soft realistic ambience, no music",
        },
    }
    return profiles.get(preset, profiles["WAN22_NATURAL"])


def _merge_prompt_hints(*parts: Any, limit: int = 300) -> str:
    merged: List[str] = []
    for part in parts:
        text = " ".join(str(part or "").split()).strip()
        if text:
            merged.append(text)
    return _limit_text(", ".join(merged), limit)


def _limit_text(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split()).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,")
    return clipped or text[:limit]


def _motion_control_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if bool(params.get("use_original_input_for_video")):
        out["use_original_input_for_video"] = True
    if bool(params.get("preserve_source_aspect")):
        out["preserve_source_aspect"] = True
    final_crop_motion = str(params.get("final_crop_motion", "")).strip().lower()
    if final_crop_motion in {"static", "pan_left_to_right", "pan_right_to_left", "push_in", "pull_out"}:
        out["final_crop_motion"] = final_crop_motion
    pan_direction = str(params.get("pan_direction", "")).strip().lower()
    if pan_direction in {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "auto"}:
        out["pan_direction"] = pan_direction
    for key in ("pan_start", "pan_end"):
        try:
            out[key] = float(clamp(float(params.get(key)), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    try:
        out["zoom_end"] = float(clamp(float(params.get("zoom_end")), 1.02, 2.2))
    except (TypeError, ValueError):
        pass
    for key in ("zoom_focus_x", "zoom_focus_y"):
        try:
            out[key] = float(clamp(float(params.get(key)), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    if str(params.get("zoom_mode", "")).strip().lower() == "enter_frame":
        out["zoom_mode"] = "enter_frame"
    try:
        out["pan_max_span"] = float(clamp(float(params.get("pan_max_span")), 0.05, 0.80))
    except (TypeError, ValueError):
        pass
    output_aspect = str(params.get("output_aspect", "")).strip().lower()
    if output_aspect in {"instagram_reel_9_16", "square_1_1"}:
        out["output_aspect"] = output_aspect
    must_keep_visible = params.get("must_keep_visible")
    if isinstance(must_keep_visible, list):
        out["must_keep_visible"] = [str(item)[:100] for item in must_keep_visible[:10]]
    focus_region = params.get("focus_region")
    if isinstance(focus_region, dict) and isinstance(focus_region.get("box"), dict):
        try:
            box = {
                key: float(clamp(float(focus_region["box"][key]), 0.0, 1.0))
                for key in ("x", "y", "w", "h")
            }
            out["focus_region"] = {
                "label": str(focus_region.get("label", ""))[:100],
                "box": box,
                "source": str(focus_region.get("source", ""))[:100],
            }
        except (KeyError, TypeError, ValueError):
            pass
    required_regions = params.get("required_regions")
    if isinstance(required_regions, list):
        cleaned_regions = []
        for region in required_regions[:10]:
            if not isinstance(region, dict) or not isinstance(region.get("box"), dict):
                continue
            try:
                cleaned_regions.append(
                    {
                        "label": str(region.get("label", ""))[:100],
                        "box": {key: float(clamp(float(region["box"][key]), 0.0, 1.0)) for key in ("x", "y", "w", "h")},
                        "source": str(region.get("source", ""))[:100],
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        out["required_regions"] = cleaned_regions
    visibility = params.get("visibility_validation")
    if isinstance(visibility, dict):
        out["visibility_validation"] = copy.deepcopy(visibility)
    return out


def _compose_audio_prompt(preset: str, scene: Dict[str, Any], audio: Dict[str, Any], current_prompt: str) -> str:
    tags_text = _scene_tags_text(scene)
    profile = _audio_profile_for_preset(preset)
    prompt = " ".join(str(current_prompt or "").split()).strip()
    trusted_soundscape = str(audio.get("prompt_source", "")).strip() == "image2json_soundscape"
    if not prompt:
        prompt = profile["audio"]

    scene_audio = ""
    if any(term in tags_text for term in ["forest", "tree", "trees", "woods", "meadow", "field", "flower", "grass", "hill", "hills", "countryside", "rural", "nature", "greenery", "mountain", "mountains"]):
        scene_audio = "distant occasional birds, soft leaves rustling"
    elif any(term in tags_text for term in ["city", "urban", "street", "skyline", "paris", "eiffel", "avenue", "rooftops"]):
        scene_audio = "distant traffic, soft city hum"
    elif any(term in tags_text for term in ["ocean", "sea", "shore", "waves", "beach"]):
        scene_audio = "small waves, distant gulls"
    elif any(term in tags_text for term in ["lake", "river", "water", "reflection"]):
        scene_audio = "gentle water ripples, distant occasional birds"
    elif any(term in tags_text for term in ["interior", "room", "building", "architecture", "museum", "hall", "indoor"]):
        scene_audio = "soft interior room tone, subtle air"
    elif any(term in tags_text for term in ["orchestra", "concert", "stage", "musicians", "trombone"]):
        scene_audio = "soft room tone, quiet audience, warm brass resonance"

    prompt_lower = prompt.lower()
    prompt_has_sound = any(word in prompt_lower for word in AUDIO_SOUND_WORDS)
    prompt_is_texture_only = any(word in prompt_lower for word in TEXTURE_WORDS) and not prompt_has_sound

    if trusted_soundscape:
        pass
    elif scene_audio and (prompt_is_texture_only or scene_audio not in prompt_lower):
        prompt = f"{scene_audio}, {prompt}"
    if "no music" not in prompt.lower():
        prompt = f"{prompt}, no music"
    return _limit_text(prompt, 96)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        ivalue = low
    return int(clamp(ivalue, low, high))


def _nearest_mult_64(width: int, high: int = 768) -> int:
    # Comfy workflows and many diffusion checkpoints are more stable on /64 widths.
    rounded = int(round(width / 64.0) * 64)
    return clamp_int(rounded, 384, high)


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
        "WAN22_NATURAL": {
            "fps": 20,
            "frames": 97,
            "resolution_width": 768,
            "params": {
                "steps": 20,
                "cfg": 5.0,
                "shift": 8.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "weight_dtype": "default",
                "diffusion_model": "wan2.2_ti2v_5B_fp16.safetensors",
                "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "vae_name": "wan2.2_vae.safetensors",
                "negative_prompt": "flicker, jitter, unstable geometry, inconsistent appearance, scene transition, low quality",
                "tiled_vae": False,
                "preserve_source_aspect": True,
                "use_original_input_for_video": True,
            },
        },
        "DETERMINISTIC_ORIGINAL": {
            "fps": 30,
            "frames": 150,
            "resolution_width": 1080,
            "params": {
                "preserve_source_aspect": True,
                "use_original_input_for_video": True,
                "final_crop_motion": "static",
                "output_aspect": "square_1_1",
            },
        },
        "HUNYUAN15_I2V_720P": {
            "fps": 12,
            "frames": 61,
            "resolution_width": 704,
            "params": {
                "steps": 50,
                "cfg": 6.0,
                "shift": 7.0,
                "weight_dtype": "fp8_e4m3fn",
                "diffusion_model": "hunyuanvideo1.5_720p_i2v_fp16.safetensors",
                "text_encoder_1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "text_encoder_2": "byt5_small_glyphxl_fp16.safetensors",
                "clip_vision_name": "sigclip_vision_patch14_384.safetensors",
                "vae_name": "hunyuanvideo15_vae_fp16.safetensors",
                "tiled_vae": True,
                "preserve_source_aspect": True,
                "use_original_input_for_video": True,
                "negative_prompt": "low quality, blurry, distorted, deformed, text, watermark, flicker, jitter",
            },
        },
        "HUNYUAN15_I2V_FAST": {
            "fps": 6,
            "frames": 30,
            "resolution_width": 704,
            "params": {
                # This machine has the full checkpoint, not the 8/12-step
                # distilled checkpoint. "FAST" reduces temporal length only.
                "steps": 50,
                "cfg": 6.0,
                "shift": 7.0,
                "weight_dtype": "fp8_e4m3fn",
                "diffusion_model": "hunyuanvideo1.5_720p_i2v_fp16.safetensors",
                "text_encoder_1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "text_encoder_2": "byt5_small_glyphxl_fp16.safetensors",
                "clip_vision_name": "sigclip_vision_patch14_384.safetensors",
                "vae_name": "hunyuanvideo15_vae_fp16.safetensors",
                "tiled_vae": True,
                "negative_prompt": "low quality, blurry, distorted, deformed, text, watermark, flicker, jitter",
            },
        },
    }
    if preset not in profiles:
        raise ValueError(f"Unsupported video preset: {preset}")
    profile = profiles[preset]
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
    requested_preset = str(video.get("preset", "WAN22_NATURAL"))
    preset = requested_preset
    if preset not in ALLOWED_PRESETS:
        raise ValueError(f"Unsupported video preset: {requested_preset}")

    # Start from preset defaults, then overlay model output.
    preset_defaults = default_video_for_preset(preset)
    v = copy.deepcopy(preset_defaults)
    default_params = v.get("params") if isinstance(v.get("params"), dict) else {}
    incoming_params = video.get("params") if isinstance(video.get("params"), dict) else {}
    v.update({k: video.get(k, v[k]) for k in ["duration_s", "fps", "frames", "resolution_width", "seed"]})
    v["params"] = {**default_params, **incoming_params}
    v["preset"] = preset
    v["recovery_only"] = bool(video.get("recovery_only", False))

    params = v.get("params") or {}
    v["duration_s"] = 3 if int(video.get("duration_s", params.get("duration_s", v.get("duration_s", 5)))) == 3 else 5
    fps_max = int(preset_defaults["fps"])
    frame_max = int(preset_defaults["frames"])
    resolution_max = int(preset_defaults["resolution_width"])
    requested_fps = video.get("fps", params.get("fps", v.get("fps", 6)))
    try:
        requested_fps_int = int(requested_fps)
    except (TypeError, ValueError):
        requested_fps_int = int(preset_defaults["fps"])
    v["fps"] = clamp_int(requested_fps_int, 3, fps_max)
    requested_frames = video.get("frames", params.get("frames", v.get("frames", 20)))
    try:
        requested_frames_int = int(requested_frames)
    except (TypeError, ValueError):
        requested_frames_int = int(preset_defaults["frames"])
    min_frames_for_duration = v["duration_s"] * v["fps"]
    v["frames"] = clamp_int(max(requested_frames_int, min_frames_for_duration), 10, frame_max)
    v["resolution_width"] = _nearest_mult_64(
        clamp_int(video.get("resolution_width", params.get("resolution_width", v.get("resolution_width", 576))), 384, resolution_max),
        high=resolution_max,
    )
    v["seed"] = _resolve_seed(video, params, v.get("seed"))

    # Clamp preset-specific parameters to safe execution ranges.
    if preset == "DETERMINISTIC_ORIGINAL":
        v["params"] = {
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
            "requested_preset": requested_preset,
            **_motion_control_params(params),
        }
    elif preset.startswith("WAN22_"):
        prompt_hint = _merge_prompt_hints(params.get("prompt", ""), params.get("animation_directions", ""), limit=420)
        v["params"] = {
            "prompt": prompt_hint or "Natural environmental motion. Preserve the original composition and viewpoint.",
            "negative_prompt": _limit_text(params.get("negative_prompt", default_params.get("negative_prompt", "")), 220),
            "steps": clamp_int(params.get("steps", 20), 20, 20),
            "cfg": float(clamp(float(params.get("cfg", 5.0)), 5.0, 5.0)),
            "shift": float(clamp(float(params.get("shift", 8.0)), 8.0, 8.0)),
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "weight_dtype": str(params.get("weight_dtype", "default"))[:40],
            "diffusion_model": str(params.get("diffusion_model", "wan2.2_ti2v_5B_fp16.safetensors"))[:160],
            "text_encoder": str(params.get("text_encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))[:160],
            "vae_name": str(params.get("vae_name", "wan2.2_vae.safetensors"))[:160],
            "tiled_vae": bool(params.get("tiled_vae", default_params.get("tiled_vae", False))),
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
            "requested_preset": requested_preset,
            **_motion_control_params(params),
        }
    elif preset.startswith("HUNYUAN15_"):
        prompt_hint = _merge_prompt_hints(params.get("prompt", ""), params.get("video_prompt", ""), params.get("animation_directions", ""))
        v["params"] = {
            "prompt": prompt_hint
            or "cinematic image-to-video motion, preserve original subject identity and composition, natural camera movement",
            "negative_prompt": str(
                params.get(
                    "negative_prompt",
                    "low quality, blurry, distorted, deformed, text, watermark, flicker, jitter",
                )
            )[:240],
            "steps": clamp_int(params.get("steps", default_params.get("steps", 12)), 8, int(default_params.get("steps", 12))),
            "cfg": float(clamp(float(params.get("cfg", default_params.get("cfg", 5.5))), 1.0, float(default_params.get("cfg", 5.5)))),
            "shift": float(clamp(float(params.get("shift", 7.0)), 1.0, 12.0)),
            "weight_dtype": str(params.get("weight_dtype", "default"))[:40],
            "diffusion_model": str(params.get("diffusion_model", "hunyuanvideo1.5_720p_i2v_fp16.safetensors"))[:160],
            "text_encoder_1": str(params.get("text_encoder_1", "qwen_2.5_vl_7b_fp8_scaled.safetensors"))[:160],
            "text_encoder_2": str(params.get("text_encoder_2", "byt5_small_glyphxl_fp16.safetensors"))[:160],
            "clip_vision_name": str(params.get("clip_vision_name", "sigclip_vision_patch14_384.safetensors"))[:160],
            "vae_name": str(params.get("vae_name", "hunyuanvideo15_vae_fp16.safetensors"))[:160],
            "tiled_vae": bool(params.get("tiled_vae", default_params.get("tiled_vae", True))),
            "duration_s": v["duration_s"],
            "fps": v["fps"],
            "frames": v["frames"],
            "resolution_width": v["resolution_width"],
            "seed": v["seed"],
            "requested_preset": requested_preset,
            **_motion_control_params(params),
        }
    else:
        raise ValueError(f"Unsupported video preset: {preset}")

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
    target_aspect = str(framing.get("target_aspect", "instagram_reel_9_16"))
    if target_aspect not in {"instagram_reel_9_16", "square_1_1"}:
        target_aspect = "instagram_reel_9_16"
    framing_out = {
        "target_aspect": target_aspect,
        "crop_anchor": anchor,
    }

    audio = raw.get("audio") or {}
    if not isinstance(audio, dict):
        audio = {}
    audio_out = {
        "prompt": " ".join(str(audio.get("prompt", "soft ambience")).split())[:220],
        "duration_s": 3 if int(audio.get("duration_s", video["duration_s"])) == 3 else 5,
        # Ambience should sit behind the image rather than dominate it.
        "mix_db": float(clamp(float(audio.get("mix_db", -6.0)), -18.0, 0.0)),
    }
    for key in (
        "prompt_source",
        "soundscape_confidence",
        "soundscape_environment_type",
        "soundscape_proximity",
        "soundscape_reasoning",
        "avoid_sounds",
    ):
        if key in audio:
            audio_out[key] = copy.deepcopy(audio[key])

    fallbacks_raw = raw.get("fallbacks") or []
    if not isinstance(fallbacks_raw, list):
        fallbacks_raw = []
    fallbacks_raw = [c for c in fallbacks_raw if isinstance(c, dict)]
    fallbacks: List[Dict[str, Any]] = []
    for candidate in fallbacks_raw[:2]:
        fallbacks.append(_validate_video(candidate))

    # Candidate 1 stays in the selected model family with an independent seed;
    # candidate 2 is deterministic recovery and is never rendered routinely.
    while len(fallbacks) < 2:
        if len(fallbacks) == 0:
            candidate = default_video_for_preset(video["preset"])
            candidate["seed"] = _resolve_seed({}, {}, None)
            candidate["params"]["seed"] = candidate["seed"]
            fallbacks.append(candidate)
        else:
            recovery = default_video_for_preset("DETERMINISTIC_ORIGINAL")
            recovery["recovery_only"] = True
            fallbacks.append(recovery)

    scene_out = {
        "tags": [str(t)[:50] for t in (scene.get("tags") or [])[:8]],
        "has_people": bool(scene.get("has_people", False)),
        "confidence": float(clamp(float(scene.get("confidence", 0.5)), 0.0, 1.0)),
    }
    if str(video["preset"]).startswith("HUNYUAN15_"):
        scene_text = _scene_terms(scene_out)
        current_prompt = str(video["params"].get("prompt", "")).strip()
        if scene_text != "original scene" and scene_text not in current_prompt.lower():
            video["params"]["prompt"] = _limit_text(f"{current_prompt}, {scene_text}".strip(" ,"), 300)
    audio_out["prompt"] = _compose_audio_prompt(video["preset"], scene_out, audio_out, audio_out["prompt"])
    for fb in fallbacks:
        if str(fb["preset"]).startswith("HUNYUAN15_"):
            scene_text = _scene_terms(scene_out)
            current_prompt = str(fb["params"].get("prompt", "")).strip()
            if scene_text != "original scene" and scene_text not in current_prompt.lower():
                fb["params"]["prompt"] = _limit_text(f"{current_prompt}, {scene_text}".strip(" ,"), 300)

    return {
        "scene": scene_out,
        "framing": framing_out,
        "video": video,
        "audio": audio_out,
        "fallbacks": fallbacks,
    }
