import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from services.orchestrator.validate import validate_and_clamp_decision

DECISION_SCHEMA: Dict[str, Any] = {
    # Hard contract enforced through Responses API structured outputs.
    "type": "object",
    "required": ["scene", "framing", "video", "audio", "fallbacks"],
    "additionalProperties": False,
    "properties": {
        "scene": {
            "type": "object",
            "required": ["tags", "has_people", "confidence"],
            "additionalProperties": False,
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "has_people": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "framing": {
            "type": "object",
            "required": ["target_aspect", "crop_anchor"],
            "additionalProperties": False,
            "properties": {
                "target_aspect": {"type": "string", "enum": ["instagram_reel_9_16"]},
                "crop_anchor": {
                    "type": "string",
                    "enum": [
                        "left_top",
                        "center_top",
                        "right_top",
                        "left_center",
                        "center_center",
                        "right_center",
                        "left_bottom",
                        "center_bottom",
                        "right_bottom"
                    ]
                }
            }
        },
        "video": {
            "$ref": "#/$defs/videoObj"
        },
        "audio": {
            "type": "object",
            "required": ["prompt", "duration_s", "mix_db"],
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string", "maxLength": 96},
                "duration_s": {"type": "integer", "enum": [3, 5]},
                "mix_db": {"type": "number"}
            }
        },
        "fallbacks": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"$ref": "#/$defs/videoObj"}
        }
    },
    "$defs": {
        "videoObj": {
            "type": "object",
            "required": ["preset", "duration_s", "fps", "frames", "resolution_width", "seed", "params"],
            "additionalProperties": False,
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": [
                        "HUNYUAN15_I2V_720P",
                        "HUNYUAN15_I2V_FAST",
                        "SVD_SUBTLE",
                        "SVD_STRONG",
                        "ANIMATEDIFF_GRASS_WIND",
                        "ANIMATEDIFF_CITY_PULSE"
                    ]
                },
                "duration_s": {"type": "integer", "enum": [3, 5]},
                "fps": {"type": "integer"},
                "frames": {"type": "integer"},
                "resolution_width": {"type": "integer"},
                "seed": {"type": "integer"},
                "params": {"type": "object"}
            }
        }
    }
}

SYSTEM_PROMPT = """You are selecting the best image-to-video treatment for one image.

Goal:
choose the preset and motion style that will create the most visually appealing final video.

Two backend families exist:

1. Hunyuan presets:
direct cinematic image-to-video realism

2. SVD / AnimateDiff presets:
anime-style redraw first, then animation

------------------------------------------------
PRESETS
------------------------------------------------

HUNYUAN15_I2V_720P
- premium cinematic realism
- best quality and temporal consistency
- ideal for people, travel, landscapes, architecture, products, cinematic shots

HUNYUAN15_I2V_FAST
- faster/lighter Hunyuan fallback
- use for complex, crowded, detailed, or expensive scenes

ANIMATEDIFF_GRASS_WIND
- anime outdoor atmosphere
- foliage, grass, forests, clouds, dreamy scenery, gentle wind

ANIMATEDIFF_CITY_PULSE
- anime urban atmosphere
- neon, rain, nightlife, reflections, city energy

SVD_STRONG
- dramatic cinematic reinterpretation
- stronger atmospheric motion and visual exaggeration
- ideal for sunsets, oceans, fog, glowing reflections, dramatic skies, haze, particles, fantasy mood, and emotionally cinematic environmental motion

SVD_SUBTLE
- emergency stability fallback
- use only when stronger cinematic motion is likely to create artifacts or unstable geometry
- primarily for fragile close-up portraits, logos, text-heavy images, dense architecture, low-detail noisy scenes, or highly detail-sensitive compositions

------------------------------------------------
SELECTION ORDER
------------------------------------------------

Prefer presets in this order when appropriate:

1. HUNYUAN15_I2V_720P
2. HUNYUAN15_I2V_FAST
3. ANIMATEDIFF_GRASS_WIND
4. ANIMATEDIFF_CITY_PULSE
5. SVD_STRONG
6. SVD_SUBTLE

------------------------------------------------
SELECTION RULES
------------------------------------------------

- prefer Hunyuan for realistic cinematic motion
- use AnimateDiff presets when anime reinterpretation improves the final video aesthetically
- use SVD_STRONG for atmospheric cinematic reinterpretation
- use SVD_STRONG mainly when the scene benefits from dramatic atmospheric exaggeration, not merely because the scene is cinematic
- use SVD_SUBTLE only when stronger cinematic motion is likely to fail or create visible artifacts
- preserve faces and composition unless stylization clearly improves the final result
- avoid chaotic or aggressive motion

Special rules:
- night city scenes can use atmospheric cinematic motion, but avoid aggressive zoom-in or push-in camera movement
- for night cities prefer stable framing, slow lateral drift, reflections, neon flicker, rain shimmer, and atmospheric glow
- do not default night city photography to SVD_SUBTLE
- for cloud-drift animation prefer HUNYUAN or AnimateDiff presets
- avoid SVD presets when cloud movement is the primary motion opportunity unless dramatic reinterpretation is desired

------------------------------------------------
MOTION GUIDANCE
------------------------------------------------

video.params.prompt or video.params.video_prompt should describe:
- motion
- atmosphere
- cinematic camera feel

Good motion examples:
- slow cinematic drift
- gentle wind through grass
- drifting clouds
- soft neon reflections
- subtle ocean movement
- atmospheric fog movement
- gentle rain shimmer
- stable cinematic hold
- slow lateral camera drift

Avoid:
- aggressive zoom-ins
- fast camera shake
- chaotic motion
- rapid scene transformations

anime_prompt_hint:
short visual enhancement hint for stylized animation

Examples:
- soft anime neon glow
- dramatic anime clouds
- dreamy watercolor foliage
- warm glowing sand tones
- atmospheric rain reflections

Do not change the core composition of the image.

Wide composition / pan rule:
- If important subjects are far apart and a single vertical crop would lose the visual story, set video.params.use_original_input_for_video=true.
- Use this for wide moon + skyline/building scenes, panoramas, broad landscapes, or subject/context pairs separated across the image.
- When using the original image, describe a cinematic result with slow lateral camera travel such as left-to-right or right-to-left drift.
- If a 9:16 pan would be too zoomed or would cut away the story, set video.params.output_aspect="square_1_1".
- Prefer square_1_1 for wide moon + building/city scenes where square framing preserves both the moon and architecture better than vertical Reel crop.
- Use video.params.output_aspect="instagram_reel_9_16" only when the vertical crop remains visually strong.
- For square moon + building/city scenes, keep the moon comfortably inside the frame, not touching the edge; pan_start is usually around 0.54-0.60 for left-to-right motion.
- You may include video.params.pan_direction as left_to_right, right_to_left, top_to_bottom, bottom_to_top, or auto.
- You may include video.params.pan_start and video.params.pan_end as floats from 0.0 to 1.0, where 0.0 is the far left/top of the source and 1.0 is the far right/bottom.
- Choose a partial pan window that fits the duration; do not sweep the entire image unless the full journey is essential.
- For a 5-second video, keep abs(pan_end - pan_start) <= 0.18. Prefer 0.08-0.16 for slow, smooth motion.
- If important subjects are farther apart than this limit, choose the best starting composition and drift only slightly toward the secondary subject.
- You may include video.params.pan_max_span, but it must be between 0.05 and 0.25.
- Pick the best starting point so the first frame is already visually useful, then drift slowly toward the secondary subject.
- The pipeline will export a still image from the final cropped video; choose output_aspect and pan window so the video and still both make sense.
- Prefer this mainly with Hunyuan presets; cropped input is still better for portraits, single subjects, text, logos, symmetry, and detail-sensitive architecture.

------------------------------------------------
FRAMING
------------------------------------------------

target_aspect must always be:
instagram_reel_9_16

crop_anchor values:
left_top
center_top
right_top
left_center
center_center
right_center
left_bottom
center_bottom
right_bottom

Keep people fully visible after crop.

Rules:
- keep people fully visible after crop
- avoid cutting heads or bodies
- if uncertain, use center_center

------------------------------------------------
SCENE
------------------------------------------------

Always return:
scene.tags
scene.has_people
scene.confidence

scene.tags:
5-8 short keywords only

Examples:
ocean
sunset
city
night
mountains
forest
fog
beach

scene.confidence:
float between 0 and 1

------------------------------------------------
AUDIO
------------------------------------------------

Return one short ambient audio prompt under 96 characters.

Format:
"<environment>, <texture>, cinematic ambience"

Examples:
quiet city night, soft neon hum, cinematic ambience
desert wind, airy sand movement, cinematic ambience
mountain valley, gentle wind and birds, cinematic ambience

Prefer:
soft
warm
airy
dreamy
gentle
atmospheric
ambient

Avoid:
loud
chaotic
aggressive

------------------------------------------------
OUTPUT RULES
------------------------------------------------

Return JSON only.

Top-level keys:
scene
framing
video
audio
fallbacks

fallbacks:
exactly two video objects

Use only valid preset enum names.

When uncertain:
prefer the most visually appealing final video while protecting composition and identity.
"""


def _fallback_decision_payload() -> Dict[str, Any]:
    return {
        "scene": {
            "tags": ["fallback", "unknown_scene"],
            "has_people": False,
            "confidence": 0.3,
        },
        "framing": {
            "target_aspect": "instagram_reel_9_16",
            "crop_anchor": "center_center",
        },
        "video": {
            "preset": "HUNYUAN15_I2V_FAST",
            "duration_s": 5,
            "fps": 6,
            "frames": 30,
            "resolution_width": 704,
            "params": {"steps": 12, "prompt": "gentle cinematic motion, preserve original composition"},
        },
        "audio": {
            "prompt": "soft ambient city",
            "duration_s": 5,
            "mix_db": -10.0,
        },
        "fallbacks": [
            {
                "preset": "SVD_SUBTLE",
                "duration_s": 5,
                "fps": 6,
                "frames": 20,
                "resolution_width": 576,
                "params": {"motion_bucket_id": 22, "steps": 14},
            },
            {
                "preset": "ANIMATEDIFF_GRASS_WIND",
                "duration_s": 5,
                "fps": 8,
                "frames": 32,
                "resolution_width": 768,
                "params": {"steps": 20, "prompt": "gentle anime-style motion, preserve original composition"},
            },
        ],
    }


def _image_to_data_uri(image_path: Path) -> str:
    # Inline image transport avoids external URL dependencies in private VPC flows.
    mime = "image/png"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    if image_path.suffix.lower() == ".webp":
        mime = "image/webp"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extract_text(response: Any) -> str:
    # SDK versions can expose output as `output_text` or nested content blocks.
    text = getattr(response, "output_text", "")
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") in {"output_text", "text"}:
                chunks.append(getattr(content, "text", ""))
    return "\n".join(chunks)


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    return {
        "input_tokens": int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _extract_chat_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _normalize_prefixed_object(container: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    obj = container.get(prefix)
    if not isinstance(obj, dict):
        obj = {}
    else:
        obj = dict(obj)

    dotted_prefix = f"{prefix}."
    normalized = {k: v for k, v in obj.items() if not (isinstance(k, str) and k.startswith(dotted_prefix))}
    for key, value in obj.items():
        if isinstance(key, str) and key.startswith(dotted_prefix):
            normalized.setdefault(key[len(dotted_prefix):], value)
    for key, value in list(container.items()):
        if isinstance(key, str) and key.startswith(dotted_prefix):
            normalized.setdefault(key[len(dotted_prefix):], value)
    return normalized


def _normalize_fallback_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    if isinstance(normalized.get("video"), dict):
        inner_video = dict(normalized["video"])
        inner_video.update(_normalize_prefixed_object(normalized, "video"))
        normalized["video"] = inner_video
    elif any(isinstance(key, str) and key.startswith("video.") for key in normalized):
        normalized = _normalize_prefixed_object(normalized, "video")
    return normalized


def _coerce_decision_shape(parsed: Dict[str, Any]) -> Dict[str, Any]:
    # Some chat-completions responses return a shorthand shape:
    # {"preset":"...", "fallbacks":["...","..."], "audio": {...}, "framing": {...}}
    # Normalize that into the strict contract expected downstream.
    if not isinstance(parsed, dict):
        return _fallback_decision_payload()
    nested = parsed.get("decision")
    if isinstance(nested, dict):
        parsed = nested
    parsed = dict(parsed)
    parsed["video"] = _normalize_prefixed_object(parsed, "video")
    parsed["audio"] = _normalize_prefixed_object(parsed, "audio")
    parsed["framing"] = _normalize_prefixed_object(parsed, "framing")
    parsed["scene"] = _normalize_prefixed_object(parsed, "scene")
    if isinstance(parsed.get("fallbacks"), list):
        parsed["fallbacks"] = [_normalize_fallback_item(item) for item in parsed["fallbacks"]]
    if "video" in parsed and "scene" in parsed and "audio" in parsed and "fallbacks" in parsed:
        return parsed

    base = _fallback_decision_payload()
    video_in = parsed.get("video")
    if not isinstance(video_in, dict):
        video_in = {}

    scene_in = parsed.get("scene")
    if not isinstance(scene_in, dict):
        scene_in = video_in.get("scene") if isinstance(video_in.get("scene"), dict) else base["scene"]

    framing_in = parsed.get("framing")
    if not isinstance(framing_in, dict):
        framing_in = video_in.get("framing") if isinstance(video_in.get("framing"), dict) else base["framing"]

    audio_in = parsed.get("audio")
    if not isinstance(audio_in, dict):
        audio_in = video_in.get("audio") if isinstance(video_in.get("audio"), dict) else base["audio"]

    out: Dict[str, Any] = {
        "scene": scene_in,
        "framing": framing_in,
        "audio": audio_in,
        "video": dict(base["video"]),
        "fallbacks": [dict(base["fallbacks"][0]), dict(base["fallbacks"][1])],
    }

    # Accept both shorthand top-level keys and nested keys under `video`.
    preset = parsed.get("preset") or parsed.get("preset_primary") or video_in.get("preset")
    if isinstance(preset, str):
        out["video"]["preset"] = preset
    for key in ["duration_s", "fps", "frames", "resolution_width", "seed", "params"]:
        if key in video_in:
            out["video"][key] = video_in[key]

    fb = parsed.get("fallbacks")
    if not isinstance(fb, list):
        fb = parsed.get("preset_fallbacks")
    if not isinstance(fb, list) and isinstance(video_in.get("fallbacks"), list):
        fb = video_in.get("fallbacks")
    if isinstance(fb, list):
        normalized = []
        for i, item in enumerate(fb[:2]):
            if isinstance(item, str):
                cfg = dict(base["fallbacks"][i] if i < len(base["fallbacks"]) else base["fallbacks"][-1])
                cfg["preset"] = item
                normalized.append(cfg)
            elif isinstance(item, dict):
                inner_video = item.get("video") if isinstance(item.get("video"), dict) else item
                if isinstance(inner_video, dict):
                    cfg = dict(base["fallbacks"][i] if i < len(base["fallbacks"]) else base["fallbacks"][-1])
                    cfg.update(inner_video)
                    normalized.append(cfg)
        if normalized:
            while len(normalized) < 2:
                normalized.append(dict(base["fallbacks"][len(normalized)]))
            out["fallbacks"] = normalized[:2]
    return out


def decide_for_image_detailed(image_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = metadata or {}
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    usage_acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    io_payload_base: Dict[str, Any] = {
        "request": {
            "api_mode_preference": "responses_json_schema_then_chat_fallback",
            "model": model,
            "image": {"path": str(image_path)},
            "metadata": metadata,
        },
        "attempts": [],
    }

    if not api_key:
        return {
            "decision": validate_and_clamp_decision(_fallback_decision_payload()),
            "openai": {
                "model": model,
                "attempts": 0,
                "usage": usage_acc,
                "status": "skipped_no_api_key",
                "io": io_payload_base,
            },
        }

    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        return {
            "decision": validate_and_clamp_decision(_fallback_decision_payload()),
            "openai": {
                "model": model,
                "attempts": 0,
                "usage": usage_acc,
                "status": "fallback_client_init_error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "io": io_payload_base,
            },
        }

    user_text = {
        "type": "input_text",
        "text": (
            "Return decision JSON for image-to-video rendering. "
            f"Metadata: {json.dumps(metadata, ensure_ascii=True)}. "
            "Enforce schema constraints, include exactly two fallbacks, and set framing.target_aspect=instagram_reel_9_16. "
            "Use only preset enum values exactly as provided. "
            "Mandatory: include scene.tags, scene.has_people, scene.confidence. "
            "Add short video.params.prompt or video.params.animation_directions when useful to describe motion while preserving composition. "
            "For wide scenes where vertical pan/crop would be too zoomed, set video.params.output_aspect=square_1_1. "
            "Choose crop_anchor to keep people uncut in frame. "
            "Top-level keys must be exactly: scene, framing, video, audio, fallbacks (do not nest scene/framing inside video). "
            "Use nested object fields only: video.preset means the `preset` field inside the `video` object, "
            "and audio.prompt/audio.duration_s/audio.mix_db are fields inside the `audio` object. "
            "Do not use alias keys like preset_primary, preset_fallbacks, audio_prompt, decision."
        ),
    }

    image_content = {
        "type": "input_image",
        "image_url": _image_to_data_uri(image_path),
    }
    image_bytes = image_path.read_bytes()
    io_payload: Dict[str, Any] = {
        "request": {
            "api_mode_preference": "responses_json_schema_then_chat_fallback",
            "model": model,
            "system_prompt": SYSTEM_PROMPT,
            "user_text": user_text["text"],
            "image": {
                "path": str(image_path),
                "bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "mime": image_content["image_url"].split(";")[0].replace("data:", ""),
            },
            "metadata": metadata,
        },
        "attempts": [],
    }

    # Retries handle occasional malformed/empty model output despite strict schema.
    try:
        for attempt in range(1, 4):
            raw_text = ""
            if hasattr(client, "responses"):
                response = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [user_text, image_content]},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "decision_schema",
                            "strict": True,
                            "schema": DECISION_SCHEMA,
                        }
                    },
                )
                raw_text = _extract_text(response)
                status = "ok"
                api_mode = "responses"
            else:
                # Compatibility path for SDKs without `client.responses`.
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Return decision JSON for image-to-video rendering. "
                                        f"Metadata: {json.dumps(metadata, ensure_ascii=True)}. "
                                        "Return strict JSON only, include exactly two fallbacks, "
                                        "set framing.target_aspect=instagram_reel_9_16, and use only preset enum values. "
                                        "Mandatory: include scene.tags, scene.has_people, scene.confidence. "
                                        "Add short video.params.prompt or video.params.animation_directions when useful to describe motion while preserving composition. "
                                        "For wide scenes where vertical pan/crop would be too zoomed, set video.params.output_aspect=square_1_1. "
                                        "Choose crop_anchor to keep people uncut in frame. "
                                        "Top-level keys must be exactly: scene, framing, video, audio, fallbacks "
                                        "(do not nest scene/framing inside video). "
                                        "Use nested object fields only: video.preset means the `preset` field inside the `video` object, "
                                        "and audio.prompt/audio.duration_s/audio.mix_db are fields inside the `audio` object. "
                                        "Do not use alias keys like preset_primary, preset_fallbacks, audio_prompt, decision."
                                    ),
                                },
                                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_path)}},
                            ],
                        },
                    ],
                )
                raw_text = _strip_json_fences(_extract_chat_text(response))
                status = "ok_chat_fallback"
                api_mode = "chat_completions"

            usage = _extract_usage(response)
            usage_acc["input_tokens"] += usage["input_tokens"]
            usage_acc["output_tokens"] += usage["output_tokens"]
            usage_acc["total_tokens"] += usage["total_tokens"]
            io_payload["attempts"].append(
                {
                    "attempt": attempt,
                    "api_mode": api_mode,
                    "usage": usage,
                    "raw_output_text": _truncate(raw_text),
                }
            )

            try:
                parsed = json.loads(raw_text)
                return {
                    "decision": validate_and_clamp_decision(_coerce_decision_shape(parsed)),
                    "openai": {
                        "model": model,
                        "attempts": attempt,
                        "usage": usage_acc,
                        "status": status,
                        "io": io_payload,
                    },
                }
            except Exception:
                if attempt == 3:
                    raise
    except Exception as exc:
        return {
            "decision": validate_and_clamp_decision(_fallback_decision_payload()),
            "openai": {
                "model": model,
                "attempts": 0,
                "usage": usage_acc,
                "status": "fallback_openai_error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "io": io_payload,
            },
        }

    return {
        "decision": validate_and_clamp_decision(_fallback_decision_payload()),
        "openai": {
            "model": model,
            "attempts": 0,
            "usage": usage_acc,
            "status": "fallback_unknown",
            "io": io_payload,
        },
    }


def decide_for_image(image_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return decide_for_image_detailed(image_path=image_path, metadata=metadata)["decision"]
