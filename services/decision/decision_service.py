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
                        "SVD_SUBTLE",
                        "SVD_STRONG",
                        "ANIMATEDIFF_GRASS_WIND",
                        "ANIMATEDIFF_CITY_PULSE",
                        "ANIMATEDIFF_LOW_MEM",
                        "FAILSAFE_LOW_MEM"
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

SYSTEM_PROMPT = """You are a decision engine for an image -> anime -> video pipeline.

The pipeline always works in two stages:
1) the input image is first redrawn into anime style while preserving composition
2) the anime image is then animated into a short cinematic clip

Your task is to choose the safest and most appropriate animation preset and provide minimal scene guidance.

Output JSON only and follow the schema exactly.

------------------------------------------------
PRESET GUIDE
------------------------------------------------

Choose ONE primary preset and EXACTLY TWO fallbacks.

SVD_SUBTLE
- default choice
- conservative anime redraw
- minimal motion
- best identity/composition preservation
- safest for 8GB GPUs

SVD_STRONG
- cinematic anime interpretation
- stronger motion
- more dynamic sky, water, foliage
- higher risk of artifacts

ANIMATEDIFF_GRASS_WIND
- outdoor anime scenes
- nature environments
- gentle wind movement
- foliage, grass, clouds drifting

ANIMATEDIFF_CITY_PULSE
- urban anime scenes
- neon lights, streets, reflections
- stronger graphic contrast and motion

ANIMATEDIFF_LOW_MEM
- simplified anime motion
- lower complexity scenes
- safer for limited VRAM

FAILSAFE_LOW_MEM
- simplest possible animation
- minimal motion
- last fallback for stability

If uncertain prefer:
SVD_SUBTLE -> FAILSAFE_LOW_MEM

------------------------------------------------
FRAMING GUIDE
------------------------------------------------

target_aspect must always be:

instagram_reel_9_16

crop_anchor indicates where the main subject should remain after cropping.

Allowed values:
left_top
center_top
right_top
left_center
center_center
right_center
left_bottom
center_bottom
right_bottom

Cropping rules:
- never cut off heads or bodies
- if people are present and position uncertain -> use center_center
- if main subject clearly centered -> center_center
- if subject is clearly near an edge -> anchor to that edge

------------------------------------------------
SCENE GUIDE (MANDATORY)
------------------------------------------------

Always return:

scene.tags
scene.has_people
scene.confidence

Rules:
- scene.tags should contain several short keywords, ideally 5-8 items
- examples: desert, road, sunset, city, night, mountains, ocean
- do not use long sentences
- scene.has_people must indicate if one or more people appear
- scene.confidence must be a float 0-1

------------------------------------------------
ANIME STYLIZATION GUIDE
------------------------------------------------

The system first redraws the image as anime.

You may add optional stylization hints in:

video.params.anime_prompt_hint

Purpose:
guide how the anime redraw should enhance the scene while preserving composition.

Keep it short and visual.

Examples:

cars -> glowing light streaks
water -> colorful anime reflections
sky -> dramatic anime clouds
city night -> soft neon glow
foliage -> simplified anime leaves
desert -> glowing warm sand tones

Do not change the core layout of the scene.

------------------------------------------------
MOTION DESIGN GUIDE
------------------------------------------------

Prefer motion that feels natural and cinematic.

Typical motion elements:

clouds drifting
grass moving in wind
light rays shifting
water ripples
soft atmospheric fog
neon reflections flicker

Avoid chaotic or aggressive motion.

------------------------------------------------
AUDIO GUIDE
------------------------------------------------

Return ONE compact ambient prompt under 96 characters.

Structure:
"<environment>, <secondary texture>, cinematic ambience"

Examples:

desert wind, airy sand movement, cinematic ambience
quiet city night, soft neon hum, cinematic ambience
mountain valley, gentle wind and birds, cinematic ambience

Rules:
- match the intended anime interpretation
- avoid aggressive words: loud, chaotic, intense
- prefer ambient soundscape over melody-heavy music
- include at least one texture word:
soft, warm, airy, dreamy, gentle, atmospheric, ambient

------------------------------------------------
KEY NAMING CONTRACT (MANDATORY)
------------------------------------------------

Use exactly these keys:

video.preset
fallbacks (array of exactly two video objects)
audio.prompt
audio.duration_s
audio.mix_db

Do NOT use aliases like:
preset_primary
preset_fallbacks
audio_prompt
decision

------------------------------------------------
FINAL RULE
------------------------------------------------

When uncertain always prioritize stability and composition preservation.
Prefer SVD_SUBTLE or FAILSAFE_LOW_MEM.
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
            "preset": "FAILSAFE_LOW_MEM",
            "duration_s": 5,
            "fps": 6,
            "frames": 20,
            "resolution_width": 576,
            "params": {"motion_bucket_id": 24, "steps": 14},
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
                "preset": "FAILSAFE_LOW_MEM",
                "duration_s": 3,
                "fps": 5,
                "frames": 15,
                "resolution_width": 512,
                "params": {"motion_bucket_id": 20, "steps": 12},
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


def _coerce_decision_shape(parsed: Dict[str, Any]) -> Dict[str, Any]:
    # Some chat-completions responses return a shorthand shape:
    # {"preset":"...", "fallbacks":["...","..."], "audio": {...}, "framing": {...}}
    # Normalize that into the strict contract expected downstream.
    if not isinstance(parsed, dict):
        return _fallback_decision_payload()
    nested = parsed.get("decision")
    if isinstance(nested, dict):
        parsed = nested
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
            "Add a short video.params.anime_prompt_hint when useful to make the anime redraw more expressive while preserving composition. "
            "Choose crop_anchor to keep people uncut in frame. "
            "Top-level keys must be exactly: scene, framing, video, audio, fallbacks (do not nest scene/framing inside video). "
            "Use exact keys only: video.preset, fallbacks, audio.prompt/audio.duration_s/audio.mix_db. "
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
                                        "Add a short video.params.anime_prompt_hint when useful to make the anime redraw more expressive while preserving composition. "
                                        "Choose crop_anchor to keep people uncut in frame. "
                                        "Top-level keys must be exactly: scene, framing, video, audio, fallbacks "
                                        "(do not nest scene/framing inside video). "
                                        "Use exact keys only: video.preset, fallbacks, audio.prompt/audio.duration_s/audio.mix_db. "
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
