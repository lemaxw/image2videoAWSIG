import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from openai import OpenAI

from services.orchestrator.validate import validate_and_clamp_decision

try:
    from image2json.analyzer import ImageAnalyzer
    from image2json.config import AnalysisConfig
    IMAGE2JSON_AVAILABLE = True
except ImportError:
    IMAGE2JSON_AVAILABLE = False


def _decision_step_log(event: str, step: str, **fields: Any) -> None:
    payload = {
        "level": "INFO" if event != "failed" else "ERROR",
        "msg": f"decision.{step}.{event}",
        "step": step,
        "event": event,
        "time": int(time.time()),
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _ollama_unload_model(ollama_url: str, model: str, *, reason: str, timeout: float = 30) -> Dict[str, Any]:
    started = time.time()
    _decision_step_log("start", "ollama_unload", model=model, reason=reason, url=ollama_url)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            )
            response.raise_for_status()
            result = {"status_code": response.status_code}
            _decision_step_log(
                "done",
                "ollama_unload",
                model=model,
                reason=reason,
                duration_s=round(time.time() - started, 3),
                result=result,
            )
            return result
    except Exception as exc:
        error = {"error": str(exc), "error_type": exc.__class__.__name__}
        _decision_step_log(
            "failed",
            "ollama_unload",
            model=model,
            reason=reason,
            duration_s=round(time.time() - started, 3),
            **error,
        )
        return error

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

TEXT_MODEL_SYSTEM_PROMPT = """You are a local decision model that selects the best image-to-video treatment for one image.

You will receive one JSON object produced by image2json. It contains general image analysis, including scene, subjects, objects, spatial_map, dynamic_potential, style, composition, text, reframe_constraints, content_complexity, framing_risks, generation_risks, and confidence.

Your task:
Analyze the provided JSON and return one decision JSON object that matches the application DECISION_SCHEMA.

You must base the decision only on the provided image2json data.
Do not invent invisible image details.
Do not add explanations outside the JSON.
Do not use markdown fences.
Return valid JSON only.

================================================
OUTPUT CONTRACT
================================================

Top-level keys must be exactly:

{
  "scene": {},
  "framing": {},
  "video": {},
  "audio": {},
  "fallbacks": []
}

No extra top-level keys.

Required structure:

{
  "scene": {
    "tags": [],
    "has_people": false,
    "confidence": 0.0
  },
  "framing": {
    "target_aspect": "instagram_reel_9_16",
    "crop_anchor": "center_center"
  },
  "video": {
    "preset": "HUNYUAN15_I2V_720P",
    "duration_s": 5,
    "fps": 24,
    "frames": 120,
    "resolution_width": 720,
    "seed": 0,
    "params": {
      "prompt": "",
      "negative_prompt": "",
      "use_original_input_for_video": false,
      "output_aspect": "instagram_reel_9_16",
      "camera_motion": "",
      "motion_strength": ""
    }
  },
  "audio": {
    "prompt": "",
    "duration_s": 5,
    "mix_db": -12.0
  },
  "fallbacks": []
}

The `fallbacks` array must contain exactly 2 video objects.
Each fallback video object must use the same structure as `video`.
Fallbacks must use presets from different backend families when possible.

================================================
VALID ENUMS
================================================

Valid video.preset values:

- HUNYUAN15_I2V_720P
- HUNYUAN15_I2V_FAST
- SVD_SUBTLE
- SVD_STRONG
- ANIMATEDIFF_GRASS_WIND
- ANIMATEDIFF_CITY_PULSE

Valid framing.crop_anchor values:

- left_top
- center_top
- right_top
- left_center
- center_center
- right_center
- left_bottom
- center_bottom
- right_bottom

Valid video.params.output_aspect values:

- instagram_reel_9_16
- square_1_1
- original

Valid video.params.motion_strength values:

- very_low
- low
- medium
- high

================================================
PRESET MEANING
================================================

HUNYUAN15_I2V_720P:
Use for premium realistic/cinematic output. Good for landscapes, architecture, travel, people, products, and realistic scenes when artifact risk is acceptable.

HUNYUAN15_I2V_FAST:
Use when the scene is realistic but dense, complex, expensive, or likely to need a safer/lighter Hunyuan option.

ANIMATEDIFF_GRASS_WIND:
Use for outdoor/nature/anime-style atmosphere involving foliage, grass, trees, clouds, dreamy scenery, or gentle wind.

ANIMATEDIFF_CITY_PULSE:
Use for urban/anime-style atmosphere involving city streets, neon, rain, nightlife, reflections, traffic, or energetic city mood.

SVD_STRONG:
Use for dramatic atmospheric reinterpretation when motion potential is high and the scene contains cinematic elements such as ocean, sunset, fog, haze, dramatic sky, reflections, particles, smoke, or glowing light.

SVD_SUBTLE:
Use as the safest stability-first option when the image contains fragile details such as faces, readable text, logos, dense architecture, complex geometry, low-detail noisy regions, or strong framing/generation risks.

================================================
DECISION PRIORITIES
================================================

Choose the best preset by balancing these priorities in order:

1. Preserve the visible image story.
2. Avoid likely visual artifacts.
3. Match the scene type and style.
4. Use natural motion opportunities from the image2json data.
5. Prefer high quality when risk is acceptable.
6. Use fallbacks to cover alternative safe strategies.

Do not blindly choose the first preset in the preference order.
The preference order is only a tie-breaker after risk and scene fit are considered.

Tie-breaker preference order:

1. HUNYUAN15_I2V_720P
2. HUNYUAN15_I2V_FAST
3. ANIMATEDIFF_GRASS_WIND
4. ANIMATEDIFF_CITY_PULSE
5. SVD_STRONG
6. SVD_SUBTLE

================================================
SCENE FIELD RULES
================================================

scene.tags:
Return 5 to 8 lowercase keywords.
Use short tags based on visible content, environment, mood, style, and important objects.
Do not include backend names or technical settings.

scene.has_people:
true if image2json.people is non-empty, or if subjects/objects indicate visible people.
false otherwise.

scene.confidence:
Use image2json.confidence.overall if available.
Otherwise infer from available confidence values.
Clamp to 0.0–1.0.

================================================
FRAMING RULES
================================================

target_aspect:
Always set to "instagram_reel_9_16" unless vertical_crop_risk is "high" and the image story would be damaged by a vertical crop. Even then, keep framing.target_aspect as "instagram_reel_9_16"; put the safer output aspect in video.params.output_aspect.

crop_anchor:
Use spatial_map.primary_regions when available.

Choose the most important region:
- Prefer regions with importance "primary".
- If multiple primary regions exist, prefer the one that best represents the image story.
- If important regions are spread across the full image, use center_center.
- If people are present and important, prefer the anchor that keeps people visible.
- If uncertain, use center_center.

Map the selected region center to a 3x3 grid:

x < 0.33 -> left
0.33 <= x <= 0.66 -> center
x > 0.66 -> right

y < 0.33 -> top
0.33 <= y <= 0.66 -> center
y > 0.66 -> bottom

Combine as:
left_top, center_top, right_top,
left_center, center_center, right_center,
left_bottom, center_bottom, right_bottom.

Wide composition:
If reframe_constraints.wide_composition is true, or full_width_important_content is true, or vertical_crop_risk is "high":
- Prefer crop_anchor center_center unless a clear subject region is dominant.
- Set video.params.use_original_input_for_video = true.
- Set video.params.output_aspect = "square_1_1" if vertical crop would lose the main story.
- Prefer "square_1_1" over "original" for panoramas, broad landscapes, or wide scenes where a 9:16 crop would be too narrow or would add padding/margins.
- Use "original" only when the input explicitly requires a full original-aspect export and no square or vertical crop can preserve the story.
- Describe slow lateral camera travel or gentle push-in instead of aggressive reframing.

People/framing safety:
- Treat people/faces as major artifact risks only when they are important to the final frame: large or medium size, central, primary subjects, or likely to remain visible after the chosen crop/pan.
- If people are tiny, partial, peripheral, or likely to be outside the final crop/pan, do not let face risk alone force SVD_SUBTLE.
- Avoid implying tight crops around people if people are tiny, partial, or near edges.
- If important people remain near image edges, prefer safer framing and lower motion strength.

================================================
PRESET SELECTION RULES
================================================

First detect risk level.

High risk if any of these are true:
- text.has_visible_text is true
- content_complexity.faces is true and faces are important to the final frame
- content_complexity.dense_details is true and level is "high"
- framing_risks is non-empty and affects important final-frame subjects
- generation_risks is non-empty and affects important final-frame subjects
- spatial_map.safe_reframe_difficulty is "high"
- reframe_constraints.vertical_crop_risk is "high"

Medium risk if:
- dense details are present
- readable text exists but is small/peripheral
- important subjects touch edges
- scene contains architecture, crowds, fine geometry, watermarks, or complex patterns

Low risk if:
- no visible text
- no important final-frame faces/hands
- no major framing risks
- clear subject/background separation
- simple natural scene

Preset choice:

Use SVD_SUBTLE when:
- risk is high and the final frame contains important fragile content, readable text, faces, logos, or dense detail
- or the source image is too fragile for strong motion

Use HUNYUAN15_I2V_FAST when:
- scene is realistic
- quality matters
- but content complexity or density suggests a safer/lighter realistic preset

Use HUNYUAN15_I2V_720P when:
- scene is realistic/cinematic/travel/landscape/architecture/product/portrait
- risk is low or medium
- motion can be restrained and natural

Use ANIMATEDIFF_GRASS_WIND when:
- scene is outdoor/nature/forest/grass/trees/mountains/clouds
- dynamic_potential includes wind, foliage, grass, clouds, atmospheric motion
- anime/stylized outdoor output is acceptable
- tiny or peripheral people may be present, as long as important final-frame faces/text/logos do not dominate the risk

Use ANIMATEDIFF_CITY_PULSE when:
- scene is urban/city/street/night/neon/rain/reflections
- style or scene suggests city energy

Use SVD_STRONG when:
- dynamic_potential.level is "high"
- scene contains dramatic sky, ocean, sunset, fog, haze, particles, reflections, smoke, or glowing light
- no fragile people/text/logo risks dominate

For realistic travel/landscape images:
- Prefer HUNYUAN15_I2V_720P if risk is acceptable.
- Prefer HUNYUAN15_I2V_FAST if the image is dense or wide with many details.
- Prefer ANIMATEDIFF_GRASS_WIND as a fallback for outdoor/nature scenes with natural wind/foliage/cloud motion when no important final-frame faces/text/logos dominate.
- Prefer SVD_SUBTLE if visible text/watermark, important faces, or important geometry risk dominates the final frame.

================================================
VIDEO PARAMS RULES
================================================

video.duration_s:
Use 5 by default.
Use 3 for high-risk images, dense scenes, text-heavy images, or fragile portraits.

video.fps:
Use 24 unless the schema or application requires another value.

video.frames:
frames = duration_s * fps.

video.resolution_width:
Use 720 by default.

video.seed:
Use an integer.
If no seed is provided in the input, use 0.

video.params.prompt:
Max 300 characters.
Base it on:
- image2json.summary
- image2json.detailed_description
- dynamic_potential.natural_motion_elements
- dynamic_potential.camera_motion_affordances
- scene mood/style/lighting

The prompt should:
- describe the visible scene
- request natural, restrained motion
- preserve original composition and identity of visible subjects
- include camera motion if useful
- avoid mentioning backend names

video.params.negative_prompt:
Max 220 characters.
Mention risks relevant to the image:
- warped faces
- distorted text
- geometry wobble
- flicker
- melting details
- extra people/objects
- unstable architecture
- over-strong motion

video.params.camera_motion:
Choose a short phrase such as:
- "slow push-in"
- "gentle lateral drift"
- "subtle parallax"
- "locked camera with atmospheric motion"
- "slow reveal"
- "minimal stabilized motion"

video.params.motion_strength:
Use:
- very_low for fragile/high-risk images
- low for text, people, dense details, architecture, or wide compositions
- medium for normal realistic scenes
- high only for dramatic scenes with high dynamic potential and low artifact risk

video.params.use_original_input_for_video:
true when preserving the original full image is safer than cropping/reframing.
Set true for wide compositions with important content spread across the image.
Otherwise false.

video.params.output_aspect:
- "instagram_reel_9_16" when vertical treatment is safe
- "square_1_1" when vertical crop risk is high, a pan/crop would be too narrow, or preserving a wide story matters
- "original" only when the input explicitly requires original-aspect final export and neither 9:16 nor 1:1 can preserve the story

================================================
AUDIO RULES
================================================

audio.prompt:
Max 96 characters.
Format:
"<environment>, <mood or texture>, cinematic ambience"

Use scene.environment, scene.mood, weather, and style.color_palette when available.
Do not mention backend names.
Do not include more than one sentence.

audio.duration_s:
Match video.duration_s.

audio.mix_db:
Use -12.0 by default.
Use -14.0 for calm/serene scenes.
Use -10.0 for energetic city/dramatic scenes.

================================================
FALLBACK RULES
================================================

Return exactly 2 fallback video objects.

Fallbacks must:
- use different preset families when possible:
  - Hunyuan family: HUNYUAN15_I2V_720P, HUNYUAN15_I2V_FAST
  - SVD family: SVD_SUBTLE, SVD_STRONG
  - AnimateDiff family: ANIMATEDIFF_GRASS_WIND, ANIMATEDIFF_CITY_PULSE
- not duplicate the primary video.preset
- use the same duration/fps/frame logic
- use safer motion than the primary when risk is medium or high
- keep prompts consistent with the visible image

Recommended fallback strategy:
- If primary is HUNYUAN15_I2V_720P:
  fallback 1: HUNYUAN15_I2V_FAST only if allowed by schema preference, otherwise SVD_SUBTLE
  fallback 2: ANIMATEDIFF_GRASS_WIND for nature/outdoor/wide landscape when important final-frame faces/text/logos do not dominate, ANIMATEDIFF_CITY_PULSE for city/urban, otherwise SVD_SUBTLE
- If primary is HUNYUAN15_I2V_FAST:
  fallback 1: HUNYUAN15_I2V_720P if risk is not high, otherwise SVD_SUBTLE
  fallback 2: appropriate AnimateDiff preset if scene matches and fragile final-frame details do not dominate, otherwise SVD_SUBTLE
- If primary is SVD_SUBTLE:
  fallback 1: HUNYUAN15_I2V_FAST
  fallback 2: appropriate AnimateDiff preset if scene matches, otherwise SVD_STRONG
- If primary is SVD_STRONG:
  fallback 1: HUNYUAN15_I2V_720P
  fallback 2: SVD_SUBTLE
- If primary is ANIMATEDIFF_GRASS_WIND:
  fallback 1: HUNYUAN15_I2V_720P
  fallback 2: HUNYUAN15_I2V_FAST when realistic fallback quality matters, otherwise SVD_SUBTLE
- If primary is ANIMATEDIFF_CITY_PULSE:
  fallback 1: HUNYUAN15_I2V_720P
  fallback 2: HUNYUAN15_I2V_FAST when realistic fallback quality matters, otherwise SVD_SUBTLE

If a fallback would duplicate the primary preset, choose the safest non-duplicate alternative.

================================================
CONSISTENCY CHECKS BEFORE RETURNING
================================================

Before returning, verify:

- Output is valid JSON.
- No markdown fences.
- Top-level keys are exactly: scene, framing, video, audio, fallbacks.
- scene.tags contains 5 to 8 strings.
- framing.crop_anchor is one of the valid enum values.
- video.preset is one of the valid enum values.
- video.duration_s is either 3 or 5.
- video.frames equals video.duration_s * video.fps.
- audio.duration_s equals video.duration_s.
- audio.prompt is 96 characters or less.
- video.params.prompt is 300 characters or less.
- fallbacks contains exactly 2 video objects.
- fallback presets do not duplicate the primary preset.
- All decisions are based on the provided image2json input.

Return JSON only.
"""


def _call_ollama_text_model(ollama_url: str, model: str, prompt: str, json_input: Dict[str, Any], schema: Dict[str, Any], timeout: float = 300) -> str:
    """Call Ollama text model with JSON input and schema, return the response."""
    full_prompt = f"""{prompt}

JSON Schema for output:
{json.dumps(schema, indent=2)}

Image analysis JSON:
{json.dumps(json_input, indent=2)}

Return JSON only. No markdown fences, no extra text."""

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": 0,
        "options": {"temperature": 0},
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
    return result.get("response", "")


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

    # Try image2json + text model two-step flow if enabled and available
    image2json_enabled = os.environ.get("IMAGE2JSON_ENABLED", "true").lower() == "true"
    if image2json_enabled and IMAGE2JSON_AVAILABLE:
        try:
            ollama_url = os.environ.get("IMAGE2JSON_URL", "http://host.docker.internal:11434")
            image2json_model = os.environ.get("IMAGE2JSON_MODEL", "qwen3-vl:8b")
            text_model = os.environ.get("IMAGE2JSON_TEXT_MODEL", "qwen3:14b")
            timeout = float(os.environ.get("IMAGE2JSON_TIMEOUT", "300"))
            if image2json_model == text_model or "vl" not in image2json_model.lower():
                raise RuntimeError(
                    "IMAGE2JSON_MODEL must be the vision model qwen3-vl:8b for this pipeline; "
                    f"got IMAGE2JSON_MODEL={image2json_model!r}, IMAGE2JSON_TEXT_MODEL={text_model!r}"
                )

            # Step 1: Get image analysis from image2json (vision model)
            config = AnalysisConfig(
                model=image2json_model,
                ollama_url=ollama_url,
                timeout=timeout,
                short_version=False,  # Use full analysis
            )
            vision_started = time.time()
            _decision_step_log(
                "start",
                "image2json_vision",
                model=image2json_model,
                url=ollama_url,
                image_path=str(image_path),
                timeout_s=timeout,
            )
            try:
                analyzer = ImageAnalyzer(config)
                analysis = analyzer.analyze_path(image_path)
                _decision_step_log(
                    "done",
                    "image2json_vision",
                    model=image2json_model,
                    duration_s=round(time.time() - vision_started, 3),
                    result={"analysis_type": analysis.__class__.__name__},
                )
            except Exception as exc:
                _decision_step_log(
                    "failed",
                    "image2json_vision",
                    model=image2json_model,
                    duration_s=round(time.time() - vision_started, 3),
                    error=str(exc),
                    error_type=exc.__class__.__name__,
                )
                raise
            finally:
                _ollama_unload_model(ollama_url, image2json_model, reason="after_image2json_vision")

            # Convert analysis to dict for text model
            analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else str(analysis)

            # Step 2: Pass analysis to text model for decision
            text_started = time.time()
            _decision_step_log(
                "start",
                "decision_text_model",
                model=text_model,
                url=ollama_url,
                analysis_model=image2json_model,
                timeout_s=timeout,
            )
            try:
                text_response = _call_ollama_text_model(
                    ollama_url=ollama_url,
                    model=text_model,
                    prompt=TEXT_MODEL_SYSTEM_PROMPT,
                    json_input=analysis_dict,
                    schema=DECISION_SCHEMA,
                    timeout=timeout,
                )
                parsed_decision = json.loads(_strip_json_fences(text_response))
                decision = _coerce_decision_shape(parsed_decision)
                _decision_step_log(
                    "done",
                    "decision_text_model",
                    model=text_model,
                    duration_s=round(time.time() - text_started, 3),
                    result={
                        "preset": (decision.get("video") or {}).get("preset"),
                        "fallbacks": [fb.get("preset") for fb in decision.get("fallbacks", []) if isinstance(fb, dict)],
                    },
                )
            except Exception as exc:
                _decision_step_log(
                    "failed",
                    "decision_text_model",
                    model=text_model,
                    duration_s=round(time.time() - text_started, 3),
                    error=str(exc),
                    error_type=exc.__class__.__name__,
                )
                raise
            finally:
                _ollama_unload_model(ollama_url, text_model, reason="after_decision_text_model")

            return {
                "decision": validate_and_clamp_decision(decision),
                "openai": {
                    "model": model,
                    "attempts": 0,
                    "usage": usage_acc,
                    "status": "skipped_image2json_used",
                    "io": io_payload_base,
                },
                "image2json": {
                    "enabled": True,
                    "used": True,
                    "model": image2json_model,
                    "vision_model": image2json_model,
                    "text_model": text_model,
                    "analysis": analysis_dict,
                    "text_response": text_response,
                },
            }
        except Exception as exc:
            # Exit on image2json error instead of falling back
            print(f"image2json decision failed: {exc.__class__.__name__}: {exc}")
            raise
    elif image2json_enabled and not IMAGE2JSON_AVAILABLE:
        # image2json enabled but not installed - exit instead of fallback
        print("image2json is enabled but package is not installed")
        raise RuntimeError("image2json package not installed")

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
            "image2json": {
                "enabled": image2json_enabled,
                "used": False,
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
            "image2json": {
                "enabled": image2json_enabled,
                "used": False,
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
                    "image2json": {
                        "enabled": image2json_enabled,
                        "used": False,
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
            "image2json": {
                "enabled": image2json_enabled,
                "used": False,
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
        "image2json": {
            "enabled": image2json_enabled,
            "used": False,
        },
    }


def decide_for_image(image_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return decide_for_image_detailed(image_path=image_path, metadata=metadata)["decision"]
