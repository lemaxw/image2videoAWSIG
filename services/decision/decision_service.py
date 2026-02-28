import base64
import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from services.orchestrator.validate import validate_and_clamp_decision

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["scene", "video", "audio", "fallbacks"],
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
        "video": {
            "$ref": "#/$defs/videoObj"
        },
        "audio": {
            "type": "object",
            "required": ["prompt", "duration_s", "mix_db"],
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string"},
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

SYSTEM_PROMPT = """You are a video preset decision engine. Analyze the input image and produce JSON only.
Choose one preset from enum, produce audio prompt, and exactly 2 fallback video objects.
Prefer low-memory-safe configs when uncertain. Keep outputs concise."""


def _image_to_data_uri(image_path: Path) -> str:
    mime = "image/png"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    if image_path.suffix.lower() == ".webp":
        mime = "image/webp"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", "") in {"output_text", "text"}:
                chunks.append(getattr(content, "text", ""))
    return "\n".join(chunks)


def decide_for_image(image_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = metadata or {}
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    user_text = {
        "type": "input_text",
        "text": (
            "Return decision JSON for image-to-video rendering. "
            f"Metadata: {json.dumps(metadata, ensure_ascii=True)}. "
            "Enforce constraints and include exactly two fallbacks."
        ),
    }

    image_content = {
        "type": "input_image",
        "image_url": _image_to_data_uri(image_path),
    }

    for attempt in range(1, 4):
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
        try:
            parsed = json.loads(raw_text)
            return validate_and_clamp_decision(parsed)
        except Exception:
            if attempt == 3:
                raise

    raise RuntimeError("Failed to generate decision JSON")
