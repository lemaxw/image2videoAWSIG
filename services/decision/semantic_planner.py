"""Semantic decision v2 and deterministic compilation to renderer settings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable

import httpx


SEMANTIC_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["classification", "motion_plan", "generation", "presentation", "audio"],
    "additionalProperties": False,
    "properties": {
        "classification": {
            "type": "object",
            "required": ["scene_classes", "environment", "important_subjects", "incidental_subjects", "sensitive_content", "preservation_risk"],
            "additionalProperties": False,
            "properties": {
                "scene_classes": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "environment": {"type": "string", "maxLength": 120},
                "important_subjects": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "incidental_subjects": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "sensitive_content": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "preservation_risk": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        },
        "motion_plan": {
            "type": "object",
            "required": ["primary_target", "primary_action", "secondary_target", "secondary_action", "keep_stable"],
            "additionalProperties": False,
            "properties": {
                "primary_target": {"type": "string", "maxLength": 100},
                "primary_action": {"type": "string", "maxLength": 160},
                "secondary_target": {"type": "string", "maxLength": 100},
                "secondary_action": {"type": "string", "maxLength": 160},
                "keep_stable": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            },
        },
        "generation": {
            "type": "object",
            "required": ["mode", "backend", "prompt", "negative_prompt", "candidate_count", "reason"],
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["generative"]},
                "backend": {"type": "string", "enum": ["wan22", "hunyuan15"]},
                "prompt": {"type": "string", "maxLength": 420},
                "negative_prompt": {"type": "string", "maxLength": 220},
                "candidate_count": {"type": "integer", "minimum": 1, "maximum": 3},
                "reason": {"type": "string", "maxLength": 300},
            },
        },
        "presentation": {
            "type": "object",
            "required": ["aspect", "operation", "crop_anchor", "focus_target", "pan_start", "pan_end", "must_keep_visible"],
            "additionalProperties": False,
            "properties": {
                "aspect": {"type": "string", "enum": ["instagram_reel_9_16", "square_1_1"]},
                "operation": {"type": "string", "enum": ["static", "pan_left_to_right", "pan_right_to_left", "push_in", "pull_out"]},
                "crop_anchor": {"type": "string", "enum": ["left_top", "center_top", "right_top", "left_center", "center_center", "right_center", "left_bottom", "center_bottom", "right_bottom"]},
                "focus_target": {"type": "string", "maxLength": 100},
                "pan_start": {"type": "number", "minimum": 0, "maximum": 1},
                "pan_end": {"type": "number", "minimum": 0, "maximum": 1},
                "must_keep_visible": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            },
        },
        "audio": {
            "type": "object",
            "required": ["prompt", "duration_s"],
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string", "maxLength": 180},
                "duration_s": {"type": "integer", "enum": [5]},
            },
        },
    },
}


SEMANTIC_SYSTEM_PROMPT = """You plan one realistic image-to-video treatment.

Input has two sections: image2json (facts observed in the current image) and
similar_cases (verbatim prior experience returned by MemPalace).

Use image2json as the only authority for visible objects. Similar cases are
evidence about model choice, prompt structure, failure warnings, seed policy,
and post-production; never copy an object or action absent from image2json.
Seeds do not transfer reliably between different images.

Routing:
- An important primary/secondary visible person, fauna/animal, or vehicle -> hunyuan15.
- Everything else -> wan22. For a visually static scene, request restrained
  environmental motion and protect all fixed geometry rather than replacing the
  generative render with a repeated still.
- Tiny distant incidental people/fauna/vehicles are risk flags, not automatic Hunyuan.
- Deterministic original-image treatment is an execution fallback after renderer
  failure; it is not a semantic primary backend.

Prompt:
- Short, action-focused, and grounded in existing content.
- State primary motion, optional secondary motion, content that remains stable,
  and preservation of original composition/viewpoint.
- Do not name concrete unwanted objects/artifacts in either prompt. Use only
  abstract negatives: flicker, jitter, unstable geometry, inconsistent
  appearance, scene transition, low quality.
- Do not request generated camera movement. Camera presentation is deterministic.

Presentation:
- Generation preserves the original aspect. Choose post-generation static crop,
  slow partial pan, push, or pull.
- When a landscape or city is viewed through a near architectural/natural
  foreground frame or opening, choose push_in. The deterministic push must
  travel through the opening and finish on the distant focal scene with the
  foreground frame outside the final crop. The frame is transitional and does
  not need to remain visible for the full operation.
- Prefer a subtle deterministic push or partial pan for city and architectural
  landscapes. Use static only when crop motion would lose or clip required content.
- Prefer square for broad landscapes/cities when 9:16 loses the story.
- Pan only when it reveals separated content; normally keep span <= 0.18 over
  five seconds.
- If reframe constraints say the composition is panoramic, full-width content
  is important, and vertical crop risk is high, use square delivery with a
  smooth deterministic traversal. A fixed portrait crop is not acceptable for
  such a scene.
- Keep required subjects visible for the full operation.
- Select one explicit focal subject for crop motion. `must_keep_visible` contains
  only essential focal subjects that can remain together in the chosen crop;
  do not list the whole scene, borders, or a watermark by default.
- Use the normalized regions in image2json (especially high-importance
  attention regions and spatial_map) to choose the crop anchor and push focus.
- Put that observed subject label in `focus_target`; it may be a static visual
  focal point even though generation motion must target a physically moving part.

Motion discipline:
- Generation prompts describe object/environment motion only. Never put push,
  pan, zoom, dolly, tracking, or other camera instructions in them; those belong
  exclusively to Presentation.
- A motion target cannot also appear in keep_stable. Name the moving part and
  the fixed part precisely (for example, flexible outer boughs move while tree
  trunks and the ground remain fixed).
- Sunlight, shadows, accumulated snow, fixed terrain, and depth layers are not
  natural-motion elements. Loose powder falling from a branch may move, but the
  accumulated snow and ground snow remain fixed.
- For flowers, request visible localized stem sway and petal flutter while rocks,
  cacti, trunks, and terrain remain fixed. Avoid vague wording such as merely
  "subtle movement".

Audio:
- Keep ambience quiet and subordinate to the picture.
- Use only plausible scene sounds. Do not add insects or wind by default; when
  they are genuinely important, describe them as faint and distant.
- No music.

Return JSON only matching the schema."""


def _text(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip()


def _enum_token(value: Any) -> str:
    """Normalize schema enum values without destroying their underscores."""
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _labels(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [_text(item.get("label")) for item in items if isinstance(item, dict) and _text(item.get("label"))]


def build_memory_query(analysis: Dict[str, Any]) -> str:
    scene = analysis.get("scene") if isinstance(analysis.get("scene"), dict) else {}
    style = analysis.get("style") if isinstance(analysis.get("style"), dict) else {}
    dynamic = analysis.get("dynamic_potential") if isinstance(analysis.get("dynamic_potential"), dict) else {}
    complexity = analysis.get("content_complexity") if isinstance(analysis.get("content_complexity"), dict) else {}
    reframe = analysis.get("reframe_constraints") if isinstance(analysis.get("reframe_constraints"), dict) else {}
    parts: list[str] = [
        _text(analysis.get("summary")),
        _text(scene.get("environment")),
        _text(scene.get("location_type")),
        "subjects " + " ".join(_labels(analysis.get("subjects"))[:6]),
        "people " + " ".join(_labels(analysis.get("people"))[:4]),
        "objects " + " ".join(_labels(analysis.get("objects"))[:6]),
        "motion " + " ".join(_text(x) for x in (dynamic.get("natural_motion_elements") or [])[:6]),
        "risks " + " ".join(_text(x) for x in (dynamic.get("motion_risks") or [])[:5]),
        "style " + _text(style.get("visual_style")),
        "complexity " + _text(complexity.get("level")),
        "vertical crop " + _text(reframe.get("vertical_crop_risk")),
    ]
    query = " ".join(part for part in parts if part).lower()
    query = re.sub(r"[^a-z0-9 ]+", " ", query)
    return " ".join(query.split())[:250]


def sanitize_analysis_for_decision(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Remove image2json motion hints that describe static scene properties."""
    cleaned = json.loads(json.dumps(analysis))
    dynamic = cleaned.get("dynamic_potential") if isinstance(cleaned.get("dynamic_potential"), dict) else {}
    values = dynamic.get("natural_motion_elements") if isinstance(dynamic.get("natural_motion_elements"), list) else []
    rejected: list[str] = []
    retained: list[Any] = []
    for value in values:
        label = _text(value).lower()
        is_static = (
            label in {"sunlight", "light", "snow", "accumulated snow", "depth layer", "depth layers"}
            or "sunlight" in label
            or "shadow" in label
            or "accumulated snow" in label
        )
        if is_static:
            rejected.append(_text(value))
        else:
            retained.append(value)
    dynamic["natural_motion_elements"] = retained
    if rejected:
        dynamic["excluded_static_motion_elements"] = rejected
    cleaned["dynamic_potential"] = dynamic
    return cleaned


def _compact_cases(payload: Dict[str, Any], limit: int) -> list[Dict[str, Any]]:
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    compact: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        content = _text(item.get("text") or item.get("content"))
        if not content:
            continue
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        compact.append(
            {
                "content": content[:1800],
                "wing": _text(item.get("wing")),
                "room": _text(item.get("room")),
                "source_file": _text(item.get("source_file")),
                "similarity": item.get("similarity"),
                "distance": item.get("distance"),
            }
        )
        if len(compact) >= limit:
            break
    return compact


def retrieve_similar_experience(analysis: Dict[str, Any]) -> Dict[str, Any]:
    enabled = os.environ.get("MEMPALACE_ENABLED", "true").lower() == "true"
    query = build_memory_query(analysis)
    metadata: Dict[str, Any] = {"enabled": enabled, "query": query, "used": False, "cases": []}
    if not enabled:
        return metadata
    url = os.environ.get("MEMPALACE_URL", "http://memory:8090").rstrip("/")
    wing = os.environ.get("MEMPALACE_WING", "image2videoAWSIG")
    limit = max(1, min(5, int(os.environ.get("MEMPALACE_RESULTS", "3"))))
    timeout = float(os.environ.get("MEMPALACE_TIMEOUT", "90"))
    started = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{url}/search",
                json={"query": query, "wing": wing, "limit": max(limit * 2, limit), "max_distance": 1.25},
            )
            response.raise_for_status()
            raw = response.json()
        cases = _compact_cases(raw, limit)
        metadata.update({"used": bool(cases), "cases": cases, "duration_s": round(time.time() - started, 3), "url": url, "wing": wing})
    except Exception as exc:
        metadata.update({"error": str(exc), "error_type": exc.__class__.__name__, "duration_s": round(time.time() - started, 3), "url": url, "wing": wing})
        if os.environ.get("MEMPALACE_REQUIRED", "false").lower() == "true":
            raise
    return metadata


def _iter_observed(analysis: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("subjects", "people", "objects"):
        value = analysis.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def _important_sensitive_subject_label(analysis: Dict[str, Any]) -> str:
    focal = " ".join(_text(x).lower() for x in ((analysis.get("composition") or {}).get("focal_points") or []))
    primary = " ".join(_text(x.get("label")).lower() for x in ((analysis.get("spatial_map") or {}).get("primary_regions") or []) if isinstance(x, dict) and x.get("importance") == "primary")
    animal_terms = {"animal", "bird", "dog", "cat", "horse", "cow", "sheep", "wildlife", "fauna"}
    vehicle_terms = {"vehicle", "car", "scooter", "motorcycle", "bike", "bicycle", "bus", "truck", "train", "boat"}
    candidates: list[tuple[int, str]] = []
    for item in _iter_observed(analysis):
        label = _text(item.get("label")).lower()
        if any(term in label for term in ("mannequin", "statue", "sculpture", "doll", "poster", "photograph")):
            continue
        spatial = item.get("spatial") if isinstance(item.get("spatial"), dict) else {}
        size = _text(spatial.get("relative_size")).lower()
        tokens = _label_tokens(label)
        is_person = item in (analysis.get("people") or []) or bool(tokens & {"person", "people", "rider", "pedestrian", "man", "woman"})
        sensitive = is_person or bool(tokens & (animal_terms | vehicle_terms))
        if not sensitive:
            continue
        important = size in {"medium", "large", "dominant"} or label in focal or label in primary
        if important:
            score = 3 if size in {"large", "dominant"} else 2 if size == "medium" else 1
            candidates.append((score, _text(item.get("label"))))
    return max(candidates, default=(0, ""))[1]


def _important_sensitive_subject(analysis: Dict[str, Any]) -> bool:
    return bool(_important_sensitive_subject_label(analysis))


def _seed_base(source_sha256: str, backend: str) -> int:
    digest = hashlib.sha256(f"{source_sha256}:{backend}:v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


_CAMERA_TERMS = ("push", "pan", "zoom", "dolly", "track", "camera", "viewpoint")
_STATIC_MOTION_TERMS = (
    "sunlight",
    "sun light",
    "shadow",
    "depth layer",
    "terrain",
    "geography",
    "rock",
    "ground snow",
    "cactus",
    "cacti",
)


def _label_tokens(value: Any) -> set[str]:
    return {token.rstrip("s") for token in re.findall(r"[a-z0-9]+", _text(value).lower()) if len(token) > 2}


def _labels_overlap(left: Any, right: Any) -> bool:
    a, b = _label_tokens(left), _label_tokens(right)
    return bool(a and b and (a <= b or b <= a or len(a & b) >= min(2, len(a), len(b))))


def _parse_box(value: Any) -> Dict[str, float] | None:
    if isinstance(value, dict):
        try:
            x, y, w, h = (float(value[key]) for key in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return None
    else:
        numbers = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", str(value or ""))
        if len(numbers) != 4:
            return None
        x1, y1, x2, y2 = (float(number) for number in numbers)
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0.0 and h > 0.0):
        return None
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    return {"x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)}


def _observed_regions(analysis: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return reliable normalized regions, preferring specific attention boxes."""
    regions: list[Dict[str, Any]] = []
    composition = analysis.get("composition") if isinstance(analysis.get("composition"), dict) else {}
    for item in composition.get("attention_regions") or []:
        if not isinstance(item, dict):
            continue
        box = _parse_box(item.get("region"))
        label = _text(item.get("label"))
        if box and label:
            regions.append({"label": label, "box": box, "source": "composition.attention_regions", "importance": _text(item.get("importance")).lower()})
    spatial = analysis.get("spatial_map") if isinstance(analysis.get("spatial_map"), dict) else {}
    for item in spatial.get("primary_regions") or []:
        if not isinstance(item, dict):
            continue
        box = _parse_box(item.get("box_normalized"))
        label = _text(item.get("label"))
        if box and label:
            regions.append({"label": label, "box": box, "source": "spatial_map.primary_regions", "importance": _text(item.get("importance")).lower()})
    return regions


def _find_region(label: Any, regions: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    wanted = _text(label).lower()
    if not wanted:
        return None
    exact = next((item for item in regions if _text(item.get("label")).lower() == wanted), None)
    if exact:
        return exact
    matches = [item for item in regions if _labels_overlap(wanted, item.get("label"))]
    return matches[0] if matches else None


def _motion_action(label: str, requested_action: str, analysis: Dict[str, Any]) -> str:
    """Turn vague/camera motion into localized, physically plausible motion."""
    target = _text(label)
    lower = target.lower()
    action = _text(requested_action)
    action_lower = action.lower()
    camera_only = any(term in action_lower for term in _CAMERA_TERMS)
    vague = not action or camera_only or any(term in action_lower for term in ("subtle movement", "simulate natural", "motion"))
    if any(term in lower for term in ("popp", "flower", "blossom", "petal")):
        description = " ".join(
            _text(item.get("description")).lower()
            for item in (analysis.get("subjects") or [])
            if isinstance(item, dict) and _labels_overlap(target, item.get("label"))
        )
        location = "upper-right " if "upper right" in description else ""
        return f"the prominent {location}flower stem sways gently and its petals flutter visibly"
    if "snow" in lower and any(term in lower for term in ("tree", "branch", "bough")):
        return "snow-laden outer boughs near the sunlit focal area flex gently and visibly in a light breeze"
    if any(term in lower for term in ("tree", "branch", "bough", "foliage", "leaves")):
        snowy = "snow" in _text(analysis.get("summary")).lower()
        return (
            "snow-laden outer boughs near the sunlit focal area flex gently and visibly in a light breeze"
            if snowy
            else f"outer {target} move gently in a light breeze"
        )
    if "snow" in lower:
        return "nearby flexible bough tips move gently while accumulated snow remains fixed"
    if "cloud" in lower:
        return f"{target} drift visibly and steadily across the sky"
    if any(term in lower for term in ("stream", "river", "water", "wave", "reflection")):
        return f"{target} show clear continuous flow and small natural ripples"
    if any(term in lower for term in ("grass", "reed", "crop")):
        return f"{target} sway gently in localized waves"
    return action if not vague else f"{target} move gently and locally"


def _is_static_motion_target(label: Any) -> bool:
    lower = _text(label).lower()
    return not lower or any(term in lower for term in _STATIC_MOTION_TERMS)


def _normalize_motion_plan(plan: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(json.dumps(plan))
    motion = normalized.get("motion_plan") if isinstance(normalized.get("motion_plan"), dict) else {}
    requested_primary = _text(motion.get("primary_target"))
    candidates = [
        (_text(motion.get("primary_target")), _text(motion.get("primary_action"))),
        (_text(motion.get("secondary_target")), _text(motion.get("secondary_action"))),
    ]
    rigid_breeze = {
        label
        for label, action in candidates
        if label
        and any(term in label.lower() for term in ("cart", "grill", "wheel", "building", "door", "vehicle"))
        and any(term in action.lower() for term in ("breeze", "wind", "sway"))
    }
    usable = [
        (label, action)
        for label, action in candidates
        if label and not _is_static_motion_target(label) and label not in rigid_breeze
    ]
    sensitive_target = _important_sensitive_subject_label(analysis)
    sensitive_motion = next(((label, action) for label, action in usable if _labels_overlap(label, sensitive_target)), None)
    if sensitive_target:
        if sensitive_motion:
            usable = [sensitive_motion, *(item for item in usable if item != sensitive_motion)]
        else:
            usable.insert(0, (sensitive_target, ""))
    if not usable:
        observed = " ".join(_text(value).lower() for value in ((analysis.get("dynamic_potential") or {}).get("natural_motion_elements") or []))
        summary = _text(analysis.get("summary")).lower()
        if "snow" in summary and any(term in summary for term in ("tree", "forest")):
            usable = [("snow-laden outer boughs", "")]
        elif "foliage" in observed:
            usable = [("outer foliage", "")]
    primary = usable[0] if usable else ("", "")
    secondary = usable[1] if len(usable) > 1 else ("", "")
    stable = [_text(value) for value in (motion.get("keep_stable") or []) if _text(value)]
    moving_labels = [primary[0], secondary[0]]
    stable = [value for value in stable if not any(_labels_overlap(value, moving) for moving in moving_labels if moving)]
    rejected_physical = [
        label
        for label, _action in candidates
        if label and ("cactus" in label.lower() or "cacti" in label.lower() or "rock" in label.lower() or "terrain" in label.lower())
    ]
    stable.extend(value for value in rejected_physical if value not in stable)
    stable.extend(value for value in rigid_breeze if value not in stable)
    if primary[0] and any(term in primary[0].lower() for term in ("snow", "tree", "branch", "bough")):
        stable = ["tree trunks", "ground snow", "overall forest geometry"]
    if primary[0] and any(term in primary[0].lower() for term in ("popp", "flower", "blossom", "petal")):
        stable = ["rocks", "cacti", "tree trunks", "fixed terrain"]
    if not stable:
        stable = ["fixed terrain and structural geometry"]
    motion.update(
        {
            "primary_target": primary[0],
            "primary_action": _motion_action(primary[0], primary[1], analysis) if primary[0] else "",
            "secondary_target": secondary[0],
            "secondary_action": _motion_action(secondary[0], secondary[1], analysis) if secondary[0] else "",
            "keep_stable": stable[:10],
        }
    )
    normalized["motion_plan"] = motion
    presentation = normalized.get("presentation") if isinstance(normalized.get("presentation"), dict) else {}
    presentation["focus_target"] = primary[0] or requested_primary
    normalized["presentation"] = presentation
    generation = normalized.get("generation") if isinstance(normalized.get("generation"), dict) else {}
    instructions = [motion.get("primary_action"), motion.get("secondary_action")]
    stable_text = ", ".join(stable)
    generated_prompt = ". ".join(_text(value).rstrip(".") for value in instructions if _text(value))
    if generated_prompt:
        generated_prompt += f". Keep {stable_text} stable. Preserve the original composition and viewpoint."
        generation["prompt"] = generated_prompt[:420]
    normalized["generation"] = generation
    return normalized


def _crop_window(source_aspect: float, target_aspect: float, focus_x: float, focus_y: float) -> Dict[str, float]:
    width = min(1.0, target_aspect / source_aspect)
    height = min(1.0, source_aspect / target_aspect)
    x = max(0.0, min(1.0 - width, focus_x - width / 2.0))
    y = max(0.0, min(1.0 - height, focus_y - height / 2.0))
    return {"x": x, "y": y, "w": width, "h": height}


def _box_inside(box: Dict[str, float], window: Dict[str, float], margin: float = 0.002) -> bool:
    return (
        box["x"] >= window["x"] - margin
        and box["y"] >= window["y"] - margin
        and box["x"] + box["w"] <= window["x"] + window["w"] + margin
        and box["y"] + box["h"] <= window["y"] + window["h"] + margin
    )


def _region_aware_presentation(plan: Dict[str, Any], analysis: Dict[str, Any], aspect: str, operation: str) -> Dict[str, Any]:
    presentation = plan.get("presentation") if isinstance(plan.get("presentation"), dict) else {}
    motion = plan.get("motion_plan") if isinstance(plan.get("motion_plan"), dict) else {}
    regions = _observed_regions(analysis)
    focus_label = _text(presentation.get("focus_target")) or _text(motion.get("primary_target"))
    focus_region = _find_region(focus_label, regions)
    if operation in {"push_in", "pull_out"}:
        dynamic = analysis.get("dynamic_potential") if isinstance(analysis.get("dynamic_potential"), dict) else {}
        affordance_text = " ".join(_text(value).lower() for value in (dynamic.get("camera_motion_affordances") or []))
        affordance_region = next(
            (
                region
                for region in regions
                if _label_tokens(region.get("label"))
                and _label_tokens(region.get("label")) <= _label_tokens(affordance_text)
            ),
            None,
        )
        if affordance_region:
            focus_region = affordance_region
            focus_label = _text(affordance_region.get("label"))
    if focus_region is None:
        for label in presentation.get("must_keep_visible") or []:
            focus_region = _find_region(label, regions)
            if focus_region:
                focus_label = _text(label)
                break
    if not focus_region:
        return {}
    box = focus_region["box"]
    focus_x = box["x"] + box["w"] / 2.0
    focus_y = box["y"] + box["h"] / 2.0
    if operation in {"push_in", "pull_out"}:
        if any(term in focus_label.lower() for term in ("popp", "flower", "blossom", "petal")):
            zoom_end = 1.25
        elif _is_static_motion_target(focus_label):
            zoom_end = 1.15
        else:
            zoom_end = 1.15
    else:
        zoom_end = 1.0
    metadata = analysis.get("image_metadata") if isinstance(analysis.get("image_metadata"), dict) else {}
    try:
        source_aspect = float(metadata.get("aspect_ratio") or (float(metadata["width"]) / float(metadata["height"])))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        source_aspect = 1.0
    target_aspect = 1.0 if aspect == "square_1_1" else 9.0 / 16.0
    initial = _crop_window(source_aspect, target_aspect, focus_x, focus_y)
    final = dict(initial)
    pan_adjustment: Dict[str, float] = {}
    if operation in {"push_in", "pull_out"}:
        final = {
            "x": initial["x"] + initial["w"] * (1.0 - 1.0 / zoom_end) / 2.0,
            "y": initial["y"] + initial["h"] * (1.0 - 1.0 / zoom_end) / 2.0,
            "w": initial["w"] / zoom_end,
            "h": initial["h"] / zoom_end,
        }
    elif operation in {"pan_left_to_right", "pan_right_to_left"} and initial["w"] < 1.0:
        overflow = 1.0 - initial["w"]
        allowed_low = max(0.0, min(1.0, (box["x"] + box["w"] - initial["w"]) / overflow))
        allowed_high = max(0.0, min(1.0, box["x"] / overflow))
        requested_start = max(0.0, min(1.0, float(presentation.get("pan_start", 0.42))))
        requested_end = max(0.0, min(1.0, float(presentation.get("pan_end", 0.50))))
        start = max(allowed_low, min(allowed_high, requested_start))
        end = max(allowed_low, min(allowed_high, requested_end))
        if operation == "pan_left_to_right" and end < start:
            start, end = end, start
        if operation == "pan_right_to_left" and start < end:
            start, end = end, start
        initial = {"x": overflow * start, "y": initial["y"], "w": initial["w"], "h": initial["h"]}
        final = {"x": overflow * end, "y": initial["y"], "w": initial["w"], "h": initial["h"]}
        pan_adjustment = {"pan_start": round(start, 6), "pan_end": round(end, 6)}
    requested = [focus_label, *(_text(value) for value in (presentation.get("must_keep_visible") or []))]
    kept: list[Dict[str, Any]] = []
    excluded: list[str] = []
    seen: set[str] = set()
    for label in requested:
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        region = _find_region(label, regions)
        if region and _box_inside(region["box"], initial) and _box_inside(region["box"], final):
            kept.append({"label": label, "box": region["box"], "source": region["source"]})
        else:
            excluded.append(label)
    horizontal = "left" if focus_x < 0.4 else "right" if focus_x > 0.6 else "center"
    vertical = "top" if focus_y < 0.4 else "bottom" if focus_y > 0.6 else "center"
    result: Dict[str, Any] = {
        "zoom_focus_x": round(focus_x, 6),
        "zoom_focus_y": round(focus_y, 6),
        "crop_anchor": f"{horizontal}_{vertical}",
        "focus_region": {"label": focus_label, "box": box, "source": focus_region["source"]},
        "must_keep_visible": [item["label"] for item in kept],
        "required_regions": kept,
        "visibility_validation": {"status": "passed" if not excluded else "adjusted", "excluded": excluded, "initial_crop": initial, "final_crop": final},
    }
    if operation in {"push_in", "pull_out"}:
        result["zoom_end"] = zoom_end
    result.update(pan_adjustment)
    return result


def _opening_push_target(analysis: Dict[str, Any]) -> Dict[str, float] | None:
    """Find the distant focal point for a view framed by near foreground geometry."""
    composition = analysis.get("composition") if isinstance(analysis.get("composition"), dict) else {}
    layout = _text(composition.get("layout")).lower()
    descriptive_text = " ".join(
        [
            _text(analysis.get("summary")),
            _text(analysis.get("detailed_description")),
            *(_text(item) for item in (composition.get("foreground") or [])),
            *(
                f"{_text(item.get('label'))} {_text(item.get('description'))} {_text(item.get('reason'))}"
                for item in (composition.get("attention_regions") or [])
                if isinstance(item, dict)
            ),
        ]
    ).lower()
    through_cue = any(term in descriptive_text for term in ("viewed through", "visible through", "through the", "opening"))
    near_frame_cue = any(term in descriptive_text for term in ("foreground structure", "architectural frame", "wooden beam", "window frame", "natural frame"))
    if layout != "framed" or not through_cue or not near_frame_cue:
        return None

    focal_labels = {_text(item).lower() for item in (composition.get("focal_points") or []) if _text(item)}
    regions = (analysis.get("spatial_map") or {}).get("primary_regions") or []
    candidates: list[Dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        label = _text(region.get("label")).lower()
        if any(term in label for term in ("frame", "foreground structure", "beam", "support", "opening")):
            continue
        center = region.get("center") if isinstance(region.get("center"), dict) else {}
        try:
            x, y = float(center.get("x")), float(center.get("y"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            candidates.append({"label": label, "x": x, "y": y, "focal": label in focal_labels})
    if not candidates:
        return {"zoom_end": 2.2, "zoom_focus_x": 0.5, "zoom_focus_y": 0.5}
    target = next((item for item in candidates if item["focal"]), candidates[0])
    return {
        "zoom_end": 2.2,
        "zoom_focus_x": max(0.0, min(1.0, float(target["x"]))),
        "zoom_focus_y": max(0.0, min(1.0, float(target["y"]))),
    }


def _requires_panorama_traversal(analysis: Dict[str, Any]) -> bool:
    composition = analysis.get("composition") if isinstance(analysis.get("composition"), dict) else {}
    constraints = analysis.get("reframe_constraints") if isinstance(analysis.get("reframe_constraints"), dict) else {}
    return (
        _text(composition.get("layout")).lower() == "panoramic"
        and bool(constraints.get("wide_composition"))
        and bool(constraints.get("full_width_important_content"))
        and _text(constraints.get("vertical_crop_risk")).lower() == "high"
    )


def compile_semantic_plan(plan: Dict[str, Any], analysis: Dict[str, Any], source_sha256: str) -> Dict[str, Any]:
    plan = _normalize_motion_plan(plan, analysis)
    generation = plan.get("generation") if isinstance(plan.get("generation"), dict) else {}
    presentation = plan.get("presentation") if isinstance(plan.get("presentation"), dict) else {}
    classification = plan.get("classification") if isinstance(plan.get("classification"), dict) else {}
    backend = _enum_token(generation.get("backend"))
    important_sensitive_subject = _important_sensitive_subject(analysis)
    if important_sensitive_subject:
        backend = "hunyuan15"
        preset = "HUNYUAN15_I2V_720P"
    else:
        backend, preset = "wan22", "WAN22_NATURAL"

    prompt = _text(generation.get("prompt"))[:420]
    negative = _text(generation.get("negative_prompt"))[:220] or "flicker, jitter, unstable geometry, inconsistent appearance, scene transition, low quality"
    aspect = _enum_token(presentation.get("aspect"))
    if aspect not in {"instagram_reel_9_16", "square_1_1"}:
        aspect = "square_1_1"
    operation = _enum_token(presentation.get("operation"))
    if operation not in {"static", "pan_left_to_right", "pan_right_to_left", "push_in", "pull_out"}:
        operation = "static"
    panorama_traversal = _requires_panorama_traversal(analysis)
    if panorama_traversal:
        aspect = "square_1_1"
        operation = "pan_left_to_right"
    anchor = _enum_token(presentation.get("crop_anchor")) or "center_center"
    scene_text = " ".join(
        [_text(value).lower() for value in (classification.get("scene_classes") or [])]
        + [_text(classification.get("environment")).lower()]
    )
    affordances = " ".join(
        _text(value).lower()
        for value in ((analysis.get("dynamic_potential") or {}).get("camera_motion_affordances") or [])
    )
    if operation == "static" and any(term in scene_text for term in ("city", "urban", "architecture", "building", "skyline")):
        if "push" in affordances or not affordances or affordances == "none":
            operation = "push_in"
    opening_push = _opening_push_target(analysis)
    if opening_push:
        operation = "push_in"
        # A portrait cover crop already removes more of the wide source frame;
        # square delivery needs the stronger endpoint to cross the same opening.
        opening_push["zoom_end"] = 2.0 if aspect == "instagram_reel_9_16" else 2.2
    region_presentation = {} if panorama_traversal else _region_aware_presentation(plan, analysis, aspect, operation)
    if region_presentation.get("crop_anchor"):
        anchor = str(region_presentation.pop("crop_anchor"))
    start = 0.10 if panorama_traversal else max(0.0, min(1.0, float(presentation.get("pan_start", 0.42))))
    end = 0.80 if panorama_traversal else max(0.0, min(1.0, float(presentation.get("pan_end", 0.50))))
    max_pan_span = 0.70 if panorama_traversal else 0.18
    if abs(end - start) > max_pan_span:
        direction = 1 if end >= start else -1
        end = max(0.0, min(1.0, start + max_pan_span * direction))
    start, end = round(start, 6), round(end, 6)
    seed = _seed_base(source_sha256, backend)
    params: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative,
        "use_original_input_for_video": True,
        "preserve_source_aspect": True,
        "output_aspect": aspect,
        "final_crop_motion": operation,
        "pan_start": start,
        "pan_end": end,
        "pan_max_span": max_pan_span,
        "must_keep_visible": [] if panorama_traversal else presentation.get("must_keep_visible") or [],
    }
    if panorama_traversal:
        params["visibility_validation"] = {
            "status": "traversal",
            "reason": "high-risk full-width panorama is shown over time instead of forced into one portrait crop",
        }
    if region_presentation:
        params.update(region_presentation)
    if opening_push:
        params.update({"zoom_mode": "enter_frame", **opening_push})
    if preset == "WAN22_NATURAL":
        # The accepted 97-frame valley render used these native WAN settings
        # with a full VAE decode. Tiled decoding introduced a visible spatial
        # lattice in detailed vegetation in the semantic-v2 valley output.
        params.update({"steps": 20, "cfg": 5.0, "shift": 8.0, "sampler_name": "uni_pc", "scheduler": "simple", "tiled_vae": False})
        video = {"preset": preset, "duration_s": 5, "fps": 20, "frames": 97, "resolution_width": 768, "seed": seed, "params": params}
    elif preset.startswith("HUNYUAN15_"):
        # Full decoding of the 61-frame Hunyuan profile has restarted Comfy on
        # this host during the VAE handoff. Keep its independently validated
        # memory-safe decoder until a clean alternative is benchmarked.
        params.update({"steps": 50, "cfg": 6.0, "shift": 7.0, "tiled_vae": True})
        video = {"preset": preset, "duration_s": 5, "fps": 12, "frames": 61, "resolution_width": 704, "seed": seed, "params": params}
    else:
        video = {"preset": preset, "duration_s": 5, "fps": 30, "frames": 150, "resolution_width": 1080, "seed": seed, "params": params}

    count = max(1, min(3, int(generation.get("candidate_count", 2))))
    fallbacks: list[Dict[str, Any]] = []
    for index in range(1, min(count, 3)):
        candidate = json.loads(json.dumps(video))
        candidate["seed"] = (seed + index) & 0x7FFFFFFF
        candidate["params"]["seed"] = candidate["seed"]
        fallbacks.append(candidate)
    while len(fallbacks) < 2:
        fallback = json.loads(json.dumps(video))
        fallback["preset"] = "DETERMINISTIC_ORIGINAL"
        fallback["recovery_only"] = True
        fallback["fps"] = 30
        fallback["frames"] = 150
        fallback["resolution_width"] = 1080
        fallbacks.append(fallback)

    people = analysis.get("people") if isinstance(analysis.get("people"), list) else []
    confidence = analysis.get("confidence") if isinstance(analysis.get("confidence"), dict) else {}
    audio_plan = plan.get("audio") if isinstance(plan.get("audio"), dict) else {}
    decision = {
        "scene": {
            "tags": [_text(x).lower()[:50] for x in (classification.get("scene_classes") or [])[:8]],
            "has_people": bool(people),
            "confidence": float(confidence.get("overall", 0.7) or 0.7),
        },
        "framing": {"target_aspect": aspect, "crop_anchor": anchor},
        "video": video,
        "audio": {"prompt": _text(audio_plan.get("prompt"))[:180] or "quiet environmental ambience, no music", "duration_s": 5, "mix_db": -6.0},
        "fallbacks": fallbacks[:2],
        "runtime": {"render_variants": "all" if count > 1 else "selected"},
        "semantic_plan": plan,
    }
    return decision
