"""Local image analysis and semantic video decision service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from services.decision.semantic_planner import (
    SEMANTIC_DECISION_SCHEMA,
    SEMANTIC_SYSTEM_PROMPT,
    compile_semantic_plan,
    retrieve_similar_experience,
    sanitize_analysis_for_decision,
)
from services.orchestrator.validate import validate_and_clamp_decision

try:
    from image2json.analyzer import ImageAnalyzer
    from image2json.config import AnalysisConfig

    IMAGE2JSON_AVAILABLE = True
    IMAGE2JSON_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    IMAGE2JSON_AVAILABLE = False
    IMAGE2JSON_IMPORT_ERROR = exc


def _decision_step_log(event: str, step: str, **fields: Any) -> None:
    payload = {
        "level": "INFO" if event != "failed" else "ERROR",
        "msg": f"decision.{step}.{event}",
        "step": step,
        "event": event,
        "time": int(time.time()),
        **fields,
    }
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


def _call_ollama_text_model(
    ollama_url: str,
    model: str,
    prompt: str,
    json_input: Dict[str, Any],
    schema: Dict[str, Any],
    timeout: float = 300,
) -> str:
    full_prompt = f"""{prompt}

JSON Schema for output:
{json.dumps(schema, indent=2)}

Decision input JSON:
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
    return str(result.get("response", ""))


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def decide_for_image_detailed(image_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the mandatory local image2json -> memory -> semantic-plan flow."""
    metadata = metadata or {}
    if os.environ.get("IMAGE2JSON_ENABLED", "true").lower() != "true":
        raise RuntimeError("IMAGE2JSON_ENABLED must be true; no alternate decision backend is supported")
    if not IMAGE2JSON_AVAILABLE:
        detail = (
            f"{IMAGE2JSON_IMPORT_ERROR.__class__.__name__}: {IMAGE2JSON_IMPORT_ERROR}"
            if IMAGE2JSON_IMPORT_ERROR is not None
            else "unknown import error"
        )
        raise RuntimeError(f"image2json package could not be imported ({detail})")

    ollama_url = os.environ.get("IMAGE2JSON_URL", "http://host.docker.internal:11434")
    vision_model = os.environ.get("IMAGE2JSON_MODEL", "qwen3-vl:8b")
    text_model = os.environ.get("IMAGE2JSON_TEXT_MODEL", "qwen3:14b")
    timeout = float(os.environ.get("IMAGE2JSON_TIMEOUT", "300"))
    if vision_model == text_model or "vl" not in vision_model.lower():
        raise RuntimeError(
            "IMAGE2JSON_MODEL must be the vision model qwen3-vl:8b; "
            f"got IMAGE2JSON_MODEL={vision_model!r}, IMAGE2JSON_TEXT_MODEL={text_model!r}"
        )

    vision_started = time.time()
    _decision_step_log(
        "start",
        "image2json_vision",
        model=vision_model,
        url=ollama_url,
        image_path=str(image_path),
        timeout_s=timeout,
    )
    try:
        analyzer = ImageAnalyzer(
            AnalysisConfig(
                model=vision_model,
                ollama_url=ollama_url,
                timeout=timeout,
                short_version=False,
            )
        )
        analysis = analyzer.analyze_path(image_path)
        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis
        if not isinstance(analysis_dict, dict):
            raise TypeError(f"image2json returned {analysis_dict.__class__.__name__}, expected object")
        analysis_dict = sanitize_analysis_for_decision(analysis_dict)
        _decision_step_log(
            "done",
            "image2json_vision",
            model=vision_model,
            duration_s=round(time.time() - vision_started, 3),
            result={"analysis_type": analysis.__class__.__name__},
        )
    except Exception as exc:
        _decision_step_log(
            "failed",
            "image2json_vision",
            model=vision_model,
            duration_s=round(time.time() - vision_started, 3),
            error=str(exc),
            error_type=exc.__class__.__name__,
        )
        raise
    finally:
        _ollama_unload_model(ollama_url, vision_model, reason="after_image2json_vision")

    memory_started = time.time()
    _decision_step_log("start", "mempalace_search", wing=os.environ.get("MEMPALACE_WING", "image2videoAWSIG"))
    memory_meta = retrieve_similar_experience(analysis_dict)
    if memory_meta.get("error"):
        _decision_step_log(
            "failed",
            "mempalace_search",
            duration_s=round(time.time() - memory_started, 3),
            error=memory_meta.get("error"),
            error_type=memory_meta.get("error_type"),
            query=memory_meta.get("query"),
        )
    else:
        _decision_step_log(
            "done",
            "mempalace_search",
            duration_s=round(time.time() - memory_started, 3),
            result={"query": memory_meta.get("query"), "case_count": len(memory_meta.get("cases") or [])},
        )

    text_started = time.time()
    _decision_step_log(
        "start",
        "decision_text_model",
        model=text_model,
        url=ollama_url,
        analysis_model=vision_model,
        timeout_s=timeout,
    )
    try:
        text_response = _call_ollama_text_model(
            ollama_url=ollama_url,
            model=text_model,
            prompt=SEMANTIC_SYSTEM_PROMPT,
            json_input={
                "image2json": analysis_dict,
                "similar_cases": memory_meta.get("cases") or [],
                "request_metadata": metadata,
            },
            schema=SEMANTIC_DECISION_SCHEMA,
            timeout=timeout,
        )
        semantic_plan = json.loads(_strip_json_fences(text_response))
        if not isinstance(semantic_plan, dict):
            raise TypeError("decision text model returned non-object JSON")
        source_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        compiled = compile_semantic_plan(semantic_plan, analysis_dict, source_sha256)
        normalized_semantic_plan = compiled.get("semantic_plan") or semantic_plan
        runtime = compiled.get("runtime")
        decision = validate_and_clamp_decision(compiled)
        if isinstance(runtime, dict):
            decision["runtime"] = runtime
        decision["semantic_plan"] = normalized_semantic_plan
        _decision_step_log(
            "done",
            "decision_text_model",
            model=text_model,
            duration_s=round(time.time() - text_started, 3),
            result={
                "preset": (decision.get("video") or {}).get("preset"),
                "fallbacks": [item.get("preset") for item in decision.get("fallbacks", []) if isinstance(item, dict)],
                "memory_cases": len(memory_meta.get("cases") or []),
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
        "decision": decision,
        "decision_engine": {
            "type": "local_image2json_ollama",
            "vision_model": vision_model,
            "text_model": text_model,
        },
        "image2json": {
            "enabled": True,
            "used": True,
            "model": vision_model,
            "vision_model": vision_model,
            "text_model": text_model,
            "analysis": analysis_dict,
            "text_response": text_response,
        },
        "mempalace": memory_meta,
        "semantic_plan": normalized_semantic_plan,
    }
