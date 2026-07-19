"""Small read-only HTTP facade over the local MemPalace search tool.

The video pipeline runs in containers while the palace is a host-local Chroma
store.  Keeping the MemPalace package and database access in this dedicated
service avoids coupling the orchestrator environment to Chroma/ONNX packages.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MEMPALACE_DISABLE_STDIO_REDIRECT", "1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mempalace.mcp_server import tool_search


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=250)
    wing: str | None = None
    room: str | None = None
    limit: int = Field(default=3, ge=1, le=10)
    max_distance: float = Field(default=1.2, ge=0.0, le=2.0)
    context: str | None = Field(default=None, max_length=4000)


app = FastAPI(title="image2video-mempalace", version="1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "palace_path": os.environ.get("MEMPALACE_PALACE_PATH", "default")}


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        result = tool_search(
            query=request.query,
            wing=request.wing,
            room=request.room,
            limit=request.limit,
            max_distance=request.max_distance,
            context=request.context,
        )
    except Exception as exc:  # pragma: no cover - backend-specific failures
        raise HTTPException(status_code=503, detail=f"MemPalace search failed: {exc}") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="MemPalace returned a non-object response")
    return result
