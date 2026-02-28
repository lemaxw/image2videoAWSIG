import math
import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="audio-service")

_PIPE = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    duration_s: int = Field(default=5)
    output_dir: str = Field(default="/outputs/audio")


class GenerateResponse(BaseModel):
    wav_path: str
    backend: str


def _load_backend():
    global _PIPE
    backend = os.environ.get("AUDIO_MODEL_BACKEND", "mock").lower()
    if backend != "audioldm":
        return None
    if _PIPE is not None:
        return _PIPE

    from diffusers import AudioLDMPipeline

    device_name = os.environ.get("AUDIO_DEVICE", "cpu")
    dtype = torch.float16 if device_name == "cuda" else torch.float32
    _PIPE = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=dtype)
    _PIPE = _PIPE.to(device_name)
    return _PIPE


def _mock_audio(prompt: str, duration_s: int, out_path: Path) -> None:
    sample_rate = 16000
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    base = 180 + (abs(hash(prompt)) % 420)
    wave = 0.2 * np.sin(2 * math.pi * base * t)
    noise = 0.02 * np.random.randn(n)
    audio = np.clip(wave + noise, -1.0, 1.0)
    sf.write(out_path, audio, sample_rate)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    duration_s = 3 if req.duration_s == 3 else 5
    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"audio-{uuid.uuid4().hex[:10]}.wav"

    backend = os.environ.get("AUDIO_MODEL_BACKEND", "mock").lower()
    if backend == "audioldm":
        try:
            pipe = _load_backend()
            device_name = os.environ.get("AUDIO_DEVICE", "cpu")
            generator = torch.Generator(device=device_name).manual_seed(42)
            result = pipe(req.prompt, audio_length_in_s=duration_s, num_inference_steps=20, generator=generator)
            audio = result.audios[0]
            sf.write(wav_path, audio, 16000)
            return GenerateResponse(wav_path=str(wav_path), backend="audioldm")
        except Exception:
            _mock_audio(req.prompt, duration_s, wav_path)
            return GenerateResponse(wav_path=str(wav_path), backend="mock-fallback")

    _mock_audio(req.prompt, duration_s, wav_path)
    return GenerateResponse(wav_path=str(wav_path), backend="mock")
