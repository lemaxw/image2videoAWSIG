import math
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="audio-service")
# Reuse uvicorn logger so entries always show in `docker logs`.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

_PIPE = None
FORBIDDEN_PROMPT_WORDS = {"loud", "chaotic", "intense", "fast", "music", "song", "melody"}
TEXTURE_HINTS = ("soft", "warm", "airy", "dreamy", "gentle", "cinematic", "atmospheric", "ambient")
SOUND_WORDS = (
    "bird",
    "birds",
    "insect",
    "insects",
    "bug",
    "bugs",
    "wind",
    "traffic",
    "train",
    "trains",
    "cars",
    "car",
    "waves",
    "water",
    "ripples",
    "hum",
    "tone",
    "audience",
    "brass",
    "gulls",
    "breeze",
)

SCENE_SOUND_HINTS = (
    (("forest", "trees", "woods", "meadow", "field", "flower", "grass", "hills", "countryside"), "birds, insects, soft wind through grass"),
    (("city", "urban", "street", "skyline", "paris", "eiffel", "avenue", "rooftops"), "distant traffic, occasional train, city ambience"),
    (("night", "neon", "bridge", "lights"), "soft city hum, distant traffic, gentle air"),
    (("ocean", "sea", "shore", "waves", "beach"), "small waves, sea breeze, distant gulls"),
    (("lake", "river", "water", "reflection"), "gentle water ripples, birds, light wind"),
    (("orchestra", "concert", "stage", "musicians", "trombone"), "soft room tone, quiet audience, warm brass resonance"),
)


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
    if backend not in {"audioldm", "tangoflux"}:
        return None
    if _PIPE is not None:
        return _PIPE

    device_name = os.environ.get("AUDIO_DEVICE", "cpu")
    dtype = torch.float16 if device_name == "cuda" else torch.float32
    cache_dir = os.environ.get("AUDIO_CACHE_DIR", "/cache")
    logger.info("audio.backend.init backend=%s device=%s dtype=%s cache_dir=%s", backend, device_name, str(dtype), cache_dir)
    if backend == "tangoflux":
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from tangoflux import TangoFluxInference

        started = time.time()
        _PIPE = TangoFluxInference(name=os.environ.get("TANGOFLUX_MODEL", "declare-lab/TangoFlux"), device=device_name)
        logger.info("audio.backend.ready backend=tangoflux duration_s=%.3f", time.time() - started)
    else:
        from diffusers import AudioLDMPipeline

        started = time.time()
        _PIPE = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-s-full-v2",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        _PIPE = _PIPE.to(device_name)
        logger.info("audio.backend.ready backend=audioldm duration_s=%.3f", time.time() - started)
    return _PIPE


def _mock_audio(prompt: str, duration_s: int, out_path: Path) -> None:
    sample_rate = 16000
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    base = 180 + (abs(hash(prompt)) % 420)
    wave = 0.2 * np.sin(2 * math.pi * base * t)
    noise = 0.02 * np.random.randn(n)
    audio = np.clip(wave + noise, -1.0, 1.0)
    sf.write(str(out_path), audio, sample_rate, format="WAV", subtype="PCM_16")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_prompt(raw_prompt: str) -> str:
    text = " ".join(raw_prompt.strip().split())
    lowered = text.lower()
    parts = [p.strip() for p in text.split(",") if p.strip()]

    cleaned_parts: List[str] = []
    for part in parts:
        words = [w for w in part.split() if w.lower().strip(".,") not in FORBIDDEN_PROMPT_WORDS]
        p = " ".join(words).strip()
        if p:
            cleaned_parts.append(p)

    if not cleaned_parts:
        cleaned_parts = ["soft ambient atmosphere", "gentle textures"]

    inferred_hints = []
    for keywords, hint in SCENE_SOUND_HINTS:
        if any(keyword in lowered for keyword in keywords):
            inferred_hints.append(hint)
            break

    scene_parts = cleaned_parts[:2]
    sound_parts = [part for part in cleaned_parts[2:] if any(word in part.lower() for word in SOUND_WORDS)]

    useful_parts: List[str] = []
    for part in scene_parts + inferred_hints + sound_parts:
        if part.lower() not in [p.lower() for p in useful_parts]:
            useful_parts.append(part)

    has_texture = any(word in lowered for word in TEXTURE_HINTS)
    tail = "soft atmospheric ambience, realistic environmental soundscape"
    if has_texture:
        tail = "realistic environmental soundscape, no music"
    useful_parts.append(tail)
    return ", ".join(useful_parts[:6])[:260]


def _score_candidate(audio: np.ndarray) -> float:
    arr = _prepare_audio_array(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    arr = np.clip(arr, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
    clipping_ratio = float(np.mean(np.abs(arr) >= 0.999))
    return rms - (clipping_ratio * 2.0)


def _prepare_audio_array(audio: object) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio)

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=np.float32)
    elif arr.ndim == 2:
        # Reduce stereo/multichannel to mono to simplify downstream post-processing.
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1.0, 1.0)
    if arr.size == 0:
        raise RuntimeError("Audio candidate is empty after normalization")
    return arr


def _split_audio_candidates(audios: object) -> List[np.ndarray]:
    if isinstance(audios, list):
        raw_candidates = audios
    else:
        arr = audios.detach().float().cpu().numpy() if isinstance(audios, torch.Tensor) else np.asarray(audios)
        if arr.ndim >= 2 and arr.shape[0] > 1:
            raw_candidates = [arr[idx] for idx in range(arr.shape[0])]
        else:
            raw_candidates = [arr]
    return [_prepare_audio_array(candidate) for candidate in raw_candidates]


def _postprocess_with_ffmpeg(input_wav: Path, output_wav: Path, duration_s: int) -> None:
    if not input_wav.exists() or input_wav.stat().st_size < 64:
        raise RuntimeError(f"ffmpeg input wav missing or too small: {input_wav}")
    # Slightly louder default so short ambience remains audible after muxing.
    target_lufs = _env_float("AUDIO_TARGET_LUFS", -14.0)
    true_peak = _env_float("AUDIO_TRUE_PEAK_DB", -1.0)
    # ffmpeg alimiter.limit expects linear gain (0.0625..1.0), not dB.
    alimiter_limit = max(0.0625, min(1.0, float(10 ** (true_peak / 20.0))))
    bass_gain = _env_float("AUDIO_BASS_GAIN_DB", 0.0)
    stereo_widen = _env_float("AUDIO_STEREO_MLEV", 0.0)
    reverb_delay_ms = _env_int("AUDIO_REVERB_DELAY_MS", 700)
    reverb_decay = _env_float("AUDIO_REVERB_DECAY", 0.08)

    fade_in_d = 0.6
    fade_out_d = 1.0 if duration_s >= 5 else min(1.0, max(0.4, duration_s * 0.3))
    fade_out_start = max(0.0, duration_s - fade_out_d)

    full_filters = [
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11",
        f"alimiter=limit={alimiter_limit}",
    ]
    if abs(bass_gain) > 0.01:
        full_filters.append(f"bass=g={bass_gain}")
    if abs(stereo_widen) > 0.001:
        full_filters.append(f"stereotools=mlev={stereo_widen}")
    if reverb_decay > 0.001:
        full_filters.append(f"aecho=0.65:0.25:{reverb_delay_ms}:{reverb_decay}")
    full_filters.extend(
        [
            f"afade=t=in:st=0:d={fade_in_d}",
            f"afade=t=out:st={fade_out_start}:d={fade_out_d}",
            "aresample=48000",
            "aformat=sample_rates=48000:channel_layouts=stereo",
        ]
    )
    full_filter_chain = ",".join(full_filters)
    # Minimal safe chain when optional filters are unavailable in a specific ffmpeg build.
    safe_filter_chain = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11,"
        f"afade=t=in:st=0:d={fade_in_d},"
        f"afade=t=out:st={fade_out_start}:d={fade_out_d},"
        "aresample=48000,"
        "aformat=sample_rates=48000:channel_layouts=stereo"
    )

    def _run_chain(filter_chain: str, label: str) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            filter_chain,
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            err_tail = "\n".join(stderr.splitlines()[-60:])
            out_tail = "\n".join(stdout.splitlines()[-20:])
            raise RuntimeError(
                "ffmpeg audio post-process failed: "
                f"mode={label} rc={exc.returncode} stderr_tail={err_tail[:3200]} stdout_tail={out_tail[:1200]}"
            ) from exc

    try:
        _run_chain(full_filter_chain, "full")
    except Exception as full_exc:
        logger.warning("audio.postprocess.retry_with_safe_chain reason=%s", str(full_exc))
        _run_chain(safe_filter_chain, "safe")


def _generate_audioldm_processed(prompt: str, duration_s: int, out_path: Path) -> None:
    pipe = _load_backend()
    device_name = os.environ.get("AUDIO_DEVICE", "cpu")
    steps = max(10, _env_int("AUDIO_INFERENCE_STEPS", 60))
    guidance_scale = max(1.0, _env_float("AUDIO_GUIDANCE_SCALE", 3.5))
    num_samples = max(1, min(6, _env_int("AUDIO_NUM_SAMPLES", 3)))
    base_seed = _env_int("AUDIO_SEED_BASE", 42)

    generation_prompt = _normalize_prompt(prompt)
    logger.info(
        "audio.generate.config model=cvssp/audioldm-s-full-v2 prompt=%s duration_s=%s steps=%s guidance_scale=%s num_samples=%s",
        generation_prompt[:180],
        duration_s,
        steps,
        guidance_scale,
        num_samples,
    )

    with torch.inference_mode():
        generator = torch.Generator(device=device_name).manual_seed(base_seed)
        result = pipe(
            generation_prompt,
            audio_length_in_s=duration_s,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_waveforms_per_prompt=num_samples,
            generator=generator,
        )

    audios = _split_audio_candidates(result.audios)
    best_audio = None
    best_score = -1e9
    for idx, audio in enumerate(audios):
        score = _score_candidate(audio)
        logger.info("audio.generate.candidate idx=%s score=%.6f", idx, score)
        if score > best_score:
            best_score = score
            best_audio = audio
    if best_audio is None:
        raise RuntimeError("AudioLDM generated no audio candidates")

    with tempfile.TemporaryDirectory(prefix="audio-gen-") as td:
        raw_wav = Path(td) / "raw.wav"
        arr = _prepare_audio_array(best_audio)
        sf.write(str(raw_wav), arr, 16000, format="WAV", subtype="PCM_16")
        _postprocess_with_ffmpeg(raw_wav, out_path, duration_s)


def _generate_tangoflux_processed(prompt: str, duration_s: int, out_path: Path) -> None:
    model = _load_backend()
    steps = max(10, _env_int("TANGOFLUX_STEPS", _env_int("AUDIO_INFERENCE_STEPS", 50)))
    # TangoFlux's unguided path is broken in the published package version.
    guidance_scale = max(1.1, _env_float("TANGOFLUX_GUIDANCE_SCALE", 4.5))
    generation_prompt = _normalize_prompt(prompt)
    logger.info(
        "audio.generate.config model=declare-lab/TangoFlux prompt=%s duration_s=%s steps=%s guidance_scale=%s",
        generation_prompt[:220],
        duration_s,
        steps,
        guidance_scale,
    )

    with torch.inference_mode():
        started = time.time()
        audio = model.generate(
            generation_prompt,
            steps=steps,
            duration=duration_s,
            guidance_scale=guidance_scale,
        )
        logger.info("audio.generate.model_done backend=tangoflux duration_s=%.3f", time.time() - started)

    with tempfile.TemporaryDirectory(prefix="audio-tangoflux-") as td:
        raw_wav = Path(td) / "raw.wav"
        arr = _prepare_audio_array(audio)
        sf.write(str(raw_wav), arr, 44100, format="WAV", subtype="PCM_16")
        _postprocess_with_ffmpeg(raw_wav, out_path, duration_s)


@app.get("/health")
def health():
    return {"ok": True}


@app.on_event("startup")
def _on_startup():
    backend = os.environ.get("AUDIO_MODEL_BACKEND", "mock").lower()
    device_name = os.environ.get("AUDIO_DEVICE", "cpu")
    logger.info("audio.startup backend=%s device=%s", backend, device_name)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    duration_s = 3 if req.duration_s == 3 else 5
    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"audio-{uuid.uuid4().hex[:10]}.wav"

    backend = os.environ.get("AUDIO_MODEL_BACKEND", "mock").lower()
    logger.info(
        "audio.generate.start backend=%s duration_s=%s output=%s prompt=%s",
        backend,
        duration_s,
        str(wav_path),
        req.prompt[:120],
    )
    if backend in {"audioldm", "tangoflux"}:
        try:
            if backend == "tangoflux":
                _generate_tangoflux_processed(req.prompt, duration_s, wav_path)
                backend_name = "tangoflux-processed"
            else:
                _generate_audioldm_processed(req.prompt, duration_s, wav_path)
                backend_name = "audioldm-processed"
            logger.info("audio.generate.done backend=%s output=%s", backend_name, str(wav_path))
            return GenerateResponse(wav_path=str(wav_path), backend=backend_name)
        except Exception as exc:
            logger.exception("audio.generate.fallback reason=%s", str(exc))
            with tempfile.TemporaryDirectory(prefix="audio-fallback-") as td:
                raw_wav = Path(td) / "raw.wav"
                _mock_audio(req.prompt, duration_s, raw_wav)
                try:
                    _postprocess_with_ffmpeg(raw_wav, wav_path, duration_s)
                    backend_name = "mock-fallback"
                except Exception as post_exc:
                    logger.exception("audio.generate.mock_fallback_postprocess_failed reason=%s", str(post_exc))
                    shutil.copy2(raw_wav, wav_path)
                    backend_name = "mock-fallback-raw"
            logger.info("audio.generate.done backend=%s output=%s", backend_name, str(wav_path))
            return GenerateResponse(wav_path=str(wav_path), backend=backend_name)

    with tempfile.TemporaryDirectory(prefix="audio-mock-") as td:
        raw_wav = Path(td) / "raw.wav"
        _mock_audio(req.prompt, duration_s, raw_wav)
        try:
            _postprocess_with_ffmpeg(raw_wav, wav_path, duration_s)
            backend_name = "mock"
        except Exception as exc:
            logger.exception("audio.generate.mock_postprocess_failed reason=%s", str(exc))
            shutil.copy2(raw_wav, wav_path)
            backend_name = "mock-raw"
    logger.info("audio.generate.done backend=%s output=%s", backend_name, str(wav_path))
    return GenerateResponse(wav_path=str(wav_path), backend=backend_name)
