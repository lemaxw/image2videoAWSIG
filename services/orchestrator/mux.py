import subprocess
import os
from pathlib import Path


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _final_video_fps() -> int:
    try:
        return _clamp_int(int(os.environ.get("FINAL_VIDEO_FPS", "30")), 1, 60)
    except ValueError:
        return 30


def _final_video_interpolation() -> str:
    mode = os.environ.get("FINAL_VIDEO_INTERPOLATION", "off").strip().lower()
    if mode in {"off", "none", "duplicate", "fps", "minterpolate"}:
        return mode
    return "off"


def _smoothness_filter(output_fps: int, interpolation: str) -> str:
    if interpolation == "minterpolate":
        return f"minterpolate=fps={output_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    if interpolation in {"duplicate", "fps"}:
        return f"fps={output_fps}"
    return ""


def _normalized_video_filter(
    video_fit: str = "contain",
    pan_start: float = 0.0,
    pan_end: float = 1.0,
    output_aspect: str = "instagram_reel_9_16",
    output_fps: int = 30,
    interpolation: str = "off",
    target_duration_s: float = 5.0,
) -> str:
    if output_aspect == "square_1_1":
        out_w = 1080
        out_h = 1080
    else:
        out_w = 1080
        out_h = 1920

    target_duration_s = _clamp_float(float(target_duration_s), 1.0, 30.0)
    smoothness = _smoothness_filter(output_fps, interpolation) or f"fps={output_fps}"
    duration_filter = f",tpad=stop_mode=clone:stop_duration={target_duration_s:.3f},trim=duration={target_duration_s:.3f},setpts=PTS-STARTPTS"
    suffix = f"{duration_filter},{smoothness},setsar=1,format=yuv420p"

    if video_fit == "cover":
        return (
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}"
            f"{suffix}"
        )
    if video_fit in {"pan_left_to_right", "pan_right_to_left"}:
        # Scale wide video to target height, then animate the final crop across it.
        # This is for panoramic originals where a static portrait crop loses key content.
        pan_start = _clamp_float(float(pan_start), 0.0, 1.0)
        pan_end = _clamp_float(float(pan_end), 0.0, 1.0)
        if video_fit == "pan_right_to_left" and pan_start < pan_end:
            pan_start, pan_end = pan_end, pan_start
        if video_fit == "pan_left_to_right" and pan_end < pan_start:
            pan_start, pan_end = pan_end, pan_start
        x_expr = f"(iw-ow)*({pan_start:.4f}+(({pan_end:.4f}-{pan_start:.4f})*min(t\\,{target_duration_s:.3f})/{target_duration_s:.3f}))"
        if video_fit == "pan_right_to_left":
            x_expr = f"(iw-ow)*({pan_start:.4f}+(({pan_end:.4f}-{pan_start:.4f})*min(t\\,{target_duration_s:.3f})/{target_duration_s:.3f}))"
        return (
            f"scale=-2:{out_h},"
            f"crop={out_w}:{out_h}:x='{x_expr}':y=0"
            f"{suffix}"
        )
    # Preserve generated composition, then pad to exact 9:16 for Instagram.
    return (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
        f"{suffix}"
    )


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    mix_db: float = -8.0,
    video_fit: str = "contain",
    pan_start: float = 0.0,
    pan_end: float = 1.0,
    output_aspect: str = "instagram_reel_9_16",
    target_duration_s: float = 5.0,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = _final_video_fps()
    interpolation = _final_video_interpolation()
    try:
        gain_boost_db = float(os.environ.get("AUDIO_MUX_GAIN_DB", "3.0"))
    except ValueError:
        gain_boost_db = 3.0
    try:
        mux_target_lufs = float(os.environ.get("AUDIO_MUX_TARGET_LUFS", "-12.0"))
    except ValueError:
        mux_target_lufs = -12.0
    try:
        mux_true_peak_db = float(os.environ.get("AUDIO_MUX_TRUE_PEAK_DB", "-1.0"))
    except ValueError:
        mux_true_peak_db = -1.0
    effective_mix_db = max(-24.0, min(12.0, float(mix_db) + gain_boost_db))
    target_duration_s = _clamp_float(float(target_duration_s), 1.0, 30.0)
    filter_complex = (
        f"[0:v]{_normalized_video_filter(video_fit, pan_start=pan_start, pan_end=pan_end, output_aspect=output_aspect, output_fps=output_fps, interpolation=interpolation, target_duration_s=target_duration_s)}[v0];"
        f"[1:a]loudnorm=I={mux_target_lufs}:TP={mux_true_peak_db}:LRA=11,"
        f"volume={effective_mix_db}dB,"
        f"apad=pad_dur={target_duration_s:.3f},atrim=duration={target_duration_s:.3f},asetpts=PTS-STARTPTS[a1]"
    )

    cmd_reencode = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v0]",
        "-map",
        "[a1]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-r",
        str(output_fps),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_reencode = (exc.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg mux failed during Instagram Reel normalization. "
            f"stderr={stderr_reencode[:1600]}"
        ) from exc
    return {
        "final_video_fps": output_fps,
        "final_video_interpolation": interpolation,
        "target_duration_s": target_duration_s,
    }


def export_video_frame_image(video_path: Path, output_path: Path, timestamp_s: float = 0.5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp_s),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg frame export failed. stderr={stderr[:1600]}") from exc
