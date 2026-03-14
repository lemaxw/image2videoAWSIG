import subprocess
import os
from pathlib import Path


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path, mix_db: float = -8.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    audio_filter = f"[1:a]loudnorm=I={mux_target_lufs}:TP={mux_true_peak_db}:LRA=11,volume={effective_mix_db}dB[a1]"
    cmd_copy = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v:0",
        "-map",
        "[a1]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd_copy, check=True, capture_output=True, text=True)
        return
    except subprocess.CalledProcessError as exc:
        # Some generated mp4 files fail stream-copy muxing due to container/timebase quirks.
        stderr_copy = (exc.stderr or "").strip()

    cmd_reencode = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v:0",
        "-map",
        "[a1]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_reencode = (exc.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg mux failed for both copy and re-encode modes. "
            f"copy_stderr={stderr_copy[:1600]} reencode_stderr={stderr_reencode[:1600]}"
        ) from exc
