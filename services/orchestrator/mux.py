import subprocess
from pathlib import Path


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path, mix_db: float = -8.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_copy = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        f"[1:a]volume={mix_db}dB[a1]",
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
        f"[1:a]volume={mix_db}dB[a1]",
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
