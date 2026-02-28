import subprocess
from pathlib import Path


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path, mix_db: float = -8.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    subprocess.run(cmd, check=True, capture_output=True)
