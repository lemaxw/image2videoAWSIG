from pathlib import Path
from typing import List

import boto3


class S3IO:
    def __init__(self, dry_run: bool = False, local_input_dir: str = "", local_output_dir: str = ""):
        self.dry_run = dry_run
        self.local_input_dir = Path(local_input_dir) if local_input_dir else None
        self.local_output_dir = Path(local_output_dir) if local_output_dir else None
        self.s3 = boto3.client("s3") if not dry_run else None

    def list_images(self, bucket: str, prefix: str) -> List[str]:
        if self.dry_run:
            assert self.local_input_dir is not None
            return [
                str(p.relative_to(self.local_input_dir))
                for p in self.local_input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]

        paginator = self.s3.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    keys.append(key)
        return keys

    def download_file(self, bucket: str, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            assert self.local_input_dir is not None
            src = self.local_input_dir / key
            local_path.write_bytes(src.read_bytes())
            return
        self.s3.download_file(bucket, key, str(local_path))

    def upload_file(self, local_path: Path, bucket: str, key: str, content_type: str | None = None) -> None:
        if self.dry_run:
            assert self.local_output_dir is not None
            dst = self.local_output_dir / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(local_path.read_bytes())
            return

        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            self.s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
        else:
            self.s3.upload_file(str(local_path), bucket, key)
