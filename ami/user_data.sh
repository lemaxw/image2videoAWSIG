#!/usr/bin/env bash
set -euo pipefail

exec > >(tee /var/log/image2video-user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

ARTIFACT_S3_URI="${artifact_s3_uri}"
AWS_REGION="${aws_region}"

export DEBIAN_FRONTEND=noninteractive

retry() {
  local tries="$1"
  shift
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [[ "$n" -ge "$tries" ]]; then
      return 1
    fi
    sleep 5
  done
}

apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release unzip jq awscli

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $$(. /etc/os-release && echo "$${VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if ! command -v nvidia-smi >/dev/null 2>&1; then
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers autoinstall || true
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

mkdir -p /data
if lsblk | grep -q nvme1n1; then
  DEV=/dev/nvme1n1
elif lsblk | grep -q xvdf; then
  DEV=/dev/xvdf
else
  DEV=""
fi

if [[ -n "$DEV" ]]; then
  if ! blkid "$DEV" >/dev/null 2>&1; then
    mkfs -t ext4 "$DEV"
  fi
  UUID=$$(blkid -s UUID -o value "$DEV")
  grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /data ext4 defaults,nofail 0 2" >> /etc/fstab
  mount -a
fi

mkdir -p /data/models /data/outputs /data/audio-cache /opt/image2video

if [[ -n "$ARTIFACT_S3_URI" ]]; then
  aws s3 cp "$ARTIFACT_S3_URI" /opt/image2video/repo.tgz --region "$AWS_REGION"
  tar -xzf /opt/image2video/repo.tgz -C /opt/image2video --strip-components=1
else
  echo "artifact_s3_uri is empty. Skipping repo download."
fi

cd /opt/image2video

if [[ -f services/comfy/docker-compose.yml ]]; then
  export MODEL_DIR=/data/models
  export OUTPUT_DIR=/data/outputs
  export AUDIO_CACHE_DIR=/data/audio-cache
  export OPENAI_MODEL="${openai_model}"
  export DEFAULT_INPUT_BUCKET="${input_bucket}"
  export DEFAULT_OUTPUT_BUCKET="${output_bucket}"

  retry 3 docker compose -f services/comfy/docker-compose.yml build
  retry 3 docker compose -f services/comfy/docker-compose.yml up -d
  bash services/comfy/scripts/wait_ready.sh
else
  echo "Compose file not found in artifact; orchestration services were not started."
fi
