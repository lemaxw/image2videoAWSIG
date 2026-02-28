# AWS Image-to-Video Batch Pipeline (MVP)

End-to-end CLI-deployable pipeline that:
- Reads images from S3
- Uses OpenAI vision decision JSON (structured outputs)
- Renders via ComfyUI on GPU EC2
- Generates audio via local FastAPI audio service
- Muxes audio + video with ffmpeg
- Uploads `final.mp4` + `debug.json` back to S3
- Orchestrates with AWS Step Functions and always terminates GPU instance

## Repo layout

- `infra/`: Terraform + Step Functions ASL
- `ami/user_data.sh`: host bootstrap (Docker, NVIDIA toolkit, EBS mount, compose up)
- `services/decision/decision_service.py`: OpenAI structured decision
- `services/orchestrator/`: batch runner and helpers
- `services/audio/audio_service.py`: audio generation service (AudioLDM option + CPU/mock fallback)
- `services/comfy/`: compose stack + workflow templates + health scripts

## Prereqs

- Terraform >= 1.5
- AWS CLI configured
- Existing VPC/subnet
- An AMI suitable for GPU instances (Ubuntu recommended)
- OpenAI API key available to orchestrator container (`OPENAI_API_KEY`)

## 1) Deploy infra

Create `infra/terraform.tfvars`:

```hcl
aws_region        = "us-east-1"
name_prefix       = "img2vid"
vpc_id            = "vpc-xxxxxxxx"
subnet_id         = "subnet-xxxxxxxx"
ami_id            = "ami-xxxxxxxx"
create_buckets    = false
input_bucket_name = "my-input-bucket"
output_bucket_name = "my-output-bucket"

# Upload this repo tarball and set URI so user_data can bootstrap host
artifact_s3_uri   = "s3://my-artifacts/image2videoAWSIG.tgz"
```

Package and upload artifact:

```bash
tar -czf /tmp/image2videoAWSIG.tgz .
aws s3 cp /tmp/image2videoAWSIG.tgz s3://my-artifacts/image2videoAWSIG.tgz
```

Deploy:

```bash
cd infra
terraform init
terraform apply
```

Capture output:

```bash
terraform output -raw state_machine_arn
```

## 2) Upload input images

Expected run input includes `input_prefix`, for example `jobs/demo-001/input`:

```bash
aws s3 cp ./my_images s3://my-input-bucket/jobs/demo-001/input/ --recursive
```

## 3) Start a Step Functions run

```bash
aws stepfunctions start-execution \
  --state-machine-arn "$(cd infra && terraform output -raw state_machine_arn)" \
  --name "demo-001-$(date +%s)" \
  --input '{
    "job_id":"demo-001",
    "input_bucket":"my-input-bucket",
    "input_prefix":"jobs/demo-001/input",
    "output_bucket":"my-output-bucket",
    "output_prefix":"jobs/demo-001/output"
  }'
```

## 4) View outputs

- Final videos:
  - `s3://<output_bucket>/<output_prefix>/<job_id>/<image_basename>/final.mp4`
- Debug payload:
  - `s3://<output_bucket>/<output_prefix>/<job_id>/<image_basename>/debug.json`

Example:

```bash
aws s3 ls s3://my-output-bucket/jobs/demo-001/output/demo-001/ --recursive
```

## Dry run (no AWS)

Run orchestrator locally against folders:

```bash
python services/orchestrator/run_batch.py \
  --job-id local-demo \
  --input-prefix . \
  --output-prefix out \
  --dry-run \
  --local-input-dir ./local_inputs \
  --local-output-dir ./local_outputs
```

## Notes

- `audio_service` supports `AUDIO_MODEL_BACKEND=audioldm` if model dependencies are available.
- Default backend is `mock` for quick MVP runs.
- ComfyUI templates are low-memory oriented and support runtime parameter injection.
