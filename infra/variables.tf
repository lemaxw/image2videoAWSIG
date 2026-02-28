variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Name prefix for resources"
  type        = string
  default     = "img2vid"
}

variable "vpc_id" {
  description = "VPC where GPU instance runs"
  type        = string
}

variable "subnet_id" {
  description = "Subnet for GPU instance"
  type        = string
}

variable "ami_id" {
  description = "AMI for GPU instance"
  type        = string
}

variable "instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "g5.xlarge"
}

variable "gpu_ebs_size_gb" {
  description = "Size of attached EBS volume for model/output cache"
  type        = number
  default     = 300
}

variable "create_buckets" {
  description = "Whether to create input/output S3 buckets"
  type        = bool
  default     = false
}

variable "input_bucket_name" {
  description = "Existing input bucket name (or name for created input bucket)"
  type        = string
}

variable "output_bucket_name" {
  description = "Existing output bucket name (or name for created output bucket)"
  type        = string
}

variable "artifact_s3_uri" {
  description = "S3 URI to tar.gz of this repo for instance bootstrap"
  type        = string
  default     = ""
}

variable "openai_model" {
  description = "OpenAI model used by decision step"
  type        = string
  default     = "gpt-4.1-mini"
}

variable "max_fail_ratio" {
  description = "Maximum failed-image ratio before orchestrator exits non-zero"
  type        = number
  default     = 0.3
}
