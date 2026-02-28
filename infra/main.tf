terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sfn_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_s3_bucket" "input" {
  count  = var.create_buckets ? 1 : 0
  bucket = var.input_bucket_name
}

resource "aws_s3_bucket" "output" {
  count  = var.create_buckets ? 1 : 0
  bucket = var.output_bucket_name
}

locals {
  input_bucket  = var.create_buckets ? aws_s3_bucket.input[0].bucket : var.input_bucket_name
  output_bucket = var.create_buckets ? aws_s3_bucket.output[0].bucket : var.output_bucket_name
  artifact_object_arn = var.artifact_s3_uri != "" ? replace(var.artifact_s3_uri, "s3://", "arn:aws:s3:::") : ""
  artifact_bucket_arn = var.artifact_s3_uri != "" ? "arn:aws:s3:::${split("/", replace(var.artifact_s3_uri, "s3://", ""))[0]}" : ""
}

resource "aws_cloudwatch_log_group" "stepfn" {
  name              = "/aws/vendedlogs/states/${var.name_prefix}-pipeline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "ssm" {
  name              = "/aws/ssm/${var.name_prefix}-orchestrator"
  retention_in_days = 14
}

resource "aws_iam_role" "ec2" {
  name               = "${var.name_prefix}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ec2_ssm_core" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "ec2_custom" {
  name = "${var.name_prefix}-ec2-custom"
  role = aws_iam_role.ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = concat(
          [
            "arn:aws:s3:::${local.input_bucket}",
            "arn:aws:s3:::${local.input_bucket}/*",
            "arn:aws:s3:::${local.output_bucket}",
            "arn:aws:s3:::${local.output_bucket}/*"
          ],
          var.artifact_s3_uri != "" ? [local.artifact_object_arn] : []
        )
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = concat(
          [
            "arn:aws:s3:::${local.input_bucket}",
            "arn:aws:s3:::${local.output_bucket}"
          ],
          var.artifact_s3_uri != "" ? [local.artifact_bucket_arn] : []
        )
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.name_prefix}-ec2-profile"
  role = aws_iam_role.ec2.name
}

resource "aws_security_group" "gpu" {
  name        = "${var.name_prefix}-gpu-sg"
  description = "No public inbound, egress-only"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_launch_template" "gpu" {
  name_prefix   = "${var.name_prefix}-gpu-"
  image_id      = var.ami_id
  instance_type = var.instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.ec2.arn
  }

  network_interfaces {
    subnet_id                   = var.subnet_id
    associate_public_ip_address = false
    security_groups             = [aws_security_group.gpu.id]
  }

  user_data = base64encode(templatefile("${path.module}/../ami/user_data.sh", {
    artifact_s3_uri = var.artifact_s3_uri
    aws_region      = var.aws_region
    name_prefix     = var.name_prefix
    openai_model    = var.openai_model
    input_bucket    = local.input_bucket
    output_bucket   = local.output_bucket
  }))

  monitoring {
    enabled = true
  }

  block_device_mappings {
    device_name = "/dev/sdf"

    ebs {
      volume_size           = var.gpu_ebs_size_gb
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_tokens = "required"
  }

  tag_specifications {
    resource_type = "instance"

    tags = {
      ManagedBy = "terraform"
      Service   = "image2video"
    }
  }
}

resource "aws_iam_role" "step_functions" {
  name               = "${var.name_prefix}-sfn-role"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume_role.json
}

resource "aws_iam_role_policy" "step_functions" {
  name = "${var.name_prefix}-sfn-policy"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:TerminateInstances",
          "ec2:DescribeInstances",
          "ec2:DescribeLaunchTemplates",
          "ec2:DescribeLaunchTemplateVersions"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:SendCommand",
          "ssm:GetCommandInvocation",
          "ssm:DescribeInstanceInformation"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.ec2.arn
      }
    ]
  })
}

resource "aws_sfn_state_machine" "batch_pipeline" {
  name     = "${var.name_prefix}-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = templatefile("${path.module}/step_function.asl.json", {
    launch_template_id = aws_launch_template.gpu.id
    ssm_log_group      = aws_cloudwatch_log_group.ssm.name
    max_fail_ratio     = var.max_fail_ratio
  })

  logging_configuration {
    level                  = "ALL"
    include_execution_data = true
    log_destination        = "${aws_cloudwatch_log_group.stepfn.arn}:*"
  }
}
