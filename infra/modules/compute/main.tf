locals {
  name = "${var.project}-${var.environment}"
  tags = {
    project   = var.project
    env       = var.environment
    ManagedBy = "Terraform"
  }
}

# Provider configuration and account restrictions belong to the environment root.
data "aws_partition" "current" {}

resource "aws_iam_role" "compute" {
  name        = "${local.name}-compute"
  description = "EC2 pipeline access to the lake and required secrets"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowEC2"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = ["sts:AssumeRole"]
    }]
  })

  tags = local.tags
}

# Create secret metadata only. An authorized operator populates its value
# separately; no API key is accepted by Terraform or placed in state/user_data.
# Omitting kms_key_id selects the AWS-managed aws/secretsmanager key.
resource "aws_secretsmanager_secret" "llm_api_key" {
  region                  = var.aws_region
  name                    = "${local.name}-llm-api-key"
  description             = "LLM API key for the ${var.environment} ALIGN pipeline"
  recovery_window_in_days = 30
  tags                    = local.tags
}

resource "aws_iam_role_policy" "data_access" {
  name = "${local.name}-data-access"
  role = aws_iam_role.compute.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      {
        Sid      = "ListLake"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [var.lake_bucket_arn]
      },
      {
        Sid      = "ReadWriteLakeObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${var.lake_bucket_arn}/*"]
      },
      {
        Sid      = "ReadRequiredSecrets"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [var.db_secret_arn, aws_secretsmanager_secret.llm_api_key.arn]
      },
      ], var.db_secret_kms_key_arn == null ? [] : [
      {
        Sid      = "ReadDatabaseSecretKey"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = [var.db_secret_kms_key_arn]
        Condition = {
          StringEquals = {
            "kms:ViaService"                  = "secretsmanager.${var.aws_region}.amazonaws.com"
            "kms:EncryptionContext:SecretARN" = var.db_secret_arn
          }
        }
      },
    ])
  })
}

# Session Manager is the access method specified by the network environment roots.
# Keep its service permissions separate from the scoped workload data policy.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.compute.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "compute" {
  name = "${local.name}-compute"
  role = aws_iam_role.compute.name
  tags = local.tags

  # EC2 must wait for its workload and management permissions before first boot.
  depends_on = [aws_iam_role_policy.data_access, aws_iam_role_policy.ecr_pull, aws_iam_role_policy_attachment.ssm]
}

# Pin an explicitly supplied AMI rather than silently following the latest image.
# Its owner and filters restrict the bootstrap to standard AL2023 x86_64 images.
data "aws_ami" "selected" {
  region = var.aws_region
  owners = ["amazon"]

  filter {
    name   = "image-id"
    values = [var.ami_id]
  }

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-kernel-*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

data "aws_ec2_instance_type" "selected" {
  region        = var.aws_region
  instance_type = var.instance_type
}

resource "aws_instance" "compute" {
  region                 = var.aws_region
  ami                    = data.aws_ami.selected.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.ec2_security_group_id]
  iam_instance_profile   = aws_iam_instance_profile.compute.name

  # The network module has no NAT and disables automatic public IP assignment.
  associate_public_ip_address = true

  user_data = local.bootstrap_user_data
  # user_data normally runs only at first launch; replace the instance when it
  # changes so a later approved deployment runs the updated bootstrap.
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
    # Permit IMDSv2 token responses across a Docker bridge for AWS SDK access.
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "disabled"
  }

  root_block_device {
    encrypted             = true
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    delete_on_termination = true
    tags                  = merge(local.tags, { Name = "${local.name}-compute-root" })
  }

  tags = merge(local.tags, { Name = "${local.name}-compute" })

  lifecycle {
    precondition {
      condition     = length(local.bootstrap_user_data) <= 16384
      error_message = "Rendered EC2 user_data exceeds 16 KiB; reduce runtime configuration or embedded assets."
    }

    precondition {
      condition = (
        data.aws_ami.selected.architecture == "x86_64" &&
        can(regex("^al2023-ami-2023\\..*-kernel-.*-x86_64$", data.aws_ami.selected.name)) &&
        data.aws_ami.selected.root_device_type == "ebs" &&
        data.aws_ami.selected.virtualization_type == "hvm"
      )
      error_message = "The compute bootstrap requires a standard Amazon Linux 2023 x86_64 HVM image with an EBS root disk."
    }

    precondition {
      condition = (
        contains(data.aws_ec2_instance_type.selected.supported_architectures, "x86_64") &&
        contains(data.aws_ec2_instance_type.selected.supported_root_device_types, "ebs") &&
        contains(data.aws_ec2_instance_type.selected.supported_virtualization_types, "hvm")
      )
      error_message = "Instance type must support x86_64, EBS root disks, and HVM virtualization."
    }

    precondition {
      condition = try(var.root_volume_size >= tonumber(one([
        for device in data.aws_ami.selected.block_device_mappings : device.ebs.volume_size
        if device.device_name == data.aws_ami.selected.root_device_name
      ])), false)
      error_message = "Root volume size must be at least as large as the selected AMI's root disk."
    }
  }
}

# Networking, lake storage, and RDS are supplied by their owning modules.
