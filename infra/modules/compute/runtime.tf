resource "aws_ecr_repository" "arbiter" {
  count                = var.existing_ecr_repository_name == null ? 1 : 0
  region               = var.aws_region
  name                 = "${local.name}-arbiter"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  tags                 = local.tags

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Set the existing repository name to consume a team-managed ECR repository.
data "aws_ecr_repository" "existing" {
  count  = var.existing_ecr_repository_name == null ? 0 : 1
  region = var.aws_region
  name   = var.existing_ecr_repository_name
}

locals {
  ecr_repository_arn = var.existing_ecr_repository_name == null ? aws_ecr_repository.arbiter[0].arn : data.aws_ecr_repository.existing[0].arn
  ecr_repository_url = var.existing_ecr_repository_name == null ? aws_ecr_repository.arbiter[0].repository_url : data.aws_ecr_repository.existing[0].repository_url
  arbiter_image      = var.arbiter_image_digest == null ? "" : "${local.ecr_repository_url}@${var.arbiter_image_digest}"

  runtime_config = {
    region                       = var.aws_region
    registry                     = split("/", local.ecr_repository_url)[0]
    image                        = local.arbiter_image
    service_enabled              = var.arbiter_enabled
    smoke_enabled                = var.smoke_test_enabled
    service_command              = var.arbiter_command
    smoke_command                = var.smoke_test_command
    service_port                 = var.arbiter_port
    service_memory_mb            = var.arbiter_memory_mb
    smoke_cpus                   = var.smoke_test_cpus
    smoke_memory_mb              = var.smoke_test_memory_mb
    smoke_host_memory_reserve_mb = var.smoke_test_host_memory_reserve_mb
    smoke_host_disk_reserve_mb   = var.smoke_test_host_disk_reserve_mb
    smoke_tmpfs_mb               = var.smoke_test_tmpfs_mb
    smoke_pids_limit             = var.smoke_test_pids_limit
    smoke_timeout_seconds        = var.smoke_test_timeout_seconds
    environment = merge(var.container_environment, {
      AWS_REGION           = var.aws_region
      AWS_DEFAULT_REGION   = var.aws_region
      ALIGN_ENVIRONMENT    = var.environment
      ALIGN_LAKE_BUCKET    = var.lake_bucket_name
      ALIGN_DB_SECRET_ARN  = var.db_secret_arn
      ALIGN_LLM_SECRET_ARN = aws_secretsmanager_secret.llm_api_key.arn
    })
    smoke_environment = merge(var.smoke_test_environment, {
      ALIGN_ENVIRONMENT   = var.environment
      ALIGN_SMOKE_OFFLINE = "true"
    })
  }

  smoke_cron = var.smoke_test_enabled ? join("\n", [
    "SHELL=/bin/bash",
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "MAILTO=\"\"",
    "CRON_TZ=${var.smoke_test_timezone}",
    "${var.smoke_test_schedule} root /usr/bin/python3 /opt/align/run_container.py smoke",
    "",
  ]) : "# ALIGN smoke tests are disabled.\n"

  # Compress embedded assets to stay within EC2's user-data size limit.
  bootstrap_user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    runtime_bundle = base64gzip(jsonencode({
      config   = local.runtime_config
      runner   = file("${path.module}/scripts/run_container.py")
      cron     = local.smoke_cron
      timezone = var.smoke_test_timezone
    }))
  })
}

resource "aws_iam_role_policy" "ecr_pull" {
  name = "${local.name}-ecr-pull"
  role = aws_iam_role.compute.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AuthenticateToECR"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        # ECR does not support repository-level scope for this authentication API.
        Resource = "*"
      },
      {
        Sid      = "PullArbiterImage"
        Effect   = "Allow"
        Action   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
        Resource = [local.ecr_repository_arn]
      },
    ]
  })
}
