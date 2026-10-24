variable "project" {
  description = "Lowercase resource-name prefix."
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,19}$", var.project))
    error_message = "Use 2-20 lowercase letters, digits or hyphens, starting with a letter."
  }
}

variable "environment" {
  description = "Deployment environment (dev or prod)."
  type        = string
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be dev or prod."
  }
}

variable "aws_region" {
  description = "Region of the inherited AWS provider."
  type        = string
}

variable "deployment_suffix" {
  description = "Optional stable, unique name suffix for compute deployments in the same environment. Empty preserves existing names."
  type        = string
  default     = ""
  nullable    = false

  validation {
    condition     = var.deployment_suffix == "" || can(regex("^[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?$", var.deployment_suffix))
    error_message = "Leave deployment_suffix empty, or use 1-20 lowercase letters, digits or hyphens, starting and ending with a letter or digit."
  }
}

variable "subnet_id" {
  description = "EC2 subnet ID from network.public_subnet_ids; the current network design requires an explicit public IP for outbound access."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^subnet-([0-9a-f]{8}|[0-9a-f]{17})$", var.subnet_id))
    error_message = "Supply an existing subnet ID from the network module."
  }
}

variable "ec2_security_group_id" {
  description = "Existing EC2 security group ID from the network module."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^sg-([0-9a-f]{8}|[0-9a-f]{17})$", var.ec2_security_group_id))
    error_message = "Supply an existing EC2 security group ID from the network module."
  }
}

variable "lake_bucket_name" {
  description = "Data lake bucket name from the storage module, for runtime configuration."
  type        = string
  nullable    = false
}

variable "lake_bucket_arn" {
  description = "Data lake bucket ARN from the storage module, for IAM access restricted to the deployment account's bucket."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^arn:[a-z0-9-]+:s3:::[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.lake_bucket_arn))
    error_message = "Supply one S3 bucket ARN without wildcards or an object path."
  }

  validation {
    condition     = try(split(":", var.lake_bucket_arn)[5] == var.lake_bucket_name, false)
    error_message = "Lake bucket ARN must identify lake_bucket_name."
  }
}

variable "db_secret_arn" {
  description = "ARN of the database credentials secret supplied by the database module in the deployment account; never supply its secret value."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^arn:[a-z0-9-]+:secretsmanager:[a-z0-9-]+:[0-9]{12}:secret:[A-Za-z0-9/_+=.@!-]+$", var.db_secret_arn))
    error_message = "Supply one complete Secrets Manager secret ARN without wildcards."
  }

  validation {
    condition     = try(split(":", var.db_secret_arn)[3] == var.aws_region, false)
    error_message = "Database secret must be in aws_region for runtime retrieval."
  }
}

variable "db_secret_kms_key_arn" {
  description = "Customer-managed KMS key ARN encrypting the database secret in the deployment account, or null for aws/secretsmanager. The database owner must also ensure its key policy allows this role to decrypt."
  type        = string
  default     = null

  validation {
    condition = var.db_secret_kms_key_arn == null ? true : can(regex(
      "^arn:[a-z0-9-]+:kms:${var.aws_region}:[0-9]{12}:key/[A-Za-z0-9-]+$",
      var.db_secret_kms_key_arn,
    ))
    error_message = "Supply one KMS key ARN in aws_region without wildcards or aliases, or leave it null."
  }
}

# Database connection inputs will be finalized with the database owner once
# the RDS output interface and credentials-secret format are established.

variable "ami_id" {
  description = "Pinned AWS-provided standard Amazon Linux 2023 x86_64 AMI ID in aws_region."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^ami-([0-9a-f]{8}|[0-9a-f]{17})$", var.ami_id))
    error_message = "Supply a specific AMI ID; automatic latest-image selection is not configured."
  }
}

variable "instance_type" {
  description = "EC2 instance type chosen for this environment; it must support x86_64, EBS root disks, and HVM."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*\\.[a-z0-9]+$", var.instance_type))
    error_message = "Supply an EC2 instance type in family.size form."
  }
}

variable "root_volume_size" {
  description = "Root EBS volume size in GiB, chosen to accommodate the selected AMI and Docker workload."
  type        = number
  nullable    = false
  validation {
    condition     = var.root_volume_size > 0 && floor(var.root_volume_size) == var.root_volume_size
    error_message = "Root volume size must be a positive integer in GiB."
  }
}

variable "existing_ecr_repository_name" {
  description = "Existing ECR repository in aws_region, or null to create an environment-prefixed arbiter repository."
  type        = string
  default     = null

  validation {
    condition     = var.existing_ecr_repository_name == null ? true : can(regex("^[a-z0-9][a-z0-9._/-]*$", var.existing_ecr_repository_name))
    error_message = "Supply an ECR repository name without a registry URL, tag, or digest."
  }
}

variable "arbiter_image_digest" {
  description = "Published linux/amd64 image digest (sha256:...) from this ECR repository. Required only when enabling a workload."
  type        = string
  default     = null

  validation {
    condition     = var.arbiter_image_digest == null ? !(var.arbiter_enabled || var.smoke_test_enabled) : can(regex("^sha256:[0-9a-f]{64}$", var.arbiter_image_digest))
    error_message = "Enabling a workload requires a specific published sha256 image digest; tags such as latest are not accepted."
  }
}

variable "arbiter_enabled" {
  description = "Start the normal long-running arbiter. Keep false until its image and runtime contract are ready."
  type        = bool
  default     = false
  nullable    = false
}

variable "arbiter_memory_mb" {
  description = "Optional hard service memory limit in MiB, required when the dev service shares its host with offline smoke tests. Swap is capped to the same amount."
  type        = number
  default     = null

  validation {
    condition     = var.arbiter_memory_mb == null ? !(var.arbiter_enabled && var.smoke_test_enabled) : var.arbiter_memory_mb >= 6 && floor(var.arbiter_memory_mb) == var.arbiter_memory_mb
    error_message = "Supply an integer service memory limit of at least 6 MiB; it is required when service and smoke are both enabled."
  }
}

variable "arbiter_command" {
  description = "Arguments appended to the image ENTRYPOINT for regular use; empty uses the image CMD. No shell evaluation or secret values."
  type        = list(string)
  default     = []
  nullable    = false

  validation {
    condition     = alltrue([for argument in var.arbiter_command : try(length(regexall("\\x00", argument)) == 0, false)])
    error_message = "Command arguments cannot be null or contain NUL characters."
  }
}

variable "arbiter_port" {
  description = "Optional container port published on the same host port at 127.0.0.1 for SSM port forwarding. No public listener is created."
  type        = number
  default     = null

  validation {
    condition     = var.arbiter_port == null ? true : var.arbiter_port >= 1 && var.arbiter_port <= 65535 && floor(var.arbiter_port) == var.arbiter_port
    error_message = "Arbiter port must be an integer from 1 through 65535, or null."
  }
}

variable "container_environment" {
  description = "Additional non-secret environment variables. Never supply passwords, keys, or tokens; applications retrieve secrets using the managed ARN variables. Managed AWS/ALIGN settings take precedence."
  type        = map(string)
  default     = {}
  nullable    = false

  validation {
    condition = alltrue([for key, value in var.container_environment :
      can(regex("^[A-Za-z_][A-Za-z0-9_]*$", key)) && try(length(regexall("[\\r\\n\\x00]", value)) == 0, false)
    ])
    error_message = "Environment entries need valid names and non-null values without newlines or NUL characters."
  }
}

variable "smoke_test_enabled" {
  description = "Enable the nightly offline smoke job in dev, optionally alongside the regular service with explicit memory and host reserve budgets."
  type        = bool
  default     = false
  nullable    = false

  validation {
    condition     = !var.smoke_test_enabled || var.environment == "dev"
    error_message = "Nightly smoke tests are restricted to dev; never enable them on the production deployment."
  }
}

variable "smoke_test_environment" {
  description = "Additional non-secret offline test settings, separate from the service environment. Never supply credentials or secret values. Managed environment/offline markers take precedence."
  type        = map(string)
  default     = {}
  nullable    = false

  validation {
    condition = alltrue([for key, value in var.smoke_test_environment :
      can(regex("^[A-Za-z_][A-Za-z0-9_]*$", key)) && try(length(regexall("[\\r\\n\\x00]", value)) == 0, false)
    ])
    error_message = "Smoke environment entries need valid names and non-null values without newlines or NUL characters."
  }
}

variable "smoke_test_command" {
  description = "Smoke-test arguments appended to the image ENTRYPOINT, supplied by its owner. No shell evaluation or secret values."
  type        = list(string)
  default     = []
  nullable    = false

  validation {
    condition     = !var.smoke_test_enabled || length(var.smoke_test_command) > 0
    error_message = "An enabled smoke job requires an explicit test command so it cannot accidentally start the regular service."
  }

  validation {
    condition     = alltrue([for argument in var.smoke_test_command : try(length(regexall("\\x00", argument)) == 0, false)])
    error_message = "Command arguments cannot be null or contain NUL characters."
  }
}

variable "smoke_test_schedule" {
  description = "Daily five-field cron expression: numeric minute/hour followed by * * *, using smoke_test_timezone."
  type        = string
  default     = "0 3 * * *"
  nullable    = false

  validation {
    condition     = can(regex("^([0-5]?[0-9]) ([01]?[0-9]|2[0-3]) \\* \\* \\*$", var.smoke_test_schedule))
    error_message = "Use a daily cron schedule with minute 0-59, hour 0-23, and * * *; no additional commands or lines."
  }
}

variable "smoke_test_timezone" {
  description = "IANA timezone for the nightly schedule; checked against installed tzdata at bootstrap."
  type        = string
  default     = "America/Toronto"
  nullable    = false

  validation {
    condition     = can(regex("^[A-Za-z_+-]+(/[A-Za-z0-9_+-]+)*$", var.smoke_test_timezone))
    error_message = "Supply an IANA timezone name without whitespace or path traversal."
  }
}

variable "smoke_test_timeout_seconds" {
  description = "Maximum duration of smoke image preparation and execution before its container is removed."
  type        = number
  default     = 900
  nullable    = false

  validation {
    condition     = var.smoke_test_timeout_seconds > 0 && var.smoke_test_timeout_seconds < 86400 && floor(var.smoke_test_timeout_seconds) == var.smoke_test_timeout_seconds
    error_message = "Smoke timeout must be a positive integer below one day."
  }
}

variable "smoke_test_cpus" {
  description = "Measured CPU limit for the smoke container. Required before enabling the job."
  type        = number
  default     = null

  validation {
    condition     = var.smoke_test_cpus == null ? !var.smoke_test_enabled : var.smoke_test_cpus > 0
    error_message = "An enabled smoke job needs a positive CPU limit."
  }
}

variable "smoke_test_memory_mb" {
  description = "Measured hard memory limit in MiB for the smoke container. Required before enabling the job. Swap is capped to the same amount."
  type        = number
  default     = null

  validation {
    condition     = var.smoke_test_memory_mb == null ? !var.smoke_test_enabled : var.smoke_test_memory_mb >= 6 && floor(var.smoke_test_memory_mb) == var.smoke_test_memory_mb
    error_message = "An enabled smoke job needs an integer memory limit of at least 6 MiB."
  }
}

variable "smoke_test_host_memory_reserve_mb" {
  description = "Available host memory in MiB to preserve beyond the smoke budget. Required before enabling smoke; the runtime skips when measured headroom is insufficient."
  type        = number
  default     = null

  validation {
    condition     = var.smoke_test_host_memory_reserve_mb == null ? !var.smoke_test_enabled : var.smoke_test_host_memory_reserve_mb > 0 && floor(var.smoke_test_host_memory_reserve_mb) == var.smoke_test_host_memory_reserve_mb
    error_message = "An enabled smoke job needs an explicit positive integer host memory reserve in MiB."
  }
}

variable "smoke_test_host_disk_reserve_mb" {
  description = "Minimum available Docker filesystem space in MiB before smoke runs. Required before enabling smoke; this is a free-space guard, not a disk allocation."
  type        = number
  default     = null

  validation {
    condition     = var.smoke_test_host_disk_reserve_mb == null ? !var.smoke_test_enabled : var.smoke_test_host_disk_reserve_mb > 0 && floor(var.smoke_test_host_disk_reserve_mb) == var.smoke_test_host_disk_reserve_mb
    error_message = "An enabled smoke job needs an explicit positive integer host disk reserve in MiB."
  }
}

variable "smoke_test_tmpfs_mb" {
  description = "Size limit in MiB for the offline smoke container's writable /tmp; its root filesystem is read-only."
  type        = number
  default     = 32
  nullable    = false

  validation {
    condition     = var.smoke_test_tmpfs_mb > 0 && floor(var.smoke_test_tmpfs_mb) == var.smoke_test_tmpfs_mb
    error_message = "Smoke tmpfs size must be a positive integer in MiB."
  }

  validation {
    condition     = !var.smoke_test_enabled || try(var.smoke_test_tmpfs_mb <= var.smoke_test_memory_mb, false)
    error_message = "Enabled smoke tmpfs size cannot exceed its container memory limit."
  }
}

variable "smoke_test_pids_limit" {
  description = "Maximum number of processes/threads in the offline smoke container."
  type        = number
  default     = 64
  nullable    = false

  validation {
    condition     = var.smoke_test_pids_limit > 0 && floor(var.smoke_test_pids_limit) == var.smoke_test_pids_limit
    error_message = "Smoke PID limit must be a positive integer."
  }
}
