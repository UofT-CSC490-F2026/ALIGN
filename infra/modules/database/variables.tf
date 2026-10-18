variable "project" {
  description = "Lowercase project prefix, e.g. align."
  type        = string

  validation {
    condition = (
      can(regex("^[a-z][a-z0-9-]{1,19}$", var.project)) &&
      !endswith(var.project, "-") && !strcontains(var.project, "--")
    )
    error_message = "Project must be 2-20 lowercase letters, digits or hyphens, starting with a letter, with no trailing or consecutive hyphens."
  }
}

variable "environment" {
  description = "Terraform environment: dev or prod."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be dev or prod."
  }
}

variable "aws_region" {
  description = "Region configured by the parent Terraform root."
  type        = string
}

variable "private_subnet_ids" {
  description = "Two private subnet IDs in distinct AZs, from module.network.private_subnet_ids."
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) == 2 && length(distinct(var.private_subnet_ids)) == 2
    error_message = "Pass two distinct private subnet IDs from the network module."
  }
}

variable "rds_security_group_id" {
  description = "Existing RDS security group ID from module.network.rds_security_group_id."
  type        = string
}

variable "db_name" {
  description = "PostgreSQL database created by RDS (not the instance identifier)."
  type        = string
  default     = "align"

  validation {
    condition     = can(regex("^[A-Za-z][A-Za-z0-9_]{0,62}$", var.db_name))
    error_message = "db_name must start with a letter and contain up to 63 letters, digits or underscores."
  }
}

variable "master_username" {
  description = "Administrator username; RDS manages the password in Secrets Manager."
  type        = string
  default     = "align_admin"

  validation {
    condition     = can(regex("^[A-Za-z][A-Za-z0-9_]{0,15}$", var.master_username))
    error_message = "master_username must start with a letter and contain up to 16 letters, digits or underscores."
  }
}

variable "db_port" {
  description = "Port must match module.network's db_port setting."
  type        = number
  default     = 5432

  validation {
    condition     = var.db_port >= 1 && var.db_port <= 65535 && floor(var.db_port) == var.db_port
    error_message = "db_port must be an integer from 1 to 65535."
  }
}

variable "postgres_major_version" {
  description = "Supported RDS PostgreSQL major version."
  type        = string
  default     = "16"
}

variable "instance_class" {
  description = "RDS instance class. Verify price and regional availability before apply."
  type        = string
  default     = "db.t4g.micro"
}

variable "allocated_storage" {
  description = "Allocated gp3 storage in GiB."
  type        = number
  default     = 20

  validation {
    condition     = var.allocated_storage >= 20 && floor(var.allocated_storage) == var.allocated_storage
    error_message = "allocated_storage must be an integer of at least 20 GiB."
  }
}

variable "backup_retention_days" {
  description = "Days to retain RDS automated backups (1-35)."
  type        = number
  default     = 7

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 35 && floor(var.backup_retention_days) == var.backup_retention_days
    error_message = "backup_retention_days must be a whole number from 1 to 35."
  }
}

variable "multi_az" {
  description = "Deploy a Multi-AZ standby; disabled for low-cost dev."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Protect RDS against accidental deletion; explicitly disable before planned teardown."
  type        = bool
  default     = true
}

variable "final_snapshot_identifier" {
  description = "Optional final snapshot name. Change if a snapshot with the default name already exists."
  type        = string
  default     = null

  validation {
    condition = var.final_snapshot_identifier == null ? true : (
      can(regex("^[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9]$", var.final_snapshot_identifier)) &&
      !strcontains(var.final_snapshot_identifier, "--") &&
      length(var.final_snapshot_identifier) <= 255
    )
    error_message = "Final snapshot name must start with a letter, end with a letter/digit, have no consecutive hyphens and be <=255 characters."
  }
}
