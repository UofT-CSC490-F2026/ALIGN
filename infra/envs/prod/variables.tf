variable "db_snapshot_identifier" {
  description = "Required restore source for the already-restored production database."
  type        = string
  nullable    = false
  validation {
    condition     = trimspace(var.db_snapshot_identifier) != ""
    error_message = "Prod requires its existing restore snapshot identifier; use prod.tfvars."
  }
}

variable "db_deletion_protection" {
  description = "Disable only during an explicitly approved teardown."
  type        = bool
  default     = true
}

variable "db_final_snapshot_identifier" {
  description = "Unique final snapshot name for an approved teardown."
  type        = string
  default     = null
}

variable "lake_force_destroy" {
  description = "Allows deletion of all prod lake object versions; only for approved teardown after backup verification."
  type        = bool
  default     = false
}

variable "project" {
  description = "Resource prefix."
  type        = string
}

variable "environment" {
  description = "Environment for this root."
  type        = string
  validation {
    condition     = var.environment == "prod"
    error_message = "This root only deploys prod; use the matching tfvars file."
  }
}

variable "aws_account_id" {
  description = "Deployment account matching this environment's backend."
  type        = string
  validation {
    condition     = var.aws_account_id == "710809468662"
    error_message = "Use AWS account 710809468662."
  }
}

variable "aws_region" {
  description = "Deployment region."
  type        = string
  validation {
    condition     = var.aws_region == "us-east-2"
    error_message = "ALIGN environments are deployed in us-east-2."
  }
}

variable "vpc_cidr" {
  description = "Environment VPC CIDR."
  type        = string
}

variable "availability_zones" {
  description = "Subnet ordering across two zones."
  type        = list(string)
}

variable "public_subnet_cidrs" {
  description = "Public networks in zone order."
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "Private networks in zone order."
  type        = list(string)
}

variable "allowed_ssh_cidrs" {
  description = "Optional SSH fallback /32 addresses. Keep empty for Session Manager access."
  type        = set(string)
}

variable "db_port" {
  description = "Database TCP port."
  type        = number
}
