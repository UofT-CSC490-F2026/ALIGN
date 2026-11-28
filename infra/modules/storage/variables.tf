variable "project" {
  description = "Lowercase project prefix for resource names."
  type        = string
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
  description = "AWS region of the lake bucket."
  type        = string
}

variable "aws_account_id" {
  description = "Account ID, used to keep bucket names globally unique."
  type        = string
}

variable "backup_bucket_name" {
  description = "Backup bucket created by infra/backup; replication destination for the lake."
  type        = string
}

variable "lake_force_destroy" {
  description = "Allow terraform destroy to delete a non-empty lake bucket."
  type        = bool
  default     = false
}
