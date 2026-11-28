variable "aws_account_id" {
  description = "Account allowed to receive the backup resources."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "AWS account ID must contain exactly 12 digits."
  }
}

variable "aws_region" {
  description = "AWS region for the backup buckets."
  type        = string
}

variable "project" {
  description = "Lowercase project prefix for resource names."
  type        = string
}

variable "environments" {
  description = "Environments that get a backup bucket."
  type        = set(string)
  default     = ["dev", "prod"]
}
