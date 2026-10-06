variable "aws_account_id" {
  description = "Account allowed to receive the bootstrap resources."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "AWS account ID must contain exactly 12 digits."
  }
}

variable "aws_region" {
  description = "AWS region for state storage and locking."
  type        = string
}

variable "project" {
  description = "Lowercase project prefix for resource names."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,19}$", var.project))
    error_message = "Use 2-20 lowercase letters, digits, or hyphens, starting with a letter."
  }
}
