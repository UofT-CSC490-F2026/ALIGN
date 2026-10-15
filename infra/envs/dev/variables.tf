variable "project" {
  description = "Resource prefix."
  type        = string
}

variable "environment" {
  description = "Environment for this root."
  type        = string
  validation {
    condition     = var.environment == "dev"
    error_message = "This root only deploys dev; use the matching tfvars file."
  }
}

variable "aws_account_id" {
  description = "Selected deployment account. Each account requires isolated state; backend authentication is configured separately."
  type        = string
  validation {
    condition     = contains(["183754987292", "710809468662"], var.aws_account_id)
    error_message = "Select an approved ALIGN account: 183754987292 or 710809468662."
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
