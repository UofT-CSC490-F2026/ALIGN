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
  description = "Allowed AWS account."
  type        = string
  validation {
    condition     = var.aws_account_id == "183754987292"
    error_message = "This backend belongs to account 183754987292."
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
