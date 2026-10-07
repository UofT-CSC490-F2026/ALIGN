variable "project" {
  description = "Lowercase resource-name prefix."
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,19}$", var.project))
    error_message = "Use 2-20 lowercase letters, digits or hyphens, starting with a letter."
  }
}

variable "environment" {
  description = "Environment isolated by this VPC."
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

variable "vpc_cidr" {
  description = "Canonical IPv4 VPC CIDR, with prefix /16 through /28."
  type        = string
  validation {
    condition = try(
      cidrnetmask(var.vpc_cidr) != "" &&
      tonumber(split("/", var.vpc_cidr)[1]) >= 16 &&
      tonumber(split("/", var.vpc_cidr)[1]) <= 28 &&
      split("/", var.vpc_cidr)[0] == cidrhost(var.vpc_cidr, 0), false
    )
    error_message = "Supply a canonical IPv4 VPC network between /16 and /28."
  }
}

variable "availability_zones" {
  description = "Two distinct standard Availability Zones; subnet lists follow this order."
  type        = list(string)
  validation {
    condition = length(var.availability_zones) == 2 && length(distinct(var.availability_zones)) == 2 && alltrue([
      for az in var.availability_zones : can(regex("^${var.aws_region}[a-z]$", az))
    ])
    error_message = "Supply two distinct Availability Zones in aws_region."
  }
}

variable "public_subnet_cidrs" {
  description = "Two public IPv4 networks ordered to match availability_zones."
  type        = list(string)
  validation {
    condition = length(var.public_subnet_cidrs) == 2 && alltrue([
      for cidr in var.public_subnet_cidrs : try(
        cidrnetmask(cidr) != "" &&
        tonumber(split("/", cidr)[1]) >= tonumber(split("/", var.vpc_cidr)[1]) &&
        tonumber(split("/", cidr)[1]) <= 28 &&
        split("/", cidr)[0] == cidrhost(cidr, 0) &&
        cidrhost("${cidrhost(cidr, 0)}/${split("/", var.vpc_cidr)[1]}", 0) == cidrhost(var.vpc_cidr, 0), false
      )
    ])
    error_message = "Supply two canonical IPv4 subnets within the VPC, no smaller than /28."
  }
}

variable "private_subnet_cidrs" {
  description = "Two private IPv4 networks ordered to match availability_zones."
  type        = list(string)
  validation {
    condition = length(var.private_subnet_cidrs) == 2 && alltrue([
      for cidr in var.private_subnet_cidrs : try(
        cidrnetmask(cidr) != "" &&
        tonumber(split("/", cidr)[1]) >= tonumber(split("/", var.vpc_cidr)[1]) &&
        tonumber(split("/", cidr)[1]) <= 28 &&
        split("/", cidr)[0] == cidrhost(cidr, 0) &&
        cidrhost("${cidrhost(cidr, 0)}/${split("/", var.vpc_cidr)[1]}", 0) == cidrhost(var.vpc_cidr, 0), false
      )
    ])
    error_message = "Supply two canonical IPv4 subnets within the VPC, no smaller than /28."
  }
  validation {
    condition = alltrue(flatten([
      for i, a in concat(var.public_subnet_cidrs, var.private_subnet_cidrs) : [
        for j, b in concat(var.public_subnet_cidrs, var.private_subnet_cidrs) :
        i == j || try(
          cidrhost("${cidrhost(a, 0)}/${min(tonumber(split("/", a)[1]), tonumber(split("/", b)[1]))}", 0) !=
          cidrhost("${cidrhost(b, 0)}/${min(tonumber(split("/", a)[1]), tonumber(split("/", b)[1]))}", 0), false
        )
      ]
    ]))
    error_message = "All public and private subnet networks must be non-overlapping."
  }
}

variable "allowed_ssh_cidrs" {
  description = "Team public IPv4 addresses as /32 networks. Empty disables all SSH ingress."
  type        = set(string)
  default     = []
  validation {
    condition = alltrue([
      for cidr in var.allowed_ssh_cidrs : try(
        cidrnetmask(cidr) != "" && split("/", cidr)[1] == "32", false
      )
    ])
    error_message = "SSH sources must be individual IPv4 /32 addresses, never broad networks."
  }
}

variable "db_port" {
  description = "TCP database port reachable only from the EC2 security group."
  type        = number
  default     = 5432
  validation {
    condition     = var.db_port >= 1 && var.db_port <= 65535 && floor(var.db_port) == var.db_port
    error_message = "Database port must be an integer from 1 through 65535."
  }
}
