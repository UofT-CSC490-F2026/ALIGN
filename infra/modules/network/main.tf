locals {
  name = "${var.project}-${var.environment}"
  tags = {
    project   = var.project
    env       = var.environment
    ManagedBy = "Terraform"
  }
  public_subnets  = zipmap(var.availability_zones, var.public_subnet_cidrs)
  private_subnets = zipmap(var.availability_zones, var.private_subnet_cidrs)
}

# Provider configuration and account restrictions belong to the environment root.
resource "aws_vpc" "this" {
  region               = var.aws_region
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "${local.name}-vpc" })
}

resource "aws_internet_gateway" "this" {
  region = var.aws_region
  vpc_id = aws_vpc.this.id
  tags   = merge(local.tags, { Name = "${local.name}-igw" })
}

resource "aws_subnet" "public" {
  for_each = local.public_subnets

  region            = var.aws_region
  vpc_id            = aws_vpc.this.id
  availability_zone = each.key
  cidr_block        = each.value
  # Compute explicitly assigns a public IP for outbound SSM and API access.
  map_public_ip_on_launch = false
  tags                    = merge(local.tags, { Name = "${local.name}-public-${each.key}" })
}

resource "aws_subnet" "private" {
  for_each = local.private_subnets

  region                  = var.aws_region
  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.key
  cidr_block              = each.value
  map_public_ip_on_launch = false
  tags                    = merge(local.tags, { Name = "${local.name}-private-${each.key}" })
}

resource "aws_route_table" "public" {
  region = var.aws_region
  vpc_id = aws_vpc.this.id
  tags   = merge(local.tags, { Name = "${local.name}-public" })
}

resource "aws_route" "internet" {
  region                 = var.aws_region
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

# Only the implicit VPC-local route: RDS needs no NAT or public default route.
resource "aws_route_table" "private" {
  region = var.aws_region
  vpc_id = aws_vpc.this.id
  tags   = merge(local.tags, { Name = "${local.name}-private" })
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  region         = var.aws_region
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  for_each = aws_subnet.private

  region         = var.aws_region
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}

resource "aws_security_group" "ec2" {
  region      = var.aws_region
  name        = "${local.name}-ec2"
  description = "SSH from approved team addresses; outbound access for agent workloads"
  vpc_id      = aws_vpc.this.id
  tags        = merge(local.tags, { Name = "${local.name}-ec2" })
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each = var.allowed_ssh_cidrs

  region            = var.aws_region
  security_group_id = aws_security_group.ec2.id
  description       = "Team SSH"
  ip_protocol       = "tcp"
  from_port         = 22
  to_port           = 22
  cidr_ipv4         = each.value
  tags              = local.tags
}

resource "aws_vpc_security_group_egress_rule" "ec2" {
  region            = var.aws_region
  security_group_id = aws_security_group.ec2.id
  description       = "Outbound dependencies, S3, model APIs and database connections"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
  tags              = local.tags
}

resource "aws_security_group" "rds" {
  region      = var.aws_region
  name        = "${local.name}-rds"
  description = "Database connections from the application EC2 security group only"
  vpc_id      = aws_vpc.this.id
  tags        = merge(local.tags, { Name = "${local.name}-rds" })
}

# No RDS egress rule is needed for replies: security groups are stateful.
resource "aws_vpc_security_group_ingress_rule" "database" {
  region                       = var.aws_region
  security_group_id            = aws_security_group.rds.id
  description                  = "Database access from EC2"
  ip_protocol                  = "tcp"
  from_port                    = var.db_port
  to_port                      = var.db_port
  referenced_security_group_id = aws_security_group.ec2.id
  tags                         = local.tags
}
