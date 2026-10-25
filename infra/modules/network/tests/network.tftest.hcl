mock_provider "aws" {}

variables {
  project              = "align"
  environment          = "dev"
  aws_region           = "us-east-2"
  vpc_cidr             = "10.10.0.0/16"
  availability_zones   = ["us-east-2a", "us-east-2b"]
  public_subnet_cidrs  = ["10.10.1.0/24", "10.10.2.0/24"]
  private_subnet_cidrs = ["10.10.11.0/24", "10.10.12.0/24"]
  allowed_ssh_cidrs    = ["198.51.100.10/32", "203.0.113.20/32"]
}

run "network_contract" {
  command = apply

  assert {
    condition     = aws_route.internet.gateway_id == aws_internet_gateway.this.id && aws_route.internet.destination_cidr_block == "0.0.0.0/0"
    error_message = "Public internet traffic must route through this VPC's internet gateway."
  }
  assert {
    condition     = alltrue([for s in aws_subnet.private : !s.map_public_ip_on_launch]) && length(aws_subnet.private) == 2
    error_message = "RDS requires two private subnets without automatic public IP assignment."
  }
  assert {
    condition     = alltrue([for a in aws_route_table_association.private : a.route_table_id == aws_route_table.private.id])
    error_message = "Private subnets must use the private route table."
  }
  assert {
    condition     = alltrue([for a in aws_route_table_association.public : a.route_table_id == aws_route_table.public.id])
    error_message = "Public subnets must use the public route table."
  }
  assert {
    condition     = length(aws_vpc_security_group_ingress_rule.ssh) == 2 && alltrue([for r in aws_vpc_security_group_ingress_rule.ssh : r.from_port == 22 && r.to_port == 22 && r.ip_protocol == "tcp" && contains(var.allowed_ssh_cidrs, r.cidr_ipv4)])
    error_message = "SSH must allow only the supplied team addresses on TCP 22."
  }
  assert {
    condition     = aws_vpc_security_group_ingress_rule.database.referenced_security_group_id == output.ec2_security_group_id && aws_vpc_security_group_ingress_rule.database.security_group_id == output.rds_security_group_id && aws_vpc_security_group_ingress_rule.database.from_port == 5432 && aws_vpc_security_group_ingress_rule.database.to_port == 5432 && aws_vpc_security_group_ingress_rule.database.cidr_ipv4 == null
    error_message = "Database ingress must reference EC2's security group, not a CIDR."
  }
  assert {
    condition     = output.public_subnet_ids == [aws_subnet.public["us-east-2a"].id, aws_subnet.public["us-east-2b"].id] && output.private_subnet_ids == [aws_subnet.private["us-east-2a"].id, aws_subnet.private["us-east-2b"].id]
    error_message = "Subnet outputs must preserve input Availability Zone order."
  }
}

run "ssh_disabled_by_empty_allowlist" {
  command = plan
  variables { allowed_ssh_cidrs = [] }
  assert {
    condition     = length(aws_vpc_security_group_ingress_rule.ssh) == 0
    error_message = "An empty allowlist must create no SSH ingress rules."
  }
}

run "reject_open_ssh" {
  command = plan
  variables { allowed_ssh_cidrs = ["0.0.0.0/0"] }
  expect_failures = [var.allowed_ssh_cidrs]
}

run "reject_overlapping_subnets" {
  command = plan
  variables { private_subnet_cidrs = ["10.10.1.0/25", "10.10.12.0/24"] }
  expect_failures = [var.private_subnet_cidrs]
}

run "reject_subnet_outside_vpc" {
  command = plan
  variables { private_subnet_cidrs = ["10.20.11.0/24", "10.10.12.0/24"] }
  expect_failures = [var.private_subnet_cidrs]
}

run "reject_invalid_database_port" {
  command = plan
  variables { db_port = 65536 }
  expect_failures = [var.db_port]
}

run "reject_duplicate_availability_zones" {
  command = plan
  variables { availability_zones = ["us-east-2a", "us-east-2a"] }
  expect_failures = [var.availability_zones]
}

run "prod_names_are_isolated" {
  command = plan
  variables { environment = "prod" }
  assert {
    condition     = aws_vpc.this.tags.Name == "align-prod-vpc" && aws_security_group.ec2.name == "align-prod-ec2" && aws_security_group.rds.name == "align-prod-rds"
    error_message = "Resource names must distinguish production from development."
  }
}
