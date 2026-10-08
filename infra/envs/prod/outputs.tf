output "vpc_id" {
  description = "VPC ID for downstream modules."
  value       = module.network.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs in availability_zones order; compute must explicitly assign a public IP for internet access."
  value       = module.network.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs in availability_zones order for the database subnet group."
  value       = module.network.private_subnet_ids
}

output "ec2_security_group_id" {
  description = "EC2 security group ID; no SSH ingress when allowed_ssh_cidrs is empty."
  value       = module.network.ec2_security_group_id
}

output "rds_security_group_id" {
  description = "RDS security group ID allowing database ingress only from the EC2 security group."
  value       = module.network.rds_security_group_id
}
