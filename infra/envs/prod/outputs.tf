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

output "lake_bucket_name" {
  description = "S3 data lake bucket name."
  value       = module.storage.lake_bucket_name
}

output "lake_bucket_arn" {
  description = "S3 data lake bucket ARN."
  value       = module.storage.lake_bucket_arn
}

output "db_address" {
  description = "Private RDS PostgreSQL hostname."
  value       = module.database.db_address
}

output "db_port" {
  description = "PostgreSQL connection port."
  value       = module.database.db_port
}

output "db_name" {
  description = "PostgreSQL database name."
  value       = module.database.db_name
}

output "db_instance_identifier" {
  description = "RDS instance identifier."
  value       = module.database.db_instance_identifier
}

output "db_master_secret_arn" {
  description = "ARN of RDS-managed credentials."
  value       = module.database.db_master_secret_arn
}

output "compute_instance_id" {
  description = "SSM-managed EC2 instance for the prod application."
  value       = module.compute.instance_id
}

output "ecr_repository_url" {
  description = "ECR repository URL for the separately published application image."
  value       = module.compute.ecr_repository_url
}
