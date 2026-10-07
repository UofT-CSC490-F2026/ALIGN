output "vpc_id" {
  description = "VPC containing this environment."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs in availability_zones input order."
  value       = [for az in var.availability_zones : aws_subnet.public[az].id]
}

output "private_subnet_ids" {
  description = "Private subnet IDs in availability_zones input order; pass to the RDS subnet group."
  value       = [for az in var.availability_zones : aws_subnet.private[az].id]
}

output "ec2_security_group_id" {
  description = "Attach to the EC2 instance; public IP assignment belongs to compute."
  value       = aws_security_group.ec2.id
}

output "rds_security_group_id" {
  description = "Attach to RDS with publicly_accessible=false."
  value       = aws_security_group.rds.id
}
