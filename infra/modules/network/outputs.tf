output "vpc_id" {
  description = "VPC containing this environment."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs in availability_zones input order."
  value       = [for az in var.availability_zones : aws_subnet.public[az].id]

  # Instances using these subnets may need internet access during first boot.
  depends_on = [
    aws_route.internet,
    aws_route_table_association.public,
  ]
}

output "private_subnet_ids" {
  description = "Private subnet IDs in availability_zones input order; pass to the RDS subnet group."
  value       = [for az in var.availability_zones : aws_subnet.private[az].id]

  # Consumers must wait until the private routing policy is associated.
  depends_on = [aws_route_table_association.private]
}

output "ec2_security_group_id" {
  description = "Attach to the EC2 instance; public IP assignment belongs to compute."
  value       = aws_security_group.ec2.id

  # Group creation alone does not make its separately managed rules ready.
  depends_on = [
    aws_vpc_security_group_ingress_rule.ssh,
    aws_vpc_security_group_egress_rule.ec2,
  ]
}

output "rds_security_group_id" {
  description = "Attach to RDS with publicly_accessible=false."
  value       = aws_security_group.rds.id

  # Database consumers need the EC2 access rule, not only an empty group.
  depends_on = [aws_vpc_security_group_ingress_rule.database]
}
