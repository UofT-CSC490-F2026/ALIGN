output "db_address" {
  description = "Private RDS hostname (without port)."
  value       = aws_db_instance.this.address
}

output "db_port" {
  description = "RDS PostgreSQL TCP port."
  value       = aws_db_instance.this.port
}

output "db_name" {
  description = "Initial PostgreSQL database name."
  value       = aws_db_instance.this.db_name
}

output "db_instance_identifier" {
  description = "RDS instance identifier for AWS CLI/console operations."
  value       = aws_db_instance.this.identifier
}

output "db_master_secret_arn" {
  description = "ARN of the RDS-managed admin credential in Secrets Manager; not the password."
  value       = aws_db_instance.this.master_user_secret[0].secret_arn
}

output "db_subnet_group_name" {
  description = "RDS DB subnet group used for private placement."
  value       = aws_db_subnet_group.this.name
}
