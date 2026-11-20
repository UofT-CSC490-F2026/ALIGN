output "backup_bucket_names" {
  description = "Backup bucket name per environment."
  value       = { for env, b in aws_s3_bucket.backup : env => b.id }
}

output "backup_bucket_arns" {
  description = "Backup bucket ARN per environment."
  value       = { for env, b in aws_s3_bucket.backup : env => b.arn }
}
