output "lake_bucket_name" {
  description = "Data lake bucket name."
  value       = aws_s3_bucket.lake.id
}

output "lake_bucket_arn" {
  description = "Data lake bucket ARN."
  value       = aws_s3_bucket.lake.arn
}

output "backup_bucket_name" {
  description = "Backup bucket name."
  value       = data.aws_s3_bucket.backup.id
}

output "backup_bucket_arn" {
  description = "Backup bucket ARN."
  value       = data.aws_s3_bucket.backup.arn
}
