output "state_bucket_name" {
  description = "Shared Terraform state bucket; never use it for application data."
  value       = aws_s3_bucket.state.id
}

output "lock_table_name" {
  description = "DynamoDB table used by the S3 backends for legacy-compatible locking."
  value       = aws_dynamodb_table.locks.name
}

output "aws_region" {
  description = "Region used by all backends."
  value       = var.aws_region
}
