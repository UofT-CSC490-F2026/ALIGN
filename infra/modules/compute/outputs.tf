output "iam_role_arn" {
  description = "IAM role ARN for the pipeline EC2 instance."
  value       = aws_iam_role.compute.arn
}

output "llm_secret_arn" {
  description = "LLM API-key secret ARN; its value is populated separately and never exposed by Terraform."
  value       = aws_secretsmanager_secret.llm_api_key.arn
}

output "instance_id" {
  description = "EC2 instance ID for the ALIGN pipeline."
  value       = aws_instance.compute.id
}

output "ecr_repository_url" {
  description = "ECR repository URL for the separately built and published arbiter image."
  value       = local.ecr_repository_url
}

output "ecr_repository_arn" {
  description = "ECR repository ARN; the image publisher needs separate push permissions."
  value       = local.ecr_repository_arn
}
